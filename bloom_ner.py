import datetime
import hashlib
import itertools
import os
import random
import re
import time
import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import logging
import torch
import requests

def example2string(example, ner_tag_id, begin_tag, end_tag, tagged=True):
    # if ner_tag_id = 3 and 3 stands for LOC, beginning tag = @@ and ending tag = ##
    # and the example is {'id': 0, 'words': ['I', 'love', 'Paris', 'and', 'Berlin'], 'ner_tags': [0, 0, 3, 0, 3]}
    # the returned string will be 'I love @@Paris## and @@Berlin##'
    words = example['words' if 'words' in example else 'tokens']
    ner_tags = example['ner_tags']
    string = ''
    for i, (word, ner_tag) in enumerate(zip(words, ner_tags)):
        if tagged and ner_tag == ner_tag_id and (ner_tags[i-1] != ner_tag_id if i > 0 else True):
            string += begin_tag
        string += word
        if tagged and ner_tag == ner_tag_id and (ner_tags[i+1] != ner_tag_id if i < len(ner_tags)-1 else True):
            string += end_tag
        string += ' '
    return string.strip()
    

def sentences_with_most_occurences(dataset, ner_tag_id, n):
    counts = [e['ner_tags'].count(ner_tag_id) for e in dataset['train']]
    return sorted(range(len(counts)), key=lambda i: counts[i])[-n:]

def sentences_with_most_common_words(dataset, example, ner_tag_id, n):
    ref_words = example['words' if 'words' in example else 'tokens']
    counts = [len(set(e['words']).intersection(set(ref_words))) for e in dataset['train']]
    res = sorted(range(len(counts)), key=lambda i: counts[i])[-n:]
    return res

def sentences_with_closest_tf_idf(dataset, example, ner_tag_id, n):
    tokenized_examples = [e['words' if 'words' in e else 'tokens'] for e in dataset['train']]
    tokenized_examples.append(example['words' if 'words' in example else 'tokens'])
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf.fit(tokenized_examples)
    tfidf_matrix = tfidf.transform(tokenized_examples)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    res = sorted(range(len(similarities[0])), key=lambda i: similarities[0][i])[-n:]
    return res

criteria = {
    'most_occurences' : sentences_with_most_occurences,
    'most_common_words' : sentences_with_most_common_words,
    'closest_tf_idf' : sentences_with_closest_tf_idf
}

def make_prompts(dataset, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag, n_few_shots, criterion):
    #this function takes an example and a ner tag and returns a prompt in english
    keywords = prompt_keywords[language]
    few_shots_for_all = []
    num_prompts = len(dataset['test'])
    if criterion == 'random':
        for i in range(num_prompts):
            few_shots_for_all.append(random.sample(range(len(dataset['train'])), n_few_shots))
    elif criterion == 'most_occurences':
        few_shots_for_all = [sentences_with_most_occurences(dataset, ner_tag_id, n_few_shots)] * num_prompts
    elif criterion == 'closest_tf_idf':
        #get the k nearest sentences in the training set
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        transformed_train = tfidf.fit_transform([e['words' if 'words' in e else 'tokens'] for e in dataset['train']])
        transformed_test = tfidf.transform([e['words' if 'words' in e else 'tokens'] for e in dataset['test']])
        similarities = cosine_similarity(transformed_test, transformed_train)
        few_shots_for_all = [sorted(range(len(similarities[i])), key=lambda j: similarities[i][j])[-n_few_shots:] for i in range(num_prompts)]

        
    prompts = []
    for i in range(num_prompts):
        example = dataset['test'][i]
        prompt = keywords['first_sentence'].format(keywords['domains_jobs'][domain], keywords['ner_tags'][ner_tag])
        few_shots= few_shots_for_all[i]
        random.shuffle(few_shots)
        for i in few_shots:
            prompt+= keywords['input_intro']+example2string(dataset['train'][i], ner_tag_id, begin_tag, end_tag, tagged=False)+'\n'
            prompt+= keywords['output_intro']+example2string(dataset['train'][i], ner_tag_id, begin_tag, end_tag, tagged=True)+'\n'
        prompt+= keywords['last_sentence'].format(keywords['ner_tags'][ner_tag], begin_tag, end_tag)
        prompt+= keywords['input_intro']+example2string(example, ner_tag_id, begin_tag, end_tag, tagged=False)+'\n'
        prompt+= keywords['output_intro']
        prompts.append(prompt)
    return prompts


prompt_keywords = {
    'en' : {
            'first_sentence' : "I am an expert {}, I can identify mentions of {} in a sentence. I can also format them. Here are some examples of sentences I can handle:\n",
            'last_sentence' : "Imitate me. Identify the mentions of {} in the following sentence, by putting \"{}\" in front and a \"{}\" behind the mention in the following sentence.\n",
            'domains_jobs' : {
                'clinical' : "clinician",
                'general' : "linguist"
            },
            'ner_tags' : {
                'PER' : "person names",
                'DISO' : "disorders",
                'LOC' : "places"
            },
            'input_intro' : "Input: ",
            'output_intro' : "Output: ",
        }
    ,
    'fr' : {
        'first_sentence' : "Je suis un {} expert, je sais identifier les mentions des {} dans une phrase. Je peux aussi les mettre en forme. Voici quelques exemples de phrases que je peux traiter :\n",
        #'last_sentence' : "Imite-moi. Identifie les mentions de {} dans la phrase suivante, en mettant \"{}\" devant et un \"{}\" derrière la mention dans la phrase suivante.\n",
        'last_sentence':"",
        'domains_jobs' : {
            'clinical' : "clinicien",
            'general' : "linguiste"
        },
        'ner_tags' : {
            'PER' : "noms de personnes",
            'DISO' : "maladies et symptômes",
            'LOC' : "lieux"
        },
        'input_intro' : "Entrée : ",
        'output_intro' : "Sortie : ",
    }
}


args = argparse.ArgumentParser()
args.add_argument("--language", type=str, default="fr", help="language of the dataset")
args.add_argument("--domain", type=str, default="general", help="domain of the dataset")
args.add_argument("--ner_tag", type=str, help="ner tag to evaluate")
args.add_argument("--begin_tag", type=str, default="@@")
args.add_argument("--end_tag", type=str, default="##")
args.add_argument("--n_few_shot", type=int, default=5)
args.add_argument("--model_name", type=str, default="bigscience/bloom-7b1")
args.add_argument("--batch_size", type=int, default=2)
args.add_argument("--criterion", type=str, default="closest_tf_idf")
args.add_argument("--overwrite_prompt_cache", action="store_true")
args.add_argument('--top_p', type=float, nargs='+', default=[0.9])
args.add_argument('--top_k', type=int, nargs='+', default=[10])
args.add_argument('--temperature', type=float, nargs='+', default=[0.7])
args.add_argument('--api_inference', action="store_true")
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom_ner")

if args.domain == 'general':
    dataset_name = 'Jean-Baptiste/wikiner_fr'
    dataset = datasets.load_dataset(dataset_name)
    dataset['train'] = [example for example in dataset['train'] if len(example['tokens']) < 40]
    tag_to_id = {"O":0,"LOC":1,"PER":2,"FAC":3,"ORG":4}
    ner_tag = args.ner_tag if args.ner_tag else 'PER'
else :
    dataset_name = 'meczifho/QuaeroFrenchMed'
    dataset = datasets.load_dataset(dataset_name,'MEDLINE')
    tag_to_id = {"O":0,"ANAT":1,"LIVB":2,"DISO":3,"PROC":4,"CHEM":5,"GEOG":6,"PHYS":7,"PHEN":8,"OBJC":9,"DEVI":10}    
    ner_tag = args.ner_tag if args.ner_tag else 'DISO'

time_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = 'hyp_search_'+time_date
os.mkdir(folder_name)

#loop over all combinations of top_p, top_k and temperature
for (top_p, top_k, temp) in itertools.product(args.top_p, args.top_k, args.temperature):
    time_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = open(folder_name+'/log_'+time_date+'.txt','w')
    logfile.write('language: '+args.language+'\n')
    logfile.write('domain: '+args.domain+'\n')
    logfile.write('ner_tag: '+ner_tag+'\n')
    logfile.write('begin_tag: '+args.begin_tag+'\n')
    logfile.write('end_tag: '+args.end_tag+'\n')
    logfile.write('n_few_shot: '+str(args.n_few_shot)+'\n')
    logfile.write('model_name: '+args.model_name+'\n')
    logfile.write('criterion: '+args.criterion+'\n')
    logfile.write('top_p:' +str(top_p)+'\n')
    logfile.write('top_k:' +str(top_k)+'\n')
    logfile.write('temperature:' +str(temp)+'\n')
    logfile.write('='*50+'\n')

    tp_sum = 0
    relevant_sum = 0
    retrieved_sum = 0

    assert args.criterion in criteria.keys(), "criterion must be in "+str(criteria.keys())
    params = dataset_name+args.language+args.domain+ner_tag+args.begin_tag+args.end_tag+str(args.n_few_shot)+args.criterion
    hash_object = hashlib.md5(params.encode())
    if os.path.exists('prompts_'+hash_object.hexdigest()+'.txt') and not args.overwrite_prompt_cache:
        logger.info("Loading prompts...")
        with open('prompts_'+hash_object.hexdigest()+'.txt', 'r') as f:
            prompts = f.read().split('='*50)
        prompts = prompts[:-1]
        logger.info("Loaded prompts.")
    else:
        logger.info("Making prompts...")
        prompts = make_prompts(
            dataset, 
            ner_tag, 
            tag_to_id[ner_tag], 
            args.language, 
            args.domain, 
            args.begin_tag, 
            args.end_tag, 
            args.n_few_shot,
            args.criterion,
        )
        
        #cache prompts
        with open('prompts_'+hash_object.hexdigest()+'.txt', 'w') as f:
            for prompt in prompts:
                f.write(prompt+'='*50)
                
    
    if args.api_inference or args.model_name == "bigscience/bloom":
        #use huggingface inference API
        logger.info("Generating...")
        API_URL = "https://api-inference.huggingface.co/models/"+args.model_name
        headers = {"Authorization": "Bearer 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"}
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            try:
                return response.json()
            except:
                return {"error":response.text}
        outputs = []
        for i in tqdm(range(0, len(prompts))):
            output = query({"inputs":prompts[i],"parameters":{"top_p":top_p,"top_k":top_k,"temperature":temp, "return_full_text":False}})
            if 'error' in output:
                logger.info("Error: "+output['error'])
                logger.info("Prompt: "+prompts[i])
                if 'Rate limit' in output['error']:
                    logger.info("Rate limit exceeded. Waiting 10 minutes...")
                    nb_retries = 0
                    while 'error' in output and nb_retries < 10:
                        time.sleep(600)
                        logger.info("Retrying...")
                        output = query({"inputs":prompts[i],"parameters":{"top_p":top_p,"top_k":top_k,"temperature":temp, "return_full_text":False, "wait_for_model":True}})
                        nb_retries += 1
                    if 'error' in output:
                        logger.info("Rate limit exceeded. Stopping...")
                        break
                else:
                    logger.info("Skipping...")
                    continue
            outputs.append(output[0]['generated_text'])               
    else:
        logger.info("Tokenizing...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Generating...")
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        model.eval()

        outputs = []
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            input_ids_batch = input_ids[i:i+args.batch_size].to(device)
            output = model.generate(input_ids_batch, max_new_tokens=40, do_sample=True, top_p=top_p, top_k=top_k, temperature=temp)
            output = output[:,input_ids_batch.size(1):]
            outputs += output.tolist()
            
        logger.info("Decoding...")
        outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    logger.info("Evaluating...")
    targets = [example2string(dataset['test'][i], tag_to_id[ner_tag], args.begin_tag, args.end_tag, tagged=True) for i in range(len(dataset['test']))]
    for target, o in tqdm(zip(targets, outputs)):
        prediction = o.split('\n')[0]
        target = target.lower()
        prediction = prediction.lower()
        #print target and predictions to a new log file
        logfile.write(target+'\n')
        logfile.write(prediction+'\n')
        logfile.write('-'*50+'\n')
        
        regex_begin_tag = re.escape(args.begin_tag)
        regex_end_tag = re.escape(args.end_tag)
        target_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', target)
        prediction_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', prediction)
        
        tp_sum += len(set(target_mentions).intersection(set(prediction_mentions)))
        relevant_sum += len(target_mentions)
        retrieved_sum += len(prediction_mentions)

    print("precision: ", tp_sum/retrieved_sum if retrieved_sum > 0 else 0)
    print("recall: ", tp_sum/relevant_sum if relevant_sum > 0 else 0)
    print("f1: ", 2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0)
    print("=====================================")

    logfile.write("precision: "+str(tp_sum/retrieved_sum if retrieved_sum > 0 else 0)+'\n')
    logfile.write("recall: "+str(tp_sum/relevant_sum if relevant_sum > 0 else 0)+'\n')
    logfile.write("f1: "+str(2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0)+'\n')
    logfile.write("="*50+'\n')
    logfile.close()
