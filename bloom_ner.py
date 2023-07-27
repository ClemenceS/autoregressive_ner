import datetime
import hashlib
import itertools
import json
import os
import re
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import logging
import torch
import requests
from prompt_maker import make_prompts, example2string


args = argparse.ArgumentParser()
args.add_argument("--language", type=str, default="fr", help="language of the dataset")
args.add_argument("--domain", type=str, default="clinical", help="domain of the dataset")
args.add_argument("--ner_tag", type=str, help="ner tag to evaluate")
args.add_argument("--begin_tag", type=str, default="@@")
args.add_argument("--end_tag", type=str, default="##")
args.add_argument("--n_few_shot", type=int, default=5)
args.add_argument("--model_name", type=str, default="bigscience/bloom")
args.add_argument("--batch_size", type=int, default=2)
args.add_argument("--criterion", type=str, default="closest_tf_idf")
args.add_argument('--top_p', type=float, nargs='+', default=[0.9])
args.add_argument('--top_k', type=int, nargs='+', default=[10])
args.add_argument('--temperature', type=float, nargs='+', default=[0.7])
args.add_argument('--api_inference', action="store_true")
args.add_argument('-o', "--overwrite_prompt_cache", action="store_true")
args.add_argument('-d', '--debug', action="store_true")
args.add_argument('-s', '--training_size', type=int)
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom_ner")


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
        'last_sentence' : "Imite-moi. Identifie les mentions de {} dans la phrase suivante, en mettant \"{}\" devant et un \"{}\" derrière la mention dans la phrase suivante.\n",
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


if args.domain == 'general':
    dataset_name = 'Jean-Baptiste/wikiner_fr'
    dataset = datasets.load_dataset(dataset_name)
    test_dataset = [example for example in dataset['test'] if len(example['tokens']) < 40]
    traindev_dataset = [example for example in dataset['train'] if len(example['tokens']) < 40]
    #get only first 100 traindev examples
    tag_to_id = {"O":0,"LOC":1,"PER":2,"FAC":3,"ORG":4}
    ner_tag = args.ner_tag if args.ner_tag else 'PER'
else :
    dataset_name = 'meczifho/QuaeroFrenchMed'
    dataset = datasets.load_dataset(dataset_name,'MEDLINE')
    traindev_dataset = [example for example in dataset['train'] if len(example['words']) < 40]
    test_dataset = [example for example in dataset['test'] if len(example['words']) < 40]
    #get only first 100 traindev examples
    tag_to_id = {"O":0,"ANAT":1,"LIVB":2,"DISO":3,"PROC":4,"CHEM":5,"GEOG":6,"PHYS":7,"PHEN":8,"OBJC":9,"DEVI":10}    
    ner_tag = args.ner_tag if args.ner_tag else 'DISO'
if not args.training_size:
    raise ValueError("Please specify training size")
dev_size = 20
#the first 20 examples of traindev_dataset are used for dev, the next args.training_size are used for training
dev_dataset = [example for i,example in enumerate(traindev_dataset) if i < dev_size]
train_dataset = [example for i,example in enumerate(traindev_dataset) if i >= dev_size and i < dev_size+args.training_size]
print(len(train_dataset), "examples in train set")
print(len(dev_dataset), "examples in dev set")
print(len(test_dataset), "examples in test set")
if args.debug:
    train_dataset = [t for i,t in enumerate(train_dataset) if i < 10]
    dev_dataset = [t for i,t in enumerate(dev_dataset) if i < 10]
    test_dataset = [t for i,t in enumerate(test_dataset) if i < 10]



time_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = 'hyp_search_'+time_date
os.mkdir(folder_name)


#convert prompt_keywords to string
prompt_keywords_string = json.dumps(prompt_keywords[args.language], ensure_ascii=False)
params = dataset_name+args.language+args.domain+ner_tag+args.begin_tag+args.end_tag+str(args.n_few_shot)+args.criterion+prompt_keywords_string
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
        train_dataset,
        dev_dataset,
        ner_tag, 
        tag_to_id[ner_tag], 
        args.language, 
        args.domain, 
        args.begin_tag, 
        args.end_tag, 
        args.n_few_shot,
        args.criterion,
        prompt_keywords=prompt_keywords,
    )
    
    #cache prompts
    with open('prompts_'+hash_object.hexdigest()+'.txt', 'w') as f:
        for prompt in prompts:
            f.write(prompt+'='*50)


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
        for i in tqdm(range(len(prompts))):
            output = query({"inputs":prompts[i],"parameters":{"top_p":top_p,"top_k":top_k,"temperature":temp, "return_full_text":False}})
            nb_retries = 0
            while 'error' in output and nb_retries < 10:
                output = query({"inputs":prompts[i],"parameters":{"top_p":top_p,"top_k":top_k,"temperature":temp, "return_full_text":False}})
                nb_retries += 1
            if 'error' in output:
                outputs.append('')
            else:
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
    targets = [example2string(example, tag_to_id[ner_tag], args.begin_tag, args.end_tag, tagged=True) for example in dev_dataset]
    for target, o in tqdm(zip(targets, outputs)):
        prediction = o.split('\n')[0]
        target = target.lower()
        prediction = prediction.lower()
        #print target and predictions to a new log file
        logfile.write(target+'\n')
        logfile.write(prediction+'\n')
        logfile.write('-'*50+'\n')
        
        regex_begin_tag = re.escape(args.begin_tag.lower())
        regex_end_tag = re.escape(args.end_tag.lower())
        target_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', target)
        prediction_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', prediction)
        
        tp_sum += len(set(target_mentions).intersection(set(prediction_mentions)))
        relevant_sum += len(target_mentions)
        retrieved_sum += len(prediction_mentions)

    print("top_p: ", top_p)
    print("top_k: ", top_k)
    print("temperature: ", temp)
    print("precision: ", tp_sum/retrieved_sum if retrieved_sum > 0 else 0)
    print("recall: ", tp_sum/relevant_sum if relevant_sum > 0 else 0)
    print("f1: ", 2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0)
    print("=====================================")

    logfile.write("precision: "+str(tp_sum/retrieved_sum if retrieved_sum > 0 else 0)+'\n')
    logfile.write("recall: "+str(tp_sum/relevant_sum if relevant_sum > 0 else 0)+'\n')
    logfile.write("f1: "+str(2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0)+'\n')
    logfile.write("="*50+'\n')
    logfile.close()