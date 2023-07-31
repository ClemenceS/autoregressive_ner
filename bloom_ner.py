import datetime
import hashlib
import itertools
import json
import os
import re
import datasets
import numpy as np
from tqdm import tqdm
import argparse
import logging
from prompt_maker import make_prompts, example2string
from bloom_predict import bloom_predict

seed=42
np.random.seed(seed)

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
args.add_argument('-t', '--test_on_test_set', action="store_true")
args.add_argument('-g', '--greedy', action="store_true")
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom_ner")


prompt_keywords = {
    'en' : {
        'first_sentence' : "I am an excellent {}. The task is to label mentions of {}. Below some examples :\n",
        #'last_sentence' : "Imitate me. Identify the mentions of {} in the following sentence, by putting \"{}\" in front and a \"{}\" behind the mention in the following sentence.\n",
        'last_sentence' : "",
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
    dataset_name = 'meczifho/WikiNER'
    dataset = datasets.load_dataset(dataset_name, args.language)
    tag_to_id = {"O":0,"LOC":1,"PER":2,"FAC":3,"ORG":4}
    ner_tag = args.ner_tag if args.ner_tag else 'PER'
else :
    dataset_name = 'meczifho/QuaeroFrenchMed'
    dataset = datasets.load_dataset(dataset_name,'MEDLINE')
    tag_to_id = {"O":0,"ANAT":1,"LIVB":2,"DISO":3,"PROC":4,"CHEM":5,"GEOG":6,"PHYS":7,"PHEN":8,"OBJC":9,"DEVI":10}    
    ner_tag = args.ner_tag if args.ner_tag else 'DISO'

if not args.training_size:
    raise ValueError("Please specify training size")
test_dataset = [example for example in dataset['test'] if len(example['words']) < 40]
traindev_dataset = [example for example in dataset['train'] if len(example['words']) < 40]
def select_dev_dataset(criterion="longest", size=50):
    if criterion == "longest":
        dev_indices = sorted(range(len(traindev_dataset)), key=lambda i: len(traindev_dataset[i]['words']), reverse=True)[:size]
    elif criterion == "random":
        dev_indices = np.random.choice(len(traindev_dataset), size=size, replace=False)
    elif criterion == "most_entities":
        dev_indices = sorted(range(len(traindev_dataset)), key=lambda i: traindev_dataset[i]['ner_tags'].count(tag_to_id[ner_tag]), reverse=True)[:size]
    else:
        raise ValueError("Criterion not recognized")
    return dev_indices

def select_dev_dataset_multiple_criteria(criteria_list, size=50):
    dev_indices = []
    for criterion in criteria_list:
        dev_indices += list(set(select_dev_dataset(criterion, size=size//len(criteria_list))))
    dev_indices = list(set(dev_indices))
    if len(dev_indices) < size:
        dev_indices += list(set(select_dev_dataset("random", size=size-len(dev_indices))))
    return dev_indices

dev_indices = select_dev_dataset_multiple_criteria(["random", "most_entities"], size=20)
dev_dataset = [traindev_dataset[i] for i in dev_indices]
train_dataset = [traindev_dataset[i] for i in range(len(traindev_dataset)) if i not in dev_indices]

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
params = dataset_name+args.language+args.domain+ner_tag+args.begin_tag+args.end_tag+str(args.n_few_shot)+args.criterion+prompt_keywords_string+str(args.training_size)+str(args.test_on_test_set)
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
        test_dataset if args.test_on_test_set else dev_dataset,
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

results = {}

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

    outputs = bloom_predict(
        prompts=prompts,
        args=args,
        logger=logger,
        kwargs={
        "do_sample": not args.greedy,
        "top_p": top_p if not args.greedy else None,
        "top_k": top_k if not args.greedy else None,
        "temperature": temp if not args.greedy else None,
        },
    )

    logger.info("Evaluating...")
    targets = [example2string(example, tag_to_id[ner_tag], args.begin_tag, args.end_tag, tagged=True) for example in (test_dataset if args.test_on_test_set else dev_dataset)]
    for target, o in tqdm(zip(targets, outputs)):
        prediction = ' '.join(o.split('\n')[:1])
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

    tp_sum = float(tp_sum)
    precision = tp_sum/retrieved_sum if retrieved_sum > 0 else 0
    recall = tp_sum/relevant_sum if relevant_sum > 0 else 0
    f1 = 2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0


    print("top_p: ", top_p)
    print("top_k: ", top_k)
    print("temperature: ", temp)
    print("tp: ", tp_sum)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    print("=====================================")

    results[(top_p, top_k, temp)] = [precision, recall, f1]

    logfile.write("precision: "+str(precision)+'\n')
    logfile.write("recall: "+str(recall)+'\n')
    logfile.write("f1: "+str(f1)+'\n')
    logfile.write("="*50+'\n')
    logfile.close()

#sort results by f1 score
results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1][2], reverse=True)}

#print them in a nice table
print("top_p\ttop_k\ttemperature\tprecision\trecall\tf1")
for (top_p, top_k, temp), (precision, recall, f1) in results.items():
    print(str(top_p)+'\t'+str(top_k)+'\t'+str(temp)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(f1))
