import datetime
import hashlib
import itertools
import json
import os
import re
import datasets
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import argparse
import logging
import random
from prompt_maker import make_prompts, example2string
from bloom_predict import bloom_predict


args = argparse.ArgumentParser()
args.add_argument("--language", type=str, default="fr", help="language of the dataset")
args.add_argument("--domain", type=str, default="clinical", help="domain of the dataset")
args.add_argument("--ner_tag", type=str, help="ner tag to evaluate")
args.add_argument("--begin_tag", type=str, default="@@")
args.add_argument("--end_tag", type=str, default="##")
args.add_argument("--n_few_shot", type=int, default=10)
args.add_argument("--model_name", type=str, default="bigscience/bloom")
args.add_argument("--batch_size", type=int, default=2)
args.add_argument("--criterion", type=str, default="most_occurences")
args.add_argument('--top_p', type=float, nargs='+', default=[0.5])
args.add_argument('--top_k', type=int, nargs='+', default=[5])
args.add_argument('--temperature', type=float, nargs='+', default=[0.5])
args.add_argument('--api_inference', action="store_true")
args.add_argument('--random_seed', type=int, default=42)
args.add_argument('-d', '--debug', action="store_true")
args.add_argument('-s', '--training_size', type=int, default=70)
args.add_argument('-t', '--test_on_test_set', action="store_true")
args.add_argument('-g', '--greedy', action="store_true")
args.add_argument('--no_self_verification', dest='self_verification', action='store_false')
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom_ner")

random.seed(args.random_seed)
np.random.seed(args.random_seed)



prompt_keywords = {
    'en' : {
        'first_sentence' : "I am an excellent {}. The task is to label mentions of {} in a sentence. {} I can also put them in a specific format. Here are some examples of sentences I can handle:\n",
        'last_sentence' : "Imitate me. Identify the mentions of {} in the following sentence, by putting \"{}\" in front and a \"{}\" behind the mention in the following sentence.\n",
        'domains_jobs' : {
            'clinical' : "clinician",
            'general' : "linguist"
            },
        'ner_tags_plural' : {
            'PER' : "person names",
            'DISO' : "disorders",
            'LOC' : "places"
            },
        'ner_tags' : {
            'PER' : "person's name",
            'DISO' : "disorder",
            'LOC' : "place"
            },
        'ner_tags_description' : {
            'PER' : "These are words that refer to the name of a real or fictional person.",
            'DISO' : "These are words that refer to an alteration or abnormality of the functions or health of the body.",
            'LOC' : "These are words that refer to a place."
            },
        'input_intro' : "Input: ",
        'output_intro' : "Output: ",
        'first_sentence_self_verif' : "I am an excellent {}. The task is to verify whether a given word is a mention of a {}. Below some examples :\n",
        "self_verif_template": "In the sentence \"{}\", is \"{}\" a ",
        "yes": "Yes",
        "no": "No",
        }
    ,
    'fr' : {
        'first_sentence' : "Je suis un {} expert, je sais identifier les mentions des {} dans une phrase. {} Je peux aussi les mettre en forme. Voici quelques exemples de phrases que je peux traiter :\n",
        'last_sentence' : "Imite-moi. Identifie les mentions de {} dans la phrase suivante, en mettant \"{}\" devant et un \"{}\" derrière la mention dans la phrase suivante.\n",
        'domains_jobs' : {
            'clinical' : "clinicien",
            'general' : "linguiste"
        },
        'ner_tags_plural' : {
            'PER' : "noms de personnes",
            'DISO' : "maladies et symptômes",
            'LOC' : "lieux"
        },
        'ner_tags' : {
            'PER' : "un nom de personne",
            'DISO' : "une altération des fonctions du corps",
            'LOC' : "lieu"
        },
        'ner_tags_description' : {
            'PER' : "Il s'agit des mots faisant mention du nom d'un personne qu'elle soit réelle ou fictive.",
            'DISO' : "Il s'agit des mots faisant mention d'une altération ou une anormalité des fonctions ou de la santé du corps.",
            'LOC' : "Il s'agit des mots faisant mention d'un lieu."
        },
        'input_intro' : "Entrée : ",
        'output_intro' : "Sortie : ",
        'first_sentence_self_verif' : "Je suis un {} expert, je sais identifier si un mot est une mention des {} dans une phrase. Voici quelques exemples de phrases que je peux traiter :\n",
        "self_verif_template": "Dans la phrase \"{}\", le mot \"{}\" désigne-t-il ",
        "yes": "Oui",
        "no": "Non",
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

time_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = 'hyp_search_'+time_date
os.mkdir(folder_name)

#convert prompt_keywords to string
prompt_keywords_string = json.dumps(prompt_keywords[args.language], ensure_ascii=False)
params = dataset_name+args.language+args.domain+ner_tag+args.begin_tag+args.end_tag+str(args.n_few_shot)+args.criterion+prompt_keywords_string+str(args.training_size)+str(args.test_on_test_set)+str(args.random_seed)
hash_object = hashlib.md5(params.encode())

traindev_dataset = [traindev_dataset[i] for i in np.random.choice(len(traindev_dataset), size=args.training_size, replace=False)]
if args.debug:
    # train_dataset = [t for i,t in enumerate(train_dataset) if i < 10]
    # dev_dataset = [t for i,t in enumerate(dev_dataset) if i < 10]
    test_dataset = [t for i,t in enumerate(test_dataset) if i < 100]
if not args.test_on_test_set:
    prompts = []
    targets = []
    self_verif_template = []
    yes_no = []
    #do k-fold cross validation
    kf = KFold(n_splits=5, shuffle=False)
    for i, (train_indices, dev_indices) in enumerate(kf.split(traindev_dataset)):
        dev_dataset = [traindev_dataset[i] for i in dev_indices]
        train_dataset = [traindev_dataset[i] for i in range(len(traindev_dataset)) if i not in dev_indices]


        print(len(train_dataset), "examples in train set")
        print(len(dev_dataset), "examples in dev set")
        print(len(test_dataset), "examples in test set")

        logger.info("Making prompts for fold {}".format(i))
        k_prompts, k_targets, k_self_verif_template, k_yes_no = make_prompts(
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
        prompts += k_prompts
        targets += k_targets
        self_verif_template = k_self_verif_template
        yes_no = k_yes_no
else:
    prompts, targets, self_verif_template, yes_no = make_prompts(
        traindev_dataset,
        test_dataset,
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



logger.info("Saving prompts at {}".format('prompts_'+hash_object.hexdigest()+'.txt'))
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
    if args.greedy:
        logfile.write('greedy: True\n')
    else:
        logfile.write('top_p: '+str(top_p)+'\n')
        logfile.write('top_k: '+str(top_k)+'\n')
        logfile.write('temperature: '+str(temp)+'\n')
    logfile.write('='*50+'\n')

    tp_sum = 0
    relevant_sum = 0
    retrieved_sum = 0

    predictions,outputs = bloom_predict(
        prompts=prompts,
        api_inference=args.api_inference,
        model_name=args.model_name,
        batch_size=args.batch_size,
        logger=logger,
        begin_tag=args.begin_tag,
        end_tag=args.end_tag,
        self_verif_template=self_verif_template,
        yes_no=yes_no,
        self_verification=args.self_verification,
        kwargs={
        "do_sample": not args.greedy,
        "top_p": top_p if not args.greedy else None,
        "top_k": top_k if not args.greedy else None,
        "temperature": temp if not args.greedy else None,
        },
    )

    logger.info("Evaluating...")
    for target, prediction in tqdm(zip(targets, predictions)):
        #target = target.lower()
        logfile.write(target+'\n')
        #logfile.write(prediction+'\n')
        logfile.write('-'*50+'\n')
        
        regex_begin_tag = re.escape(args.begin_tag.lower())
        regex_end_tag = re.escape(args.end_tag.lower())
        target_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', target)
        
        tp_sum += len(set(target_mentions).intersection(set(prediction)))
        relevant_sum += len(target_mentions)
        retrieved_sum += len(prediction)

    tp_sum = float(tp_sum)
    precision = tp_sum/retrieved_sum if retrieved_sum > 0 else 0
    recall = tp_sum/relevant_sum if relevant_sum > 0 else 0
    f1 = 2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0


    if args.greedy:
        print("greedy")
    else:
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

if not args.test_on_test_set:
    #print them in a nice table
    print("top_p\ttop_k\ttemperature\tprecision\trecall\tf1")
    for (top_p, top_k, temp), (precision, recall, f1) in results.items():
        print(str(top_p)+'\t'+str(top_k)+'\t'+str(temp)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(f1))