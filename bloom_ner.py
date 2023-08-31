import datetime
import hashlib
import itertools
import json
import os
import re
import datasets
import numpy as np
from sklearn.model_selection import KFold
import argparse
import logging
import random
from prompt_maker import make_prompts
from bloom_predict import bloom_predict
from fastchat.model import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer


args = argparse.ArgumentParser()
args.add_argument("--language", type=str, default="fr", help="language of the dataset")
args.add_argument("--domain", type=str, default="general", help="domain of the dataset")
args.add_argument("--ner_tag", type=str, help="ner tag to evaluate")
args.add_argument("--begin_tag", type=str, default="@@")
args.add_argument("--end_tag", type=str, default="##")
args.add_argument("--n_few_shot", type=int, nargs='+', default=[5])
args.add_argument("--model_name", type=str, default="bigscience/bloom")
args.add_argument("--batch_size", type=int, default=2)
args.add_argument("--criterion", type=str, default="most_occurences")
args.add_argument("--prompt_dict", type=str)
args.add_argument('--top_p', type=float, nargs='+', default=[0.5])
args.add_argument('--top_k', type=int, nargs='+', default=[5])
args.add_argument('--temperature', type=float, nargs='+', default=[0.5])
args.add_argument('--api_inference', action="store_true")
args.add_argument('--random_seed', type=int, nargs='+', default=[42])
args.add_argument('-d', '--debug', action="store_true")
args.add_argument('-s', '--training_size', type=int, default=70)
args.add_argument('-t', '--test_on_test_set', action="store_true")
args.add_argument('-g', '--greedy', action="store_true")
args.add_argument('--no_control', dest='control', action='store_false')
args.add_argument('--no_self_verification', dest='self_verification', action='store_false')
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom_ner")

#random deals with choosing the few-shot examples, so we want that fixed
random.seed(42)

prompt_keywords = {
    'en' : {
        'first_sentence' : "I am an excellent {}. The task is to label all mentions of {} in a sentence. {} I can also put them in a specific format. Here are some examples of sentences I can handle:\n",
        'last_sentence' : "Imitate me. Identify all the mentions of {} in the following sentence, by putting \"{}\" in front and a \"{}\" behind each of them.\n",
        'domains_jobs' : {
            'clinical' : "clinician",
            'general' : "linguist"
            },
        'ner_tags_plural' : {
            'PER' : "person names",
            'DISO' : "disorders",
            'LOC' : "places",
            'ORG' : "organizations",
            'ANAT' : "parts of the body",
            },
        'ner_tags' : {
            'PER' : "a person's name",
            'DISO' : "an alteration of the functions of the body",
            'LOC' : "a place",
            'ORG' : "an organization",
            'ANAT' : "a part of the body",
            },
        'ner_tags_description' : {
            'PER' : "These are words that refer to the name of a real or fictional person.",
            'DISO' : "These are words that refer to an alteration or abnormality of the functions or health of the body.",
            'LOC' : "These are words that refer to the name of a place.",
            'ORG' : "These are words that refer to the name of an organization.",
            'ANAT' : "These are words that refer to a part of the human body.",
            },
        'input_intro' : "Input: ",
        'output_intro' : "Output: ",
        'first_sentence_self_verif' : "I am an excellent {}. The task is to verify whether a given word is a mention of a {}. Below some examples :\n",
        "self_verif_template": "In the sentence \"{{sentence}}\", is \"{{word}}\" {ner_tag}?\n",
        "yes": "Yes",
        "no": "No",
        }
    ,
    'vicuna_assistant' : {
        'first_sentence' : "A chat between a curious {} and an artificial intelligence assistant. The assistant can label all mentions of {} in a sentence. {} It can also put them in a specific format. Here are some examples of sentences it can handle:\n",
        'last_sentence' : "",
        'domains_jobs' : {
            'clinical' : "clinician",
            'general' : "linguist"
            },
        'ner_tags_plural' : {
            'PER' : "person names",
            'DISO' : "disorders",
            'LOC' : "places",
            'ORG' : "organizations",
            'ANAT' : "parts of the body",
            },
        'ner_tags' : {
            'PER' : "a person's name",
            'DISO' : "an alteration of the functions of the body",
            'LOC' : "a place",
            'ORG' : "an organization",
            'ANAT' : "a part of the body",
            },
        'ner_tags_description' : {
            'PER' : "These are words that refer to the name of a real or fictional person.",
            'DISO' : "These are words that refer to an alteration or abnormality of the functions or health of the body.",
            'LOC' : "These are words that refer to the name of a place.",
            'ORG' : "These are words that refer to the name of an organization.",
            'ANAT' : "These are words that refer to a part of the human body.",
            },
            'input_intro' : "USER : ",
            'output_intro' : "ASSISTANT : ",
            'first_sentence_self_verif' : "A chat between a curious {} and an artificial intelligence assistant. The assistant can verify whether a given word is a mention of a {}. Below some examples :\n",
            "self_verif_template": "USER : In the sentence \"{{sentence}}\", is \"{{word}}\" {ner_tag}?\n",
            "yes": "ASSISTANT : Yes",
            "no": "ASSISTANT : No",
    },
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
            'LOC' : "lieux",
            'ORG' : "organisations",
            'ANAT' : "parties du corps",
        },
        'ner_tags' : {
            'PER' : "un nom de personne",
            'DISO' : "une altération des fonctions du corps",
            'LOC' : "lieu",
            'ORG' : "une organisation",
            'ANAT' : "une partie du corps",
        },
        'ner_tags_description' : {
            'PER' : "Il s'agit des mots faisant mention du nom d'un personne qu'elle soit réelle ou fictive.",
            'DISO' : "Il s'agit des mots faisant mention d'une altération ou une anormalité des fonctions ou de la santé du corps.",
            'LOC' : "Il s'agit des mots faisant mention du nom d'un lieu.",
            'ORG' : "Il s'agit des mots faisant mention du nom d'une organisation.",
            'ANAT' : "Il s'agit des mots faisant mention d'une partie du corps humain.",
        },
        'input_intro' : "Entrée : ",
        'output_intro' : "Sortie : ",
        'first_sentence_self_verif' : "Je suis un {} expert, je sais identifier si un mot est une mention des {} dans une phrase. Voici quelques exemples de phrases que je peux traiter :\n",
        "self_verif_template": "Dans la phrase \"{{sentence}}\", le mot \"{{word}}\" désigne-t-il {ner_tag} ?\n",
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
if not args.prompt_dict:
    raise ValueError("Please specify prompt dictionary")

test_dataset = [example for example in dataset['test'] if len(example['words']) < 40]
traindev_dataset = [example for example in dataset['train'] if len(example['words']) < 40]


time_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = 'hyp_search_'+time_date
os.mkdir(folder_name)

logger.info("Loading model...")
model, tokenizer = load_model(
        args.model_name,
        device=args.device,
        num_gpus=1,
        load_8bit=True,
        debug=False,
        )

#loop over all combinations of n_few_shot and random_seed
for n_few_shot, random_seed in itertools.product(args.n_few_shot, args.random_seed):
    # #convert prompt_keywords to string
    # prompt_keywords_string = json.dumps(prompt_keywords[args.prompt_dict], ensure_ascii=False)
    # params = dataset_name+args.language+args.domain+ner_tag+args.begin_tag+args.end_tag+str(n_few_shot)+args.criterion+prompt_keywords_string+str(args.training_size)+str(args.test_on_test_set)+str(args.random_seed)
    # hash_object = hashlib.md5(params.encode())
    
    #np random deals with choosing the traindev dataset
    np.random.seed(random_seed)
    
    traindev_dataset_this_seed = [traindev_dataset[i] for i in np.random.choice(len(traindev_dataset), size=args.training_size, replace=False)]
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
        for i, (train_indices, dev_indices) in enumerate(kf.split(traindev_dataset_this_seed)):
            dev_dataset = [traindev_dataset_this_seed[i] for i in dev_indices]
            train_dataset = [traindev_dataset_this_seed[i] for i in train_indices]


            logger.info("{} examples in train set".format(len(train_dataset)))
            logger.info("{} examples in dev set".format(len(dev_dataset)))
            
            logger.info("Making prompts for fold {}".format(i))
            k_prompts, k_targets, k_self_verif_template, k_yes_no = make_prompts(
                train_dataset,
                dev_dataset,
                ner_tag, 
                tag_to_id[ner_tag], 
                args.domain, 
                args.begin_tag, 
                args.end_tag, 
                n_few_shot,
                args.criterion,
                self_verification=args.self_verification,
                sticked=True,
                keywords=prompt_keywords[args.prompt_dict]
            )
            prompts += k_prompts
            targets += k_targets
            self_verif_template = k_self_verif_template
            yes_no = k_yes_no
    else:
        logger.info("{} examples in train set".format(len(traindev_dataset)))
        logger.info("{} examples in test set".format(len(test_dataset)))
        prompts, targets, self_verif_template, yes_no = make_prompts(
            traindev_dataset,
            test_dataset,
            ner_tag,
            tag_to_id[ner_tag],
            args.domain,
            args.begin_tag,
            args.end_tag,
            n_few_shot,
            args.criterion,
            keywords=prompt_keywords[args.prompt_dict]
        )


    logger.info("Number of prompts : {}".format(len(prompts)))
    logger.info("Here is an example prompt :\n{}".format(prompts[0]))
    # logger.info("Saving prompts at {}".format('prompts_'+hash_object.hexdigest()+'.txt'))
    # with open('prompts_'+hash_object.hexdigest()+'.txt', 'w') as f:
    #     for prompt in prompts:
    #         f.write(prompt+'='*50)

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
        logfile.write('n_few_shot: '+str(n_few_shot)+'\n')
        logfile.write('model_name: '+args.model_name+'\n')
        logfile.write('criterion: '+args.criterion+'\n')
        logfile.write('prompt_dict: '+args.prompt_dict+'\n')
        logfile.write('training_size: '+str(args.training_size)+'\n')
        logfile.write('random_seed: '+str(random_seed)+'\n')
        logfile.write('self verification: '+str(args.self_verification)+'\n')
        logfile.write('example prompt: \n'+prompts[0]+'\n')
        logfile.write('self_verif_template: \n'+self_verif_template+'\n')
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
            model=model,
            tokenizer=tokenizer,
            contol=args.control,
            kwargs={
            "do_sample": not args.greedy,
            "top_p": top_p if not args.greedy else None,
            "top_k": top_k if not args.greedy else None,
            "temperature": temp if not args.greedy else None,
            },
        )

        logger.info("Evaluating...")
        for target, prediction, o in zip(targets, predictions, outputs):
            #target = target.lower()
            logfile.write('target: '+target+'\n')
            logfile.write('predictions: '+str(prediction)+'\n')
            logfile.write('-'*50+'\n')
            
            regex_begin_tag = re.escape(args.begin_tag)
            regex_end_tag = re.escape(args.end_tag)
            target_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', target)
            
            tp_sum += len(set(target_mentions).intersection(set(prediction)))
            relevant_sum += len(target_mentions)
            retrieved_sum += len(prediction)

        tp_sum = float(tp_sum)
        precision = tp_sum/retrieved_sum if retrieved_sum > 0 else 0
        recall = tp_sum/relevant_sum if relevant_sum > 0 else 0
        f1 = 2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0


        if args.greedy:
            logger.info("greedy")
        else:
            logger.info("top_p: {}".format(top_p))
            logger.info("top_k: {}".format(top_k))
            logger.info("temperature: {}".format(temp))
        logger.info("tp: {}".format(tp_sum))
        logger.info("precision = {}/{} = {}".format(tp_sum, retrieved_sum, precision))
        logger.info("recall: {}/{} = {}".format(tp_sum, relevant_sum, recall))
        logger.info("f1: {}".format(f1))
        logger.info("=====================================")

        results[(top_p, top_k, temp)] = [precision, recall, f1]

        logfile.write("precision = {}/{} = ".format(tp_sum, retrieved_sum)+str(precision)+'\n')
        logfile.write("recall: {}/{} = ".format(tp_sum, relevant_sum)+str(recall)+'\n')
        logfile.write("f1: "+str(f1)+'\n')
        logfile.write("="*50+'\n')
        logfile.close()

    #sort results by f1 score
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1][2], reverse=True)}

    if not args.test_on_test_set:
        #print them in a nice table
        logger.info("top_p\ttop_k\ttemperature\tprecision\trecall\tf1")
        for (top_p, top_k, temp), (precision, recall, f1) in results.items():
            logger.info("{}\t{}\t{}\t{}\t{}\t{}".format(top_p, top_k, temp, precision, recall, f1))