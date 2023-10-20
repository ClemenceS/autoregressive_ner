import datetime
import os
import numpy as np
import argparse
import logging
import random

from bloom_predict import bloom_predict
from nlstruct import BRATDataset, HuggingfaceNERDataset
from nlstruct.metrics import MetricsCollection
from nlstruct.registry import get_instance

args = argparse.ArgumentParser()
args.add_argument("--domain", type=str, default="general", help="domain of the dataset")
args.add_argument("--dataset_name", type=str, help="dataset name")
args.add_argument('-d', "--load_dataset_from_disk", action="store_true")
args.add_argument("--begin_tag", type=str, default="@@")
args.add_argument("--end_tag", type=str, default="##")
args.add_argument("--n_few_shot", type=int, default=5)
args.add_argument("--model_name", type=str, default="bigscience/bloom")
args.add_argument("--criterion", type=str, default="most_occurences")
args.add_argument("--prompt_dict", type=str, default="en")
args.add_argument('--top_p', type=float, default=1.0)
args.add_argument('--top_k', type=int, default=50)
args.add_argument('--temperature', type=float, default=1.0)
args.add_argument('--num_beams', type=int, default=1)
args.add_argument('--api_inference', action="store_true")
args.add_argument('--partition_seed', type=int, default=1)
args.add_argument('--random_seed', type=int, default=42)
args.add_argument('-n', '--n_gpus', type=int, default=1)
args.add_argument('-s', '--training_size', type=int, default=100)
args.add_argument('-t', '--test_on_test_set', action="store_true")
args.add_argument('--do_sample', action="store_true")
args.add_argument('--no_control', dest='control', action='store_false')
args.add_argument('--no_self_verification', dest='self_verification', action='store_false')
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom_ner")

#random deals with choosing the few-shot examples, so we want that fixed
random.seed(args.random_seed)

#assert dataset_name is not None
assert args.dataset_name is not None

prompt_keywords = {
    'en' : {
        'first_sentence' : "{}The task is to label all mentions of {} in a sentence. {} I can also put them in a specific format. Here are some examples of sentences I can handle:\n",
        'last_sentence' : "Imitate me. Identify all the mentions of {} in the following sentence, by putting \"{}\" in front and a \"{}\" behind each of them.\n",
        'domains_jobs' : {
            # 'clinical' : "I am an excellent clinician. ",
            'clinical' : "",
            #'general' : "I am an excellent linguist. "
            'general' : "",
            },
        'ner_tags_plural' : {
            'PER' : "person names",
            'DISO' : "disorders",
            'LOC' : "places",
            'ORG' : "organizations",
            'ANAT' : "parts of the body",
            "LIVB" : "living beings",
            "PROC" : "procedures",
            "FAC" : "facilities",
            "CHEM" : "chemicals",
            "DEVI" : "medical devices",
            "GEOG" : "geographical zones",
            "OBJC" : "non-medical objects",
            "PHEN" : "physiolocal phenomema",
            "PHYS" : "human physiology",

            },
        'ner_tags' : {
            'PER' : "a person's name",
            'DISO' : "an alteration of the functions of the body",
            'LOC' : "a place",
            'ORG' : "an organization",
            'ANAT' : "a part of the body",
            "LIVB" : "a living being",
            "PROC" : "a procedure",
            "FAC" : "a facility",
            "CHEM" : "a chemical",
            "DEVI" : "a medical device",
            "GEOG" : "a geographical zone",
            "OBJC" : "a non-medical object",
            "PHEN" : "a physiolocal phenomemon",
            "PHYS" : "a human physiology",
            },
        'ner_tags_description' : {
            'PER' : "These are words that refer to the name of a real or fictional person.",
            'DISO' : "These are words that refer to an alteration or abnormality of the functions or health of the body.",
            'LOC' : "These are words that refer to the name of a place.",
            'ORG' : "These are words that refer to the name of an organization.",
            'ANAT' : "These are words that refer to a part of the human body.",
            "LIVB" : "These are words that refer to a living being.",
            "PROC" : "These are words that refer to a medical procedure.",
            "FAC" : "These are words that refer to a facility made by humans.",
            "CHEM" : "These are words that refer to a drug or a chemical substance.",
            "DEVI" : "These are words that refer to a medical device or a medical instrument.",
            "GEOG" : "These are words that refer to a geographical zone.",
            "OBJC" : "These are words that refer to an object that is not necessarily medical.",
            "PHEN" : "These are words that refer to a physiolocal phenomemon or a physiolocal function.",
            "PHYS" : "These are words that refer to a human physiology.",
            },
        'input_intro' : "Input: ",
        'output_intro' : "Output: ",
        'first_sentence_self_verif' : "{}The task is to verify whether a given word is a mention of a {}. Below some examples :\n",
        "self_verif_template": "In the sentence \"{{sentence}}\", is \"{{word}}\" {ner_tag}?\n",
        "yes": "Yes",
        "no": "No",
        }
    ,
    # 'vicuna_assistant' : {
    #     'first_sentence' : "A chat between a curious {} and an artificial intelligence assistant. The assistant can label all mentions of {} in a sentence. {} It can also put them in a specific format. Here are some examples of sentences it can handle:\n",
    #     'last_sentence' : "",
    #     'domains_jobs' : {
    #         'clinical' : "clinician",
    #         'general' : "linguist"
    #         },
    #     'ner_tags_plural' : {
    #         'PER' : "person names",
    #         'DISO' : "disorders",
    #         'LOC' : "places",
    #         'ORG' : "organizations",
    #         'ANAT' : "parts of the body",
    #         'LIVB' : "living beings",
    #         'PROC' : "procedures",
    #         },
    #     'ner_tags' : {
    #         'PER' : "a person's name",
    #         'DISO' : "an alteration of the functions of the body",
    #         'LOC' : "a place",
    #         'ORG' : "an organization",
    #         'ANAT' : "a part of the body",
    #         'LIVB' : "a living being",
    #         'PROC' : "a procedure",
    #         },
    #     'ner_tags_description' : {
    #         'PER' : "These are words that refer to the name of a real or fictional person.",
    #         'DISO' : "These are words that refer to an alteration or abnormality of the functions or health of the body.",
    #         'LOC' : "These are words that refer to the name of a place.",
    #         'ORG' : "These are words that refer to the name of an organization.",
    #         'ANAT' : "These are words that refer to a part of the human body.",
    #         'LIVB' : "These are words that refer to a living being.",
    #         'PROC' : "These are words that refer to a medical procedure.",
    #         },
    #         'input_intro' : "USER : ",
    #         'output_intro' : "ASSISTANT : ",
    #         'first_sentence_self_verif' : "A chat between a curious {} and an artificial intelligence assistant. The assistant can verify whether a given word is a mention of a {}. Below some examples :\n",
    #         "self_verif_template": "USER : In the sentence \"{sentence}\", is \"{{word}}\" {ner_tag}?\n",
    #         "yes": "ASSISTANT : Yes",
    #         "no": "ASSISTANT : No",
    # },
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
            'LIVB' : "êtres vivants",
            'PROC' : "procédures médicales",
            "FAC" : "installations",
            "CHEM" : "substances chimiques",
            "DEVI" : "appareils médicaux",
            "GEOG" : "zones géographiques",
            "OBJC" : "objets non médicaux",
            "PHEN" : "phénomènes physiologiques",
            "PHYS" : "physiologie humaine",
        },
        'ner_tags' : {
            'PER' : "un nom de personne",
            'DISO' : "une altération des fonctions du corps",
            'LOC' : "lieu",
            'ORG' : "une organisation",
            'ANAT' : "une partie du corps",
            'LIVB' : "un être vivant",
            'PROC' : "une procédure médicale",
            "FAC" : "une installation",
            "CHEM" : "une substance chimique",
            "DEVI" : "un appareil médical",
            "GEOG" : "une zone géographique",
            "OBJC" : "un objet non médical",
            "PHEN" : "un phénomène physiologique",
            "PHYS" : "une physiologie humaine",
        },
        'ner_tags_description' : {
            'PER' : "Il s'agit des mots faisant mention du nom d'un personne qu'elle soit réelle ou fictive.",
            'DISO' : "Il s'agit des mots faisant mention d'une altération ou une anormalité des fonctions ou de la santé du corps.",
            'LOC' : "Il s'agit des mots faisant mention du nom d'un lieu.",
            'ORG' : "Il s'agit des mots faisant mention du nom d'une organisation.",
            'ANAT' : "Il s'agit des mots faisant mention d'une partie du corps humain.",
            'LIVB' : "Il s'agit des mots faisant mention d'un être vivant.",
            'PROC' : "Il s'agit des mots faisant mention d'une procédure médicale.",
            "FAC" : "Il s'agit des mots faisant mention d'une installation faite/construite par les humains.",
            "CHEM" : "Il s'agit des mots faisant mention d'un médicament ou d'une substance chimique.",
            "DEVI" : "Il s'agit des mots faisant mention d'un appareil médical ou d'un instrument médical.",
            "GEOG" : "Il s'agit des mots faisant mention d'une zone géographique.",
            "OBJC" : "Il s'agit des mots faisant mention d'un objet qui n'est pas nécessairement médical.",
            "PHEN" : "Il s'agit des mots faisant mention d'un phénomène physiologique ou d'une fonction physiologique.",
            "PHYS" : "Il s'agit des mots faisant mention d'une physiologie humaine.",
        },
        'input_intro' : "Entrée : ",
        'output_intro' : "Sortie : ",
        'first_sentence_self_verif' : "Je suis un {} expert, je sais identifier si un mot est une mention des {} dans une phrase. Voici quelques exemples de phrases que je peux traiter :\n",
        "self_verif_template": "Dans la phrase \"{{sentence}}\", le mot \"{{word}}\" désigne-t-il {ner_tag} ?\n",
        "yes": "Oui",
        "no": "Non",
    }
}

ner_tags_by_dataset = {
    "WikiNER/en" : ["PER", "LOC", "ORG"],
    "WikiNER/fr" : ["PER", "LOC", "ORG"],
    "conll2003" : ["PER", "LOC", "ORG"],
    "quaero" : ["ANAT", "CHEM", "DEVI", "DISO", "GEOG", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
}
colnames_by_hf_dataset = {
    "WikiNER" : ("id", "words", "ner_tags"),
    "conll2003" : ("id", "tokens", "ner_tags"),
}
tag_map_by_hf_dataset = {
    "WikiNER" : {
        0: "O",
        1: "LOC",
        2: "PER",
        3: "FAC",
        4: "ORG",
    },
    "conll2003" : {
        0: "O",
        1: "PER",
        2: "PER",
        3: "ORG",
        4: "ORG",
        5: "LOC",
        6: "LOC",
        7: "O",
        8: "O",
    },
}

def get_if_key_in_x(dict, x):
    return next((dict[key] for key in dict if key in x), None)

try:
    doc_id_colname, words_colname, ner_tags_colname = get_if_key_in_x(colnames_by_hf_dataset, args.dataset_name)
    dataset = HuggingfaceNERDataset(
        dataset_name=args.dataset_name,
        tag_map=get_if_key_in_x(tag_map_by_hf_dataset, args.dataset_name),
        doc_id_colname=doc_id_colname,
        words_colname=words_colname,
        ner_tags_colname=ner_tags_colname,
        load_from_disk=args.load_dataset_from_disk,
    )
except:
    dataset = BRATDataset(
        train= f"{args.dataset_name}/train",
        val= 0, 
        test= f"{args.dataset_name}/test",
    )
ner_tags = get_if_key_in_x(ner_tags_by_dataset, args.dataset_name)


traindev_dataset = [e for e in dataset.train_data if len(e['text']) < 512]
test_dataset = [e for e in dataset.test_data if len(e['text']) < 512]

folder_name = 'results'
os.makedirs(folder_name, exist_ok=True)

#np random deals with choosing the traindev dataset
np.random.seed(args.partition_seed)
traindev_dataset_this_seed = [traindev_dataset[i] for i in np.random.choice(len(traindev_dataset), size=args.training_size, replace=False)]

time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
last_two_dirs = '-'.join(args.dataset_name.split('/')[-2:])
logfile = open(folder_name+f'/log_{args.domain}_{last_two_dirs}_{args.random_seed}_{time_str}.txt', 'w')
logfile.write('dataset_name: '+last_two_dirs+'\n')
logfile.write('domain: '+args.domain+'\n')
logfile.write('begin_tag: '+args.begin_tag+'\n')
logfile.write('end_tag: '+args.end_tag+'\n')
logfile.write('n_few_shot: '+str(args.n_few_shot)+'\n')
logfile.write('model_name: '+args.model_name+'\n')
logfile.write('criterion: '+args.criterion+'\n')
logfile.write('prompt_dict: '+args.prompt_dict+'\n')
logfile.write('training_size: '+str(args.training_size)+'\n')
logfile.write('partition_seed: '+str(args.partition_seed)+'\n')
logfile.write('random_seed: '+str(args.random_seed)+'\n')
logfile.write('control: '+str(args.control)+'\n')
logfile.write('num_beams: '+str(args.num_beams)+'\n')
logfile.write('self verification: '+str(args.self_verification)+'\n')
# logfile.write('example prompt: \n'+prompts[0]+'\n')
# logfile.write('self_verif_template: \n'+self_verif_template+'\n')
if args.do_sample:
    logfile.write('top_p: '+str(args.top_p)+'\n')
    logfile.write('top_k: '+str(args.top_k)+'\n')
    logfile.write('temperature: '+str(args.temperature)+'\n')
else:
    logfile.write('greedy'+'\n')
logfile.write('='*50+'\n')

model_kwargs = {
    "num_beams": args.num_beams,
}
if args.do_sample:
    model_kwargs.update({
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
    })
else:
    model_kwargs.update({
        "do_sample": False,
    })
textual_outputs, predicted_dataset = bloom_predict(
    training_data=traindev_dataset_this_seed,
    testing_data=test_dataset if args.test_on_test_set else None,
    ner_tags=ner_tags,
    model_name=args.model_name,
    logger=logger,
    begin_tag=args.begin_tag,
    end_tag=args.end_tag,
    self_verification=args.self_verification,
    control=args.control,
    n_few_shot=args.n_few_shot,
    criterion=args.criterion,
    keywords=prompt_keywords[args.prompt_dict],
    domain=args.domain,
    model_kwargs=model_kwargs,
    n_gpus=args.n_gpus,
)

logger.info("Evaluating...")
metric_names = {
        "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., add_label_specific_metrics=ner_tags, filter_entities=ner_tags),
        "partial": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., add_label_specific_metrics=ner_tags, filter_entities=ner_tags),
}
metrics = MetricsCollection({k: get_instance(m) for k, m in metric_names.items()})

s_metrics = ""
for metric_name, metric in metrics.items():
    metric(predicted_dataset, test_dataset if args.test_on_test_set else traindev_dataset_this_seed)
    metric_dict = metric.compute()
    s_metrics+="="*20+metric_name+"="*20+'\n'
    s_metrics+='ALL    tp: '+str(int(metric_dict['tp'].item()))+'    precision: '+str(round(metric_dict['precision'].item(), 3))+'    recall: '+str(round(metric_dict['recall'].item(), 3))+'    f1: '+str(round(metric_dict['f1'].item(), 3))+'\n'
    for tag in ner_tags:
        s_metrics+=tag+'    tp: '+str(int(metric_dict[tag+'_tp'].item()))+'    precision: '+str(round(metric_dict[tag+'_precision'].item(), 3))+'    recall: '+str(round(metric_dict[tag+'_recall'].item(), 3))+'    f1: '+str(round(metric_dict[tag+'_f1'].item(), 3))+'\n'
logfile.write(s_metrics)
print(s_metrics)

for i, (o, pred, gold) in enumerate(zip(textual_outputs, predicted_dataset, test_dataset if args.test_on_test_set else traindev_dataset_this_seed)):
        logfile.write('='*50+'\n')
        logfile.write('input: '+pred['text']+'\n')
        for j,tag in enumerate(ner_tags):
            logfile.write('-'*50+'\n')
            logfile.write(tag+' output: '+textual_outputs[j*len(predicted_dataset)+i]+'\n')
            logfile.write('final: '+str([p['text'] for p in pred['entities'] if p['label']==tag])+'\n')
            logfile.write('gold: '+str([g['text'] for g in gold['entities'] if g['label']==tag])+'\n')

logfile.close()