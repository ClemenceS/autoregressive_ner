import random
import re
import datasets
import requests

def example2string(example, ner_tag_id, begin_tag, end_tag):
    # if ner_tag_id = 3 and 3 stands for LOC, beginning tag = @@ and ending tag = ##
    # and the example is {'id': 0, 'words': ['I', 'love', 'Paris', 'and', 'Berlin'], 'ner_tags': [0, 0, 3, 0, 3]}
    # the returned string will be 'I love @@Paris## and @@Berlin##'
    words = example['words' if 'words' in example else 'tokens']
    ner_tags = example['ner_tags']
    # initialize the string
    string = ''
    for i, (word, ner_tag) in enumerate(zip(words, ner_tags)):
        # if the ner tag is equal to the given ner tag id and the last ner tag was not equal to the given ner tag id
        if ner_tag == ner_tag_id and (ner_tags[i-1] != ner_tag_id if i > 0 else True):
            # add the beginning tag to the string
            string += begin_tag
        # add the word to the string
        string += word
        # if the ner tag is equal to the given ner tag id and the next ner tag is not equal to the given ner tag id
        if ner_tag == ner_tag_id and (ner_tags[i+1] != ner_tag_id if i < len(ner_tags)-1 else True):
            # add the ending tag to the string
            string += end_tag
        # add a space to the string
        string += ' '
    # return the string
    return string.strip()


english_keywords = {
    'first_sentence' : "I am an expert {}, I can identify mentions of {} in a sentence. I can also format them. Here are some examples of sentences I can handle:\n",
    'last_sentence' : "Imitate me. Identify the mentions of {} in the following sentence, by putting \"{}\" in front and a \"{}\" behind the mention in the following sentence.\n",
    'domains_jobs' : {
        'clinical' : "clinician",
        'general' : "linguist"
    },
    'ner_tags' : {
        'PER' : "persons",
        'DISO' : "disorders",
        'LOC' : "places"
    },
    'input_intro' : "Input: ",
    'output_intro' : "Output: ",
}

french_keywords = {
    'first_sentence' : "Je suis un {} expert, je sais identifier les mentions des {} dans une phrase. Je peux aussi les mettre en forme. Voici quelques exemples de phrases que je peux traiter :\n",
    'last_sentence' : "Imite-moi. Identifie les mentions de {} dans la phrase suivante, en mettant \"{}\" devant et un \"{}\" derrière la mention dans la phrase suivante.\n",
    'domains_jobs' : {
        'clinical' : "clinicien",
        'general' : "linguiste"
    },
    'ner_tags' : {
        'PER' : "personnes",
        'DISO' : "maladies et symptômes",
        'LOC' : "lieux"
    },
    'input_intro' : "Entrée : ",
    'output_intro' : "Sortie : ",
}

quaero_tags = {
    "O": 0,
    "ANAT": 1,
    "LIVB": 2,
    "DISO": 3,
    "PROC": 4,
    "CHEM": 5,
    "GEOG": 6,
    "PHYS": 7,
    "PHEN": 8,
    "OBJC": 9,
    "DEVI": 10
}

wikiner_tags = {
    "O": 0,
    "LOC": 1,
    "PER": 2,
    "FAC": 3,
    "ORG": 4,
}

prompt_keywords = {
    'en' : english_keywords,
    'fr' : french_keywords
}

def demonstrate(dataset, example_index, ner_tag_id, language, begin_tag, end_tag, test=False):
    keywords = prompt_keywords[language]
    example = dataset['test' if test else 'train'][example_index]
    output = keywords['input_intro']+' '.join(example['words' if 'words' in example else 'tokens'])+'\n'
    output+= keywords['output_intro']
    if not test:
        output+=example2string(example, ner_tag_id, begin_tag, end_tag)+'\n'
    return output


def make_prompt(dataset, example_index, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag):
    #this function takes an example and a ner tag and returns a prompt in english
    keywords = prompt_keywords[language]
    prompt = keywords['first_sentence'].format(keywords['domains_jobs'][domain], keywords['ner_tags'][ner_tag])
    #get the first example
    prompt+= demonstrate(dataset, 0, ner_tag_id, language, begin_tag, end_tag)
    prompt+= demonstrate(dataset, 1, ner_tag_id, language, begin_tag, end_tag)
    prompt+= demonstrate(dataset, 2, ner_tag_id, language, begin_tag, end_tag)
    prompt+= keywords['last_sentence'].format(keywords['ner_tags'][ner_tag], begin_tag, end_tag)
    prompt+= demonstrate(dataset, example_index, ner_tag_id, language, begin_tag, end_tag, test=True)
    return prompt


def query(payload):
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
    
def get_model_predictions(dataset, example_index, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag):
    prompt = make_prompt(dataset, example_index, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag)
    print(prompt)
    print("-"*50)
    
    output = query({
        "inputs": prompt, "options": {"use_cache": True},
        "parameters": {"max_new_tokens": 100,"return_full_text": False,"top_p": 0.9,"top_k": 3,"temperature": 0.7,},
    })
    if "error" in output:
        raise Exception(output['error'])
    return output[0]['generated_text'].split('\n')[0]
     
def evaluate_model_prediction(dataset, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag):
    tp_sum = 0
    relevant_sum = 0
    retrieved_sum = 0

    for _ in range(10):
        example_index = random.randint(0, len(dataset['test'])-1)
        target = example2string(dataset['test'][example_index], ner_tag_id, begin_tag, end_tag)
        prediction = get_model_predictions(dataset, example_index, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag)
        print(target)
        print(prediction)
        print("-"*50)
    
        regex_begin_tag = re.escape(begin_tag)
        regex_end_tag = re.escape(end_tag)
        target_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', target)
        prediction_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', prediction)
        print(target_mentions)
        print(prediction_mentions)
        print("-"*50)
        
        tp = len(set(target_mentions).intersection(set(prediction_mentions)))
        relevant = len(target_mentions)
        retrieved = len(prediction_mentions)
        print("tp: ", tp)
        print("relevant: ", relevant)
        print("retrieved: ", retrieved)
        print("-"*50)
        
        tp_sum += len(set(target_mentions).intersection(set(prediction_mentions)))
        relevant_sum += len(target_mentions)
        retrieved_sum += len(prediction_mentions)
        print("tp_sum: ", tp_sum)
        print("relevant_sum: ", relevant_sum)
        print("retrieved_sum: ", retrieved_sum)
        print("-"*50)
        
        print("precision: ", tp_sum/retrieved_sum if retrieved_sum > 0 else 0)
        print("recall: ", tp_sum/relevant_sum if relevant_sum > 0 else 0)
        print("f1: ", 2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0)
        print("=====================================")
    
dataset = datasets.load_dataset('meczifho/QuaeroFrenchMed','MEDLINE')
# dataset = datasets.load_dataset('Jean-Baptiste/wikiner_fr')

evaluate_model_prediction(dataset, 'DISO', quaero_tags["DISO"], 'en', 'clinical', '@@', '##')