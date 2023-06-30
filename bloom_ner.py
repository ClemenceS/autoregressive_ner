import re
import datasets
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

def query(payload):
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(API_URL, headers=headers, json=payload)
    #check if the response is an error
    if response.status_code != 200:
        raise Exception(response.json())
    return response.json()

def sentences_with_most_occurences(dataset, example_index, ner_tag_id, n):
    counts = [e['ner_tags'].count(ner_tag_id) for e in dataset['train']]
    return sorted(range(len(counts)), key=lambda i: counts[i])[-n:]

def sentences_with_most_common_words(dataset, example_index, ner_tag_id, n):
    counts = [len(set(e['words']).intersection(set(dataset['train'][example_index]['words']))) for e in dataset['train']]
    return sorted(range(len(counts)), key=lambda i: counts[i])[-n:]


def make_prompt(dataset, example_index, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag):
    #this function takes an example and a ner tag and returns a prompt in english
    keywords = prompt_keywords[language]
    prompt = keywords['first_sentence'].format(keywords['domains_jobs'][domain], keywords['ner_tags'][ner_tag])
    #get the first example
    for i in sentences_with_most_common_words(dataset, example_index, ner_tag_id, 5):
        prompt+= keywords['input_intro']+example2string(dataset['train'][i], ner_tag_id, begin_tag, end_tag, tagged=False)+'\n'
        prompt+= keywords['output_intro']+example2string(dataset['train'][i], ner_tag_id, begin_tag, end_tag, tagged=True)+'\n'
    prompt+= keywords['last_sentence'].format(keywords['ner_tags'][ner_tag], begin_tag, end_tag)
    prompt+= keywords['input_intro']+example2string(dataset['test'][example_index], ner_tag_id, begin_tag, end_tag, tagged=False)+'\n'
    prompt+= keywords['output_intro']
    return prompt

def evaluate_model_prediction(dataset, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag):
    tp_sum = 0
    relevant_sum = 0
    retrieved_sum = 0

    prompt = make_prompt(dataset, 100, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag)
    print(prompt)
    for i in range(len(dataset['test'])):
        target = example2string(dataset['test'][i], ner_tag_id, begin_tag, end_tag, tagged=True)
        prompt = make_prompt(dataset, i, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag)
        output = query({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 100,"return_full_text": False,"top_p": 0.9,"top_k": 3,"temperature": 0.7,},
        })
        if "error" in output:
            raise Exception(output['error'])
        prediction = output[0]['generated_text'].split('\n')[0]
        #print target and predictions to a new log file
        with open('../autoreg_logs/'+ner_tag+'_'+domain+'_'+language+'_'+begin_tag+'_'+end_tag+'.txt', 'a') as f:
            f.write(target+'\n')
            f.write(prediction+'\n\n')
        
        regex_begin_tag = re.escape(begin_tag)
        regex_end_tag = re.escape(end_tag)
        target_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', target)
        prediction_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', prediction)
        
        tp_sum += len(set(target_mentions).intersection(set(prediction_mentions)))
        relevant_sum += len(target_mentions)
        retrieved_sum += len(prediction_mentions)

        print("precision: ", tp_sum/retrieved_sum if retrieved_sum > 0 else 0)
        print("recall: ", tp_sum/relevant_sum if relevant_sum > 0 else 0)
        print("f1: ", 2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0)
        print("=====================================")

prompt_keywords = {
    'en' : {
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
            'PER' : "personnes",
            'DISO' : "maladies et symptômes",
            'LOC' : "lieux"
        },
        'input_intro' : "Entrée : ",
        'output_intro' : "Sortie : ",
    }
}

dataset = datasets.load_dataset('meczifho/QuaeroFrenchMed','MEDLINE')
# dataset = datasets.load_dataset('Jean-Baptiste/wikiner_fr')

quaero_tags = {"O":0,"ANAT":1,"LIVB":2,"DISO":3,"PROC":4,"CHEM":5,"GEOG":6,"PHYS":7,"PHEN":8,"OBJC":9,"DEVI":10}
wikiner_tags = {"O":0,"LOC":1,"PER":2,"FAC":3,"ORG":4}

evaluate_model_prediction(dataset, 'DISO', quaero_tags["DISO"], 'fr', 'clinical', '@@', '##')