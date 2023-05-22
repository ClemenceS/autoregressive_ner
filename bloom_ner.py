import re
import datasets
import requests
# from huggingface_hub import HfApi
from sklearn.metrics import f1_score, precision_score, recall_score

def datasets2bloom_readable_format(example, ner_tag_id, begin_tag='@@', end_tag='##'):
    # this function takes a dataset example and a ner_tag_id and returns a string
    # the example is a dictionary with the following keys: 'id', 'words', 'ner_tags'
    # the returned string is a sentence with the words of the given entity type surrounded by ## and @@
    # for example, if ner_tag_id = 3 and 3 stands for LOC, beginning tag = ## and ending tag = @@
    # and the example is {'id': 0, 'words': ['I', 'love', 'Paris', 'and', 'Berlin'], 'ner_tags': [0, 0, 3, 0, 3]}
    # the returned string will be 'I love ##Paris@@ and ##Berlin@@'
    # if the ner_tag_id = 2 and 2 stands for PER, beginning tag = ## and ending tag = @@
    # and the example is {'id': 0, 'words': ['Barack', 'Obama', 'is', 'the', 'president', 'of', 'the', 'United', 'States', 'of', 'America'], 'ner_tags': [2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3]}
    # the returned string will be '##Barack Obama@@ is the president of the United States of America'
    
    # get the words of the example
    words = example['words']
    # get the ner tags of the example
    ner_tags = example['ner_tags']
    # initialize the string
    string = ''
    # iterate over the words and the ner tags
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

def bloom_readable_format2datasets(string, ner_tag_id, begin_tag='@@', end_tag='##'):
    # this function takes a string and returns a list of words and a list of ner tags
    # the string is a sentence with the words of the given entity type surrounded by ## and @@
    # for example, if ner_tag_id = 3 and 3 stands for LOC, beginning tag = ## and ending tag = @@
    # and the string is 'I love ##Paris@@ and ##Berlin@@'
    # the returned list of words will be ['I', 'love', 'Paris', 'and', 'Berlin']
    # and the returned list of ner tags will be [0, 0, 3, 0, 3]
    # if the ner_tag_id = 2 and 2 stands for PER, beginning tag = ## and ending tag = @@
    # and the string is '##Barack Obama@@ is the president of the United States of America'
    # the returned list of words will be ['Barack', 'Obama', 'is', 'the', 'president', 'of', 'the', 'United', 'States', 'of', 'America']
    # and the returned list of ner tags will be [2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3]

    # initialize the list of words
    words = []
    # initialize the list of ner tags
    ner_tags = []
    # split the string into around the beginning and ending tags
    words_split = re.findall(r"[\w']+|@@|##", string, re.UNICODE)
    # loop over the words
    inside_tag = False
    for word in words_split:
        # if the word is equal to the beginning tag
        if word == begin_tag:
            # set inside tag to True
            inside_tag = True
        # if the word is equal to the ending tag
        elif word == end_tag:
            # set inside tag to False
            inside_tag = False
        # if the word is not equal to the beginning tag and not equal to the ending tag
        else:
            # add the word to the list of words
            words.append(word)
            # if inside tag is True
            if inside_tag:
                # add the ner tag id to the list of ner tags
                ner_tags.append(ner_tag_id)
            # if inside tag is False
            else:
                # add 0 to the list of ner tags
                ner_tags.append(0)
    
    # return the list of words and the list of ner tags
    return {'words': words, 'ner_tags': ner_tags}

def DISO_prompt(example):
    #this function takes an example and a ner tag and returns a prompt
    prompt = "Je suis un clinicien expert, je sais identifier les mentions des maladies et des symptômes dans une phrase. Je peux aussi les mettre en forme. Voici quelques exemples de phrases que je peux traiter :\n"
    prompt+= "Entrée : Diagnostic et traitement de l' impuissance . Indications des injections intracaverneuses .\n"
    prompt+= "Sortie : Diagnostic et traitement de l' @@impuissance## . Indications des injections intracaverneuses .\n"
    prompt+= "Entrée : Stratégie chirurgicale de l' adénocarcinome du cardia .\n"
    prompt+= "Sortie : Stratégie chirurgicale de l' @@adénocarcinome du cardia## .\n"
    prompt+= "Entrée : Le paracétamol dans le traitement des douleurs arthrosiques .\n"
    prompt+= "Sortie : Le paracétamol dans le traitement des @@douleurs arthrosiques## .\n"
    prompt+= "Imite-moi. Identifie les mentions de maladies ou de symptômes dans la phrase suivante, en mettant \"@@\" devant et un \"##\" derrière la mention.\n"
    prompt+= "Entrée : "+example+"\n"
    prompt+= "Sortie : "
    return prompt

def PER_prompt(example):
    #this function takes an example and a ner tag and returns a prompt
    prompt = "Je suis un linguiste expert, je sais identifier les mentions des personnes dans une phrase. Je peux aussi les mettre en forme. Voici quelques exemples de phrases que je peux traiter :\n"
    prompt+= "Entrée : Le président de la République française est Emmanuel Macron .\n"
    prompt+= "Sortie : Le président de la République française est @@Emmanuel Macron## .\n"
    prompt+= "Entrée : Barack Obama est le président des États-Unis .\n"
    prompt+= "Sortie : @@Barack Obama## est le président des États-Unis .\n"
    prompt+= "Entrée : Zinedine Zidane explique que le Real Madrid a besoin de Karim Benzema .\n"
    prompt+= "Sortie : Zinedine Zidane explique que le Real Madrid a besoin de @@Karim Benzema## .\n"
    prompt+= "Imite-moi. Identifie les mentions de personnes dans la phrase suivante, en mettant \"@@\" devant et un \"##\" derrière la mention.\n"
    prompt+= "Entrée : "+example+"\n"
    prompt+= "Sortie : "
    return prompt

def get_bloom_predictions(example_string, ner_tag):
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    if ner_tag == 'DISO':
        prompt = DISO_prompt(example_string)
    elif ner_tag == 'PER':
        prompt = PER_prompt(example_string)
    else:
        raise NotImplementedError

    output = query({
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "return_full_text": False,
            "top_p": 0.9,
            "top_k": 3,
            "temperature": 0.7,
        },
        "options": {
            "use_cache": True
        }     
    })
    return output[0]['generated_text']
     
def evaluate_bloom_prediction(example, ner_tag, ner_tag_id):
    #example is a dictionary with the keys 'doc_id', 'words', 'ner_tags'

    words = example['words'] if 'words' in example else example['tokens']
    ner_tags = example['ner_tags']
    print(datasets2bloom_readable_format({'words': words, 'ner_tags': ner_tags}, ner_tag_id))
    
    #get the bloom prediction
    bloom_prediction = get_bloom_predictions(' '.join(words), ner_tag)
    print(bloom_prediction)

    #get the bloom prediction in the format of a list of words and a list of ner tags
    bloom_prediction = bloom_readable_format2datasets(bloom_prediction, ner_tag_id)
    words_bloom = bloom_prediction['words']
    ner_tags_bloom = bloom_prediction['ner_tags']

    if words != words_bloom:
        f1_s = 0
        precision = 0
        recall = 0

    else:
        f1_s = f1_score(ner_tags, ner_tags_bloom, average='micro')
        precision = precision_score(ner_tags, ner_tags_bloom, average='micro')
        recall = recall_score(ner_tags, ner_tags_bloom, average='micro')

    print("f1 score : ", f1_s)
    print("precision : ", precision)
    print("recall : ", recall)
    

# dataset = datasets.load_dataset('meczifho/quaero')
# examples = dataset['train']

# for i in range(10):
#     evaluate_bloom_prediction(examples[i], 'DISO', 3)

dataset = datasets.load_dataset('Jean-Baptiste/wikiner_fr')
examples = dataset['train']

for i in range(10, 20):
    evaluate_bloom_prediction(examples[i], 'PER', 2)