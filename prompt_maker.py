import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
    
def example2string(example, ner_tag, begin_tag, end_tag, sticked, tagged):
    if not tagged:
        return example['text'].rstrip()
    begins = [e['fragments'][0]['begin'] for e in example['entities'] if e['label'] == ner_tag]
    ends = [e['fragments'][0]['end'] for e in example['entities'] if e['label'] == ner_tag]
    res_text = ''
    for i,c in enumerate(example['text']):
        for _ in range(begins.count(i)):
            res_text+=begin_tag + ('' if sticked else ' ')
        for _ in range(ends.count(i)):
            res_text+=('' if sticked else ' ') + end_tag
        res_text+=c
    return res_text.rstrip()

def make_prompts(train_dataset, test_dataset, ner_tag, domain, begin_tag, end_tag, n_few_shot, criterion, keywords, self_verification):
    #this function takes an example and a ner tag and returns a prompt in english
    few_shots_for_all = []
    num_prompts = len(test_dataset)
    def sentences_with_most_occurences(train_dataset, ner_tag, n):
        return sorted(range(len(train_dataset)), key=lambda i: len([ent for ent in train_dataset[i]['entities'] if ent['label'] == ner_tag]), reverse=True)[:n]
    if criterion == 'random':
        for i in range(num_prompts):
            few_shots_for_all.append(random.sample(range(len(train_dataset)), n_few_shot))
    elif criterion == 'most_occurences':
        few_shots_for_all = [sentences_with_most_occurences(train_dataset, ner_tag, n_few_shot)] * num_prompts
    elif criterion == 'closest_tf_idf':
        #get the k nearest sentences in the training set
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        transformed_train = tfidf.fit_transform([e['words' if 'words' in e else 'tokens'] for e in train_dataset])
        transformed_test = tfidf.transform([e['words' if 'words' in e else 'tokens'] for e in test_dataset])
        similarities = cosine_similarity(transformed_test, transformed_train)
        few_shots_for_all = [sorted(range(len(similarities[i])), key=lambda j: similarities[i][j])[-n_few_shot:] for i in range(num_prompts)]
    elif criterion == 'closest_tf_idf_and_most_occurences':
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        transformed_train = tfidf.fit_transform([e['words' if 'words' in e else 'tokens'] for e in train_dataset])
        transformed_test = tfidf.transform([e['words' if 'words' in e else 'tokens'] for e in test_dataset])
        similarities = cosine_similarity(transformed_test, transformed_train)
        few_shots_for_all = [sorted(range(len(similarities[i])), key=lambda j: similarities[i][j])[-n_few_shot//2:]+sentences_with_most_occurences(train_dataset, ner_tag, n_few_shot//2) for i in range(num_prompts)]

    prompts = []
    for p in range(num_prompts):
        example = test_dataset[p]
        prompt = keywords['first_sentence'].format(keywords['domains_jobs'][domain], keywords['ner_tags_plural'][ner_tag], keywords['ner_tags_description'][ner_tag])
        few_shots= few_shots_for_all[p]
        random.shuffle(few_shots)
        for i in few_shots:
            prompt+= keywords['input_intro']+example2string(train_dataset[i], ner_tag, begin_tag, end_tag, sticked=True, tagged=False)+'\n'
            prompt+= keywords['output_intro']+example2string(train_dataset[i], ner_tag, begin_tag, end_tag, sticked=True, tagged=True)+'\n'
        prompt+= keywords['last_sentence'].format(keywords['ner_tags_plural'][ner_tag], begin_tag, end_tag)
        prompt+= keywords['input_intro']+example2string(example, ner_tag, begin_tag, end_tag, sticked=True, tagged=False)+'\n'
        prompt+= keywords['output_intro']
        prompts.append(prompt) 
    
    if not self_verification:
        return prompts, None
    
    self_verification_template = keywords['first_sentence_self_verif'].format(keywords['domains_jobs'][domain], keywords['ner_tags_plural'][ner_tag])
    examples=[]
    #add positive examples
    if n_few_shot > len([e for e in train_dataset if ner_tag in [ent['label'] for ent in e['entities']] ]):
        pos_examples = [e for e in train_dataset if ner_tag in [ent['label'] for ent in e['entities']] ]
    else:
        pos_examples = random.sample([e for e in train_dataset if ner_tag in [ent['label'] for ent in e['entities']] ], n_few_shot)
    for example in pos_examples:
        example_string = example2string(example, ner_tag, begin_tag, end_tag, sticked=True, tagged=False)
        entities = [ent for ent in example['entities'] if ent['label'] == ner_tag]
        entity = random.choice(entities)['text']
        examples.append((example_string, entity, "yes"))
    #add negative examples with another entity
    if n_few_shot > len([e for e in train_dataset if set([ent['label'] for ent in e['entities']])-{'O',ner_tag} ]):
        neg_examples = [e for e in train_dataset if set([ent['label'] for ent in e['entities']])-{'O',ner_tag} ]
    else:
        neg_examples = random.sample([e for e in train_dataset if set([ent['label'] for ent in e['entities']])-{'O',ner_tag} ], n_few_shot)
    for example in neg_examples:
        other_label = list(set([ent['label'] for ent in example['entities']])-{'O',ner_tag})[0]
        example_string = example2string(example, other_label, begin_tag, end_tag, sticked=True, tagged=False)
        other_entities = [ent for ent in example['entities'] if ent['label'] == other_label]
        entity = random.choice(other_entities)['text']
        examples.append((example_string, entity, "no"))

    #shuffle the examples
    random.shuffle(examples)

    for example, pred, label in examples:
        self_verification_template+= keywords['self_verif_template'].format(ner_tag=keywords['ner_tags'][ner_tag]).format(word=pred,sentence=example,)+keywords[label]+"\n"
    self_verification_template+= keywords['self_verif_template'].format(ner_tag=keywords['ner_tags'][ner_tag])

    return prompts, self_verification_template

def get_yes_no_words(keywords):
    return (keywords['yes'], keywords['no'])