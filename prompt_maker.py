import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prompt_strings import get_prompt_strings, strings
    
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

def get_first_prompt_examples_for_all(train_dataset, test_dataset, ner_tag, n_few_shot, one_step, random_seed):
    random.seed(random_seed)
    num_prompts = len(test_dataset)
    few_shots_for_all = []
    def sentences_with_most_occurences(train_dataset, ner_tag, n):
        return sorted(range(len(train_dataset)), key=lambda i: len([ent for ent in train_dataset[i]['entities'] if ent['label'] == ner_tag]), reverse=True)[:n]
    if not one_step:
        few_shots_for_all = [sentences_with_most_occurences(train_dataset, ner_tag, n_few_shot)] * num_prompts
    else :
        #get the k nearest sentences in the training set tf-idf wise
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        transformed_train = tfidf.fit_transform([e['text'] for e in train_dataset])
        transformed_test = tfidf.transform([e['text'] for e in test_dataset])
        similarities = cosine_similarity(transformed_test, transformed_train)
        few_shots_for_all = [sorted(range(len(similarities[i])), key=lambda j: similarities[i][j])[-n_few_shot:] for i in range(num_prompts)]
    return few_shots_for_all

def introduce(keywords, ner_tag, specialist_name):
    return keywords['task_introduction'].format(ner_tag_plural=keywords['ner_tags_names_in_plural'][ner_tag], ner_tag_description=keywords['ner_tags_description'][ner_tag], specialist=specialist_name)

def demonstrate(example, ner_tag, begin_tag, end_tag, keywords):
    prompt = keywords['input_intro']+example2string(example, ner_tag, begin_tag, end_tag, sticked=True, tagged=False)+'\n'
    prompt+= keywords['output_intro']+example2string(example, ner_tag, begin_tag, end_tag, sticked=True, tagged=True)+'\n'
    return prompt

def ask(example, ner_tag, begin_tag, end_tag, keywords):
    prompt = keywords['ask'].format(ner_tag_plural=keywords['ner_tags_names_in_plural'][ner_tag], begin_tag=begin_tag, end_tag=end_tag)
    prompt+= keywords['input_intro']+example2string(example, ner_tag, begin_tag, end_tag, sticked=True, tagged=False)+'\n'
    prompt+= keywords['output_intro']
    return prompt

def get_self_verif_examples(train_dataset, ner_tag, n_few_shot, begin_tag, end_tag):
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
    return examples

def get_yes_no_words(prompt_language):
    return (strings[prompt_language]['yes_short'], strings[prompt_language]['no_short'])

def make_prompts(
        train_dataset,
        test_dataset,
        ner_tag,
        begin_tag,
        end_tag,
        n_few_shot,
        one_step,
        random_seed,
        prompt_specialist_name,
        prompt_language,
        prompt_youre_a_specialist,
        prompt_label_description,
        prompt_ask,
        prompt_long_answer,
        prompt_dash,
    ):

    few_shots_for_all = get_first_prompt_examples_for_all(train_dataset, test_dataset, ner_tag, n_few_shot, one_step, random_seed)
    keywords = get_prompt_strings(language=prompt_language, youre_a_specialist=prompt_youre_a_specialist, label_description=prompt_label_description, ask=prompt_ask, long_answer=prompt_long_answer, dash=prompt_dash)

    prompts = []
    for p in range(len(test_dataset)):
        prompt=""
        prompt+=introduce(keywords, ner_tag, prompt_specialist_name)+"\n"
        few_shots= few_shots_for_all[p]
        random.shuffle(few_shots)
        for i in few_shots:
            prompt+=demonstrate(train_dataset[i], ner_tag, begin_tag, end_tag, keywords)
        prompt+=ask(test_dataset[p], ner_tag, begin_tag, end_tag, keywords)
        prompts.append(prompt) 
    
    if one_step:
        return prompts, None
    
    self_verification_template = keywords['task_introduction_self_verif'].format(ner_tag_sing=keywords['ner_tags_names_in_plural'][ner_tag], ner_tag_description=keywords['ner_tags_description'][ner_tag], specialist=prompt_specialist_name)+"\n"
    examples = get_self_verif_examples(train_dataset, ner_tag, n_few_shot, begin_tag, end_tag)
    for example, pred, label in examples:
        self_verification_template+= keywords['self_verif_template'].format(ner_tag_sing=keywords['ner_tags_names'][ner_tag]).format(word=pred,sentence=example,)+keywords[label].format(word=pred, ner_tag_sing=keywords['ner_tags_names'][ner_tag])+"\n"
    self_verification_template+= keywords['self_verif_template'].format(ner_tag_sing=keywords['ner_tags_names'][ner_tag])

    return prompts, self_verification_template
