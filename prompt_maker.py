import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def example2string(example, ner_tag_id, begin_tag, end_tag, tagged=True):
    # if ner_tag_id = 3 and 3 stands for LOC, beginning tag = @@ and ending tag = ##
    # and the example is {'id': 0, 'words': ['I', 'love', 'Paris', 'and', 'Berlin'], 'ner_tags': [0, 0, 3, 0, 3]}
    # the returned string will be 'I love @@Paris## and @@Berlin##'

    #make sure ner_tag_id is an int
    assert isinstance(ner_tag_id, int)
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
    

def make_prompts(train_dataset, test_dataset, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag, n_few_shots, criterion, prompt_keywords):
    #this function takes an example and a ner tag and returns a prompt in english
    keywords = prompt_keywords[language]
    few_shots_for_all = []
    num_prompts = len(test_dataset)
    if criterion == 'random':
        for i in range(num_prompts):
            few_shots_for_all.append(random.sample(range(len(train_dataset)), n_few_shots))
    elif criterion == 'most_occurences':
        def sentences_with_most_occurences(train_dataset, ner_tag_id, n):
            counts = [e['ner_tags'].count(ner_tag_id) for e in train_dataset]
            return sorted(range(len(counts)), key=lambda i: counts[i])[-n:]
        few_shots_for_all = [sentences_with_most_occurences(train_dataset, ner_tag_id, n_few_shots)] * num_prompts
    elif criterion == 'closest_tf_idf':
        #get the k nearest sentences in the training set
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        transformed_train = tfidf.fit_transform([e['words' if 'words' in e else 'tokens'] for e in train_dataset])
        transformed_test = tfidf.transform([e['words' if 'words' in e else 'tokens'] for e in test_dataset])
        similarities = cosine_similarity(transformed_test, transformed_train)
        few_shots_for_all = [sorted(range(len(similarities[i])), key=lambda j: similarities[i][j])[-n_few_shots:] for i in range(num_prompts)]

        
    prompts = []
    for p in range(num_prompts):
        example = test_dataset[p]
        prompt = keywords['first_sentence'].format(keywords['domains_jobs'][domain], keywords['ner_tags'][ner_tag])
        few_shots= few_shots_for_all[p]
        random.shuffle(few_shots)
        for i in few_shots:
            prompt+= keywords['input_intro']+example2string(train_dataset[i], ner_tag_id, begin_tag, end_tag, tagged=False)+'\n'
            prompt+= keywords['output_intro']+example2string(train_dataset[i], ner_tag_id, begin_tag, end_tag, tagged=True)+'\n'
        prompt+= keywords['last_sentence'].format(keywords['ner_tags'][ner_tag], begin_tag, end_tag)
        prompt+= keywords['input_intro']+example2string(example, ner_tag_id, begin_tag, end_tag, tagged=False)+'\n'
        prompt+= keywords['output_intro']
        prompts.append(prompt)
    return prompts
