import os
from datasets import load_dataset
from tqdm import tqdm

#select only the french subset
dataset_fr = load_dataset("meczifho/WikiNER", "fr")
dataset_en = load_dataset("meczifho/WikiNER", "en")

#sort by id
dataset_fr['train'] = dataset_fr['train'].sort(column_names='id')
dataset_fr['test'] = dataset_fr['test'].sort(column_names='id')

dataset_en['train'] = dataset_en['train'].sort(column_names='id')
dataset_en['test'] = dataset_en['test'].sort(column_names='id')

tag_map = {0:"O",1:"LOC",2:"PER",3:"FAC",4:"ORG"}

def example_to_id_text_and_ann(example):
    text = ' '.join(example['words'])
    id_example = example['id']
    ann = []
    i=0
    while i < len(example['ner_tags']):
        tag = example['ner_tags'][i]
        if tag != 0:
            j = i+1
            while j < len(example['ner_tags']) and example['ner_tags'][j] == tag:
                j += 1
            
            ent_type = tag_map[tag]
            ent_id = f"T{len(ann)+1}"
            ent_text = ' '.join(example['words'][i:j])
            ent_begin = len(' '.join(example['words'][:i])) + 1 if i > 0 else 0
            ent_end = ent_begin + len(ent_text)
            ann.append((ent_id, ent_type, ent_begin, ent_end, ent_text))
            i = j
        else:
            i += 1
    return id_example, text, ann

def examples_of_the_same_doc_to_id_text_and_ann(examples):
    examples = sorted(examples, key=lambda x: int(x['id'].split('-')[-1].replace('sent','')))
    big_example = {'id':'-'.join(examples[0]['id'].split('-')[:-1]), 'words':[], 'ner_tags':[]}
    for example in examples:
        big_example['words'] += example['words']
        big_example['ner_tags'] += example['ner_tags']
    return example_to_id_text_and_ann(big_example)


def group_examples_by_doc(examples):
    docs = {}
    for example in examples:
        doc_id = '-'.join(example['id'].split('-')[:-1])
        if doc_id not in docs:
            docs[doc_id] = []
        docs[doc_id].append(example)
    #transform each doc into a single example
    for doc_id in docs:
        docs[doc_id] = examples_of_the_same_doc_to_id_text_and_ann(docs[doc_id])
    #transform the dict into a list sort by id
    docs = [docs[doc_id] for doc_id in sorted(docs, key=lambda x: int(x.split('-')[-1].replace('doc','')))]
    return docs

def assert_entity_text_matches_text(doc_id, doc_text, doc_ann):
    print(doc_id)
    for ent_id, ent_type, ent_begin, ent_end, ent_text in doc_ann:
        assert ent_text == doc_text[ent_begin:ent_end], f"Entity text does not match text: {ent_text} != {doc_text[ent_begin:ent_end]} in doc {doc_id}"

os.makedirs('../WikiNER/fr/training', exist_ok=False)
os.makedirs('../WikiNER/fr/test', exist_ok=False)
os.makedirs('../WikiNER/en/training', exist_ok=False)
os.makedirs('../WikiNER/en/test', exist_ok=False)

for doc in tqdm(group_examples_by_doc([example for example in dataset_fr['train']])):
    id, text, ann = doc
    ann_text = '\n'.join([f"{e[0]}\t{e[1]} {e[2]} {e[3]}\t{e[4]}" for e in ann])
    assert not os.path.exists(f"../WikiNER/fr/training/{id}.txt"), f"File already exists: {id}.txt"
    assert not os.path.exists(f"../WikiNER/fr/training/{id}.ann"), f"File already exists: {id}.ann"
    with open(f"../WikiNER/fr/training/{id}.txt", 'w') as f:
        f.write(text)
    with open(f"../WikiNER/fr/training/{id}.ann", 'w') as f:
        f.write(ann_text)

for doc in tqdm(group_examples_by_doc([example for example in dataset_fr['test']])):
    id, text, ann = doc
    ann_text = '\n'.join([f"{e[0]}\t{e[1]} {e[2]} {e[3]}\t{e[4]}" for e in ann])
    assert not os.path.exists(f"../WikiNER/fr/test/{id}.txt"), f"File already exists: {id}.txt"
    assert not os.path.exists(f"../WikiNER/fr/test/{id}.ann"), f"File already exists: {id}.ann"
    with open(f"../WikiNER/fr/test/{id}.txt", 'w') as f:
        f.write(text)
    with open(f"../WikiNER/fr/test/{id}.ann", 'w') as f:
        f.write(ann_text)

for doc in tqdm(group_examples_by_doc([example for example in dataset_en['train']])):
    id, text, ann = doc
    ann_text = '\n'.join([f"{e[0]}\t{e[1]} {e[2]} {e[3]}\t{e[4]}" for e in ann])
    assert not os.path.exists(f"../WikiNER/en/training/{id}.txt"), f"File already exists: {id}.txt"
    assert not os.path.exists(f"../WikiNER/en/training/{id}.ann"), f"File already exists: {id}.ann"
    with open(f"../WikiNER/en/training/{id}.txt", 'w') as f:
        f.write(text)
    with open(f"../WikiNER/en/training/{id}.ann", 'w') as f:
        f.write(ann_text)

for doc in tqdm(group_examples_by_doc([example for example in dataset_en['test']])):
    id, text, ann = doc
    ann_text = '\n'.join([f"{e[0]}\t{e[1]} {e[2]} {e[3]}\t{e[4]}" for e in ann])
    assert not os.path.exists(f"../WikiNER/en/test/{id}.txt"), f"File already exists: {id}.txt"
    assert not os.path.exists(f"../WikiNER/en/test/{id}.ann"), f"File already exists: {id}.ann"
    with open(f"../WikiNER/en/test/{id}.txt", 'w') as f:
        f.write(text)
    with open(f"../WikiNER/en/test/{id}.ann", 'w') as f:
        f.write(ann_text)
    