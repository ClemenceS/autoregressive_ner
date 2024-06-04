from nlstruct import HuggingfaceNERDataset, BRATDataset
from dataset_info import get_dataset_colnames, get_dataset_tag_map
from prompt_maker import make_prompts, example2string
from nlstruct.data_utils import sentencize

#dataset_name = "/people/mnaguib/n2c2/"
dataset_name = "mnaguib/WikiNER/fr"
doc_id_colname, words_colname, ner_tags_colname = get_dataset_colnames(dataset_name)
dataset = HuggingfaceNERDataset(
    dataset_name=dataset_name,
    tag_map=get_dataset_tag_map(dataset_name),
    doc_id_colname=doc_id_colname,
    words_colname=words_colname,
    ner_tags_colname=ner_tags_colname,
    #load_from_disk=True,
)
dataset.train_data = [e for e in dataset.train_data if e['doc_id'].startswith("fr")]
dataset.test_data = [e for e in dataset.test_data if e['doc_id'].startswith("fr")]
#dataset = BRATDataset(
#        train= f"{dataset_name}/train",
#        val= 0, 
#        test= f"{dataset_name}/test",
#    )

traindev_dataset = []
for e in dataset.train_data:
    sentences = sentencize(e, reg_split=r"(?<=[.|\s])(?:\s+)(?=[A-Z])", entity_overlap="split")
    traindev_dataset.extend([s for s in sentences if len(s['text']) < 512])
test_dataset = []
for e in dataset.test_data:
    sentences = sentencize(e, reg_split=r"(?<=[.|\s])(?:\s+)(?=[A-Z])", entity_overlap="split")
    test_dataset.extend([s for s in sentences if len(s['text']) < 512])

dataset.train_data = traindev_dataset
dataset.test_data = test_dataset
first_prompts_ner_tag, self_verif_template_ner_tag = make_prompts(
    dataset.train_data,
    #dataset.test_data[5:10],
    dataset.test_data[:100],
    ner_tag="PER",
    begin_tag="@@",
    end_tag="##",
    one_step=False,
    random_seed=22,
    prompt_specialist_name="clinicien",
    n_few_shot=40,
    list_separator=", ",
    listing=False,
    prompt_language="fr",
    prompt_youre_a_specialist=True,
    prompt_label_description=True,
    prompt_ask=True,
    prompt_long_answer=False,
    prompt_dash=False,
)
#print prompt
prompt = first_prompts_ner_tag[0]
print(prompt)
#print ideal answer
string = example2string(dataset.test_data[5], ner_tag="DISO", begin_tag="@@",
                          end_tag="##", sticked=True, tagged=True, list_separator=", ", listing=False)
print(string)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#t = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
lengths = []
for prompt in first_prompts_ner_tag:
    lengths.append(len(t.encode(prompt)))
print(sum(lengths)/len(lengths))
#print(self_verif_template_ner_tag)
