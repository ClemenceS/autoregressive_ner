from nlstruct import HuggingfaceNERDataset
from dataset_info import get_dataset_colnames, get_dataset_tag_map
from prompt_maker import make_prompts

dataset_name = "mnaguib/WikiNER/en"

doc_id_colname, words_colname, ner_tags_colname = get_dataset_colnames(dataset_name)
dataset = HuggingfaceNERDataset(
    dataset_name=dataset_name,
    tag_map=get_dataset_tag_map(dataset_name),
    doc_id_colname=doc_id_colname,
    words_colname=words_colname,
    ner_tags_colname=ner_tags_colname,
    load_from_disk=False,
)
dataset.train_data = [e for e in dataset.train_data if e['doc_id'].startswith("en")]
dataset.test_data = [e for e in dataset.test_data if e['doc_id'].startswith("en")]

first_prompts_ner_tag, self_verif_template_ner_tag = make_prompts(
    dataset.train_data,
    dataset.test_data[:10],
    ner_tag="PER",
    begin_tag="@@",
    end_tag="##",
    one_step=True,
    random_seed=42,
    prompt_specialist_name="w",
    n_few_shot=5,
    prompt_language="en",
    prompt_youre_a_specialist=False,
    prompt_label_description=False,
    prompt_ask=False,
    prompt_long_answer=False,
    prompt_dash=False,
)
print(first_prompts_ner_tag[0])