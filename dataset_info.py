ner_tags_by_dataset = {
    "WikiNER" : ["PER", "LOC", "ORG"],
    "conll2003" : ["PER", "LOC", "ORG"],
    "conll2002" : ["PER", "LOC", "ORG"],
    "medline" : ["ANAT", "CHEM", "DEVI", "DISO", "GEOG", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
    "emea" : ["ANAT", "CHEM", "DEVI", "DISO", "GEOG", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
    "n2c2" : ["ACTI", "ANAT", "CHEM", "CONC", "DEVI", "DISO", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
}
colnames_by_hf_dataset = {
    "WikiNER" : ("id", "words", "ner_tags"),
    "conll2003" : ("id", "tokens", "ner_tags"),
    "conll2002" : ("id", "tokens", "ner_tags"),
}
tag_map_by_hf_dataset = {
    "WikiNER" : {0: "O", 1: "LOC", 2: "PER", 3: "FAC", 4: "ORG", }, 
    "conll2003" : {0: "O", 1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC", 7: "O", 8: "O", },
    "conll2002" : {0: "O", 1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC", 7: "O", 8: "O", },
}
language_by_dataset = {
    "WikiNER/en" : "en",
    "WikiNER/fr" : "fr",
    "WikiNER/es" : "es",
    "conll2003" : "en",
    "conll2002" : "es",
    "medline" : "fr",
    "emea" : "fr",
    "n2c2" : "en",
}

def _get_if_key_in_x(dict, x):
    return next((dict[key] for key in dict if key in x), None)

def get_dataset_ner_tags(dataset_name):
    return _get_if_key_in_x(ner_tags_by_dataset, dataset_name)

def get_dataset_colnames(dataset_name):
    return _get_if_key_in_x(colnames_by_hf_dataset, dataset_name)

def get_dataset_tag_map(dataset_name):
    return _get_if_key_in_x(tag_map_by_hf_dataset, dataset_name)

def get_dataset_language(dataset_name):
    return _get_if_key_in_x(language_by_dataset, dataset_name)