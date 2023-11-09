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

def get_if_key_in_x(dict, x):
    return next((dict[key] for key in dict if key in x), None)