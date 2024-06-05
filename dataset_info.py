ner_tags_by_dataset = {
    "WikiNER" : ["PER", "LOC", "ORG"],
    "conll2003" : ["PER", "LOC", "ORG"],
    "conll2002" : ["PER", "LOC", "ORG"],
    "medline" : ["ANAT", "CHEM", "DEVI", "DISO", "GEOG", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
    "emea" : ["ANAT", "CHEM", "DEVI", "DISO", "GEOG", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
    "n2c2" : ["ACTI", "ANAT", "CHEM", "CONC", "DEVI", "DISO", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
    # "e3c" : ["ACTOR", "BODYPART", "CLINENTITY", "EVENT", "RML", "TIMEX3"],
    "e3c" : ["ACTOR", "ANAT", "DISO", "EVENT", "RML", "TIMEX3"],
    "cwlc" : ["Abbreviation", "Body_Part", "Clinical_Finding", "Diagnostic_Procedure", "Disease", "Family_Member", "Laboratory_or_Test_Result", "Laboratory_Procedure", "Medication", "Procedure", "Sign_or_Symptom", "Therapeutic_Procedure"],
    "QFP" : ["PER", "LOC", "ORG", "FAC", "FUNC"],
    "NCBI" : ['CompositeMention', 'DiseaseClass', 'Modifier', 'SpecificDisease'],
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
    "QFP" : "fr",
    "n2c2" : "en",
    "e3c_en" : "en",
    "e3c_fr" : "fr",
    "e3c_es" : "es",
    "cwlc" : "es",
    "NCBI" : "en",
}

clinician_datasets = ["medline", "emea", "n2c2", "e3c", "cwlc", "NCBI"]

#include clinician datasets
specialist_name_by_dataset = {
    "en" : {
        k: ("clinician" if k in clinician_datasets else "linguist") for k in ner_tags_by_dataset
    },
    "fr" : {
        k: ("clinicien" if k in clinician_datasets else "linguiste") for k in ner_tags_by_dataset
    },
    "es" : {
        k: ("clínico" if k in clinician_datasets else "lingüista") for k in ner_tags_by_dataset
    },
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

def get_dataset_specialist_name(dataset_name, dataset_language):
    return _get_if_key_in_x(specialist_name_by_dataset, dataset_name)[dataset_language]