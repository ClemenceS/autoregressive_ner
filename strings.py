strings={
    "en":{
        "task_introduction" : "The task is to label all mentions of {ner_tag_plural} in a sentence, by putting them in a specific format. {ner_tag_description} Here are some examples:\n",
        "task_introduction_youre_a_specialist" : "You are an excellent {specialist}. You can identify all the mentions of {ner_tag_plural} in a sentence, by putting them in a specific format. {ner_tag_description} Here are some examples you can handle:\n",
        "ask": "Identify all the mentions of {ner_tag_plural} in the following sentence, by putting \"{begin_tag}\" in front and a \"{end_tag}\" behind each of them.\n",
        "ask_youre_a_specialist": "Continue. Identify all the mentions of {ner_tag_plural} in the following sentence, by putting \"{begin_tag}\" in front and a \"{end_tag}\" behind each of them.\n",
        "input_word": "Input: ",
        "input_dash": "- ",
        "output_word": "Output: ",
        "output_dash": "- ",
        "task_introduction_self_verif": "The task is to verify whether a given word is a mention of {ner_tag_sing}. {ner_tag_description} Here are some examples:\n",
        "task_introduction_self_verif_youre_a_specialist": "You are an excellent {specialist}. You can verify whether a given word is a mention of {ner_tag_sing}. {ner_tag_description} Here are some examples you can handle:\n",
        "self_verif_template": "In the sentence \"{{sentence}}\", is \"{{word}}\" {ner_tag_sing}?\n",
        "yes_short": "Yes",
        "no_short": "No",
        "yes_long": "{word} is {ner_tag_sing}, yes.",
        "no_long": "{word} is not {ner_tag_sing}, no.",
        "ner_tags_names_in_plural" : {
            'PER' : "person names",
            "FAC" : "facilities",
            'LOC' : "locations",
            'ORG' : "organizations",
            "ACTI" : "activities and behaviors",
            'ANAT' : "anatomy",
            "CHEM" : "chemicals and drugs",
            "CONC" : "concepts and ideas",
            "DEVI" : "medical devices",
            'DISO' : "disorders",
            'GENE' : "genes and molecular sequences",
            "GEOG" : "geographical areas",
            "LIVB" : "living beings",
            "OBJC" : "objects",
            "OCCU" : "occupations",
            "ORGA" : "organizations",
            "PHEN" : "phenomema",
            "PHYS" : "physiology",
            "PROC" : "procedures",
            },
        'ner_tags_names' : {
            'PER' : "a person's name",
            "FAC" : "a facility",
            'LOC' : "a location",
            'ORG' : "an organization",
            "ACTI" : "an activity or behavior",
            'ANAT' : "an anatomy",
            "CHEM" : "a chemical or a drug",
            "CONC" : "a concept or an idea",
            "DEVI" : "a device",
            'DISO' : "a disorder",
            'GENE' : "a gene or a molecular sequence",
            "GEOG" : "a geographical area",
            "LIVB" : "a living being",
            "OBJC" : "an object",
            "OCCU" : "an occupation",
            "ORGA" : "an organization",
            "PHEN" : "a phenomemon",
            "PHYS" : "a physiology",
            "PROC" : "a procedure",
            },
        'ner_tags_description' : {
            'PER' : "These are names of persons such as real people or fictional characters.",
            "FAC" : "These are names of man-made structures such as infrastructure, buildings and monuments.",
            "LOC" : "These are names of geographical locations such as landmarks, cities, countries and regions.",
            "ORG" : "These are names of organizations such as companies, agencies and political parties.",
            "ACTI" : "These are words that refer to human activities, behaviors or events as well as governmental or regulatory activities.",
            "ANAT" : "These are words that refer to the structure of the human body, its organs and their position, such as body parts or organs, systems, tissues, cells, body substances and embryonic structures.",
            "CHEM" : "These are words that refer to a substance or composition that has a chemical characteristic, especially a curative or preventive property with regard to human or animal diseases, such as drugs, antibiotics, proteins, hormones, enzymes and hazardous or poisonous substances.",
            "CONC" : "These are words that refer to a concept or an idea, such as a classification, an intellectual product, a language, a law or a regulation.",
            "DEVI" : "These are words that refer to a medical device used to administer care or perform medical research.",
            "DISO" : "These are words that refer to an alteration of morphology, function or health of a living organism, animal or plant, such as congenital abnormalities, dysfunction, injuries, signs or symptoms or observations.",
            "GENE" : "These are words that refer to a gene, a genome or a molecular sequence.",
            "GEOG" : "These are words that refer to a country, a region or a city.",
            "LIVB" : "These are words that refer to a living being or a group of living beings, such as a person or a group of persons, a plant or a category of plants, an animal or a category of animals.",
            "OBJC" : "These are words that refer to anything animate or inanimate that affects the senses, such as physical manufactured objects.",
            "OCCU" : "These are words that refer to a professional occipation or discipline.",
            "ORGA" : "These are words that refer to an organization such as healthcare related organizations.",
            "PHEN" : "These are words that refer to a phenomenon that occurs naturally or as a result of an activity, such as a biologic function.",
            "PHYS" : "These are words that refer to any element that contributes to the mechanical, physical and biochemical functioning or organization of living organisms and their components.",
            "PROC" : "These are words that refer to an activity or a procedure that contributes to the diagnosis or treatment of patients, the information of patients, the training of medical personnel or biomedical research.",
            },
    },
    "fr":{
        "task_introduction" : "La tâche est d'identifier toutes les mentions {ner_tag_plural} dans une phrase, en les mettant en forme. {ner_tag_description} Voici quelques exemples :\n",
        "task_introduction_youre_a_specialist" : "Tu es un excellent {specialist}. Tu sais identifier toutes les mentions {ner_tag_plural} dans une phrase, en les mettant en forme. {ner_tag_description} Voici quelues exemples que tu peux traiter :\n",
        "ask": "Identifie toutes les mentions {ner_tag_plural} dans la phrase suivante, en mettant \"{begin_tag}\" devant et \"{end_tag}\" derrière chacune d'entre elles.\n",
        "ask_youre_a_specialist": "Continue. Identifie toutes les mentions {ner_tag_plural} dans la phrase suivante, en mettant \"{begin_tag}\" devant et \"{end_tag}\" derrière chacune d'entre elles.\n",
        "input_word": "Entrée : ",
        "input_dash": "- ",
        "output_word": "Sortie : ",
        "output_dash": "- ",
        "task_introduction_self_verif": "La tâche est de vérifier si un mot est une mention d'{ner_tag_sing} dans une phrase. {ner_tag_description} Voici quelques exemples :\n",
        "task_introduction_self_verif_youre_a_specialist": "Tu es un excellent {specialist}. Tu sais vérifier si un mot est une mention d'{ner_tag_sing} dans une phrase. {ner_tag_description} Voici quelues exemples que tu peux traiter :\n",
        "self_verif_template": "Dans la phrase \"{{sentence}}\", le mot \"{{word}}\" désigne-t-il {ner_tag_sing} ?\n",
        "yes_short": "Oui",
        "no_short": "Non",
        "yes_long": "{word} est {ner_tag_sing}, oui.",
        "no_long": "{word} n'est pas {ner_tag_sing}, non.",
        "ner_tags_names_in_plural" : {
            'PER' : "de noms de personnes",
            "FAC" : "de productions humaines",
            'LOC' : "de lieux",
            'ORG' : "d'organisations",
            "ANAT" : "d'anatomie",
            "CHEM" : "de médicaments et substances chimiques",
            "DEVI" : "de matériel",
            "DISO" : "de problèmes médicaux",
            "GEOG" : "de zones géographiques",
            "LIVB" : "d'êtres vivants",
            "OBJC" : "d'objets",
            "PHEN" : "de phénomènes",
            "PHYS" : "de physiologie",
            "PROC" : "de procédures",
        },
        'ner_tags_names' : {
            'PER' : "un nom de personne",
            'LOC' : "un lieu",
            'ORG' : "une organisation",
            "FAC" : "une production humaine",
            "ANAT" : "une partie du corps",
            "CHEM" : "un médicament ou une substance chimique",
            "DEVI" : "un matériel",
            "DISO" : "un problème médical",
            "GEOG" : "une zone géographique",
            "LIVB" : "un être vivant",
            "OBJC" : "un objet",
            "PHEN" : "un phénomène",
            "PHYS" : "une physiologie",
            "PROC" : "une procédure",
        },
        'ner_tags_description' : {
            'PER' : "Il s'agit des noms de personnes, qu'elles soient réelles ou fictives.",
            "LOC" : "Il s'agit des noms de lieux comme des endroits, villes, pays ou régions.",
            "ORG" : "Il s'agit des noms d'organisations comme des entreprises, des agences ou des partis politiques.",
            "FAC" : "Il s'agit des noms de structures faites par les humains comme des infrastructures, des bâtiments ou des monuments.",
            "ANAT" : "Il s'agit d'une entité se rapportant à la structure du corps humain, ses organes et leur position. Il s’agit principalement des parties du corpus ou organes, des appareils, des tissus, des cellules, des substances corporelles et des organismes embryonaires.",
            "CHEM" : "Il s'agit d'une substance ou composition présentant des propriétés chimiques caractéristiques, en particulier des propriétés curatives ou préventives à l’égard des maladies humaines ou animales. Il s’agit principalement des médicaments disponibles en pharmacie, des antibiotiques, des proteines, des hormones, des substances dangereuses, des enzymes.",
            "DEVI" : "Il s'agit d'un matériel utilisé pour administrer des soins ou effectuer des recherches médicales.",
            "DISO" : "Il s'agit d'une altération de la morphologie, des fonctions, ou de la santé d’un organisme vivant, animal ou végétal. Il peut s’agir de malformations, de maladies, de blessure, de signe ou symptome ou d’une observation.",
            "GEOG" : "Il s'agit d'un pays, une région, ou une ville.",
            "LIVB" : "Il s'agit d'un être vivant ou groupe d’êtres vivants. Il peut s’agir d’une personne ou d’un groupe de personnes, d’une plante ou d’une catégorie de végétaux, d’un animal ou d’une catégorie d’animaux.",
            "OBJC" : "Il s'agit de tout ce qui, animé ou inanimé, affecte les sens. Ici, il s’agit principalement d’objets physiques manufacturés.",
            "PHEN" : "Il s'agit d'un phénomène qui se produit naturellement ou à la suite d’une activité. Il s’agit principalement de fonctions biologiques.",
            "PHYS" : "Il s'agit de tout élément contribuant au fonctionnement ou à l’organisation mécanique, physique et biochimique des organismes vivants et de leurs composants.",
            "PROC" : "Il s'agit d'une activité ou procédure contribuant au diagnostic ou au traitement des patients, à l’information des patients, la formation du personnel médical ou à la recherche biomédicale.",
        },
    }
}

def get_prompt_strings(
        language : str,
        youre_a_specialist : bool,
        label_description : bool,
        ask : bool,
        long_answer : bool,
        dash : bool,
):
    """Parameters:
    - language : str
        "en" or "fr"
    - youre_a_specialist : bool
        if True, start the task by saying "You are an excellent {specialist}."
    - ner_tag_source : str
        length of the description of the NER tags
    - ask : bool
        if True, ask the model to perform the task
    - long_answer : bool
        if True, give a long answer to the self-verification task
    - dash : bool
        if True, add a dash before the input and output sentences
    Output:
    - prompt_strings : dict containing the following keys:
        "task_introduction" : str
        "ask" : str
        "input_intro" : str
        "output_intro" : str
        "task_introduction_self_verif" : str
        "self_verif_template" : str
        "ner_tags_names" : dict
        "ner_tags_names_in_plural" : dict
        "ner_tags_description" : dict
        "yes" : str
        "no" : str
    """
    result = {}
    if youre_a_specialist:
        result["task_introduction"] = strings[language]["task_introduction_youre_a_specialist"]
        result["task_introduction_self_verif"] = strings[language]["task_introduction_self_verif_youre_a_specialist"]
    else:
        result["task_introduction"] = strings[language]["task_introduction"]
        result["task_introduction_self_verif"] = strings[language]["task_introduction_self_verif"]
    result["self_verif_template"] = strings[language]["self_verif_template"]
    if ask:
        result["ask"] = strings[language]["ask" if not youre_a_specialist else "ask_youre_a_specialist"]
    else:
        result["ask"] = ""
    if dash:
        result["input_intro"] = strings[language]["input_dash"]
        result["output_intro"] = strings[language]["output_dash"]
    else:
        result["input_intro"] = strings[language]["input_word"]
        result["output_intro"] = strings[language]["output_word"]
    if long_answer:
        result["yes"] = strings[language]["yes_long"]
        result["no"] = strings[language]["no_long"]
    else:
        result["yes"] = strings[language]["yes_short"]
        result["no"] = strings[language]["no_short"]
    result["ner_tags_names"] = strings[language]["ner_tags_names"]
    result["ner_tags_names_in_plural"] = strings[language]["ner_tags_names_in_plural"]
    if label_description:
        result["ner_tags_description"] = strings[language]["ner_tags_description"]
    else:
        result["ner_tags_description"] = {ner_tag:"" for ner_tag in result["ner_tags_names"]}

    return result