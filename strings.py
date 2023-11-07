strings={
    "en":{
        "task_introduction_objective": "The task is to label all mentions of {ner_tag_plural} in a sentence, by putting them in a specific format. {ner_tag_description} Here are some examples:\n",
        "task_introduction_subjective_annotator": "I am an excellent annotator. I can identify all the mentions of {ner_tag_plural} in the following sentence, by putting them in a specific format. {ner_tag_description} Here are some examples I can handle:\n",
        "task_introduction_subjective_linguist": "I am an excellent linguist. I can identify all the mentions of {ner_tag_plural} in the following sentence, by putting them in a specific format. {ner_tag_description} Here are some examples I can handle:\n",
        "task_introduction_subjective_clinician": "I am an excellent clinician. I can identify all the mentions of {ner_tag_plural} in the following sentence, by putting them in a specific format. {ner_tag_description} Here are some examples I can handle:\n",
        "ask_objective": "Identify all the mentions of {ner_tag_plural} in the following sentence, by putting \"{begin_tag}\" in front and a \"{end_tag}\" behind each of them.\n",
        "ask_subjective": "Imitate me. Identify all the mentions of {ner_tag_plural} in the following sentence, by putting \"{begin_tag}\" in front and a \"{end_tag}\" behind each of them.\n",
        "dont_ask": "",
        "input_word": "Input: ",
        "input_dash": "- ",
        "output_word": "Output: ",
        "output_dash": "- ",
        "task_introduction_self_verif_objective": "The task is to verify whether a given word is a mention of a {ner_tag_sing}. {ner_tag_description} Here are some examples:\n",
        "task_introduction_self_verif_subjective_annotator": "I am an excellent annotator. I can verify whether a given word is a mention of {ner_tag_sing}. {ner_tag_description} Here are some examples:\n",
        "task_introduction_self_verif_subjective_linguist": "I am an excellent linguist. I can verify whether a given word is a mention of {ner_tag_sing}. {ner_tag_description} Here are some examples:\n",
        "task_introduction_self_verif_subjective_clinician": "I am an excellent clinician. I can verify whether a given word is a mention of {ner_tag_sing}. {ner_tag_description} Here are some examples:\n",
        "self_verif_template": "In the sentence \"{{sentence}}\", is \"{{word}}\" {ner_tag_sing}?\n",
        "yes_short": "Yes",
        "no_short": "No",
        "yes_long": "{word} is {ner_tag_sing}, yes.",
        "no_long": "{word} is not {ner_tag_sing}, no.",
        "standard_ner_tags_names_in_plural" : {
            'PER' : "person names",
            'DISO' : "disorders",
            'LOC' : "places",
            'ORG' : "organizations",
            'ANAT' : "parts of the body",
            "LIVB" : "living beings",
            "PROC" : "procedures",
            "FAC" : "facilities",
            "CHEM" : "chemicals",
            "DEVI" : "medical devices",
            "GEOG" : "geographical zones",
            "OBJC" : "non-medical objects",
            "PHEN" : "physiolocal phenomema",
            "PHYS" : "human physiology",
            "ACTI" : "activities",
            "CONC" : "concepts",
            },
        'standard_ner_tags_names' : {
            'PER' : "a person's name",
            'DISO' : "an alteration of the functions of the body",
            'LOC' : "a place",
            'ORG' : "an organization",
            'ANAT' : "a part of the body",
            "LIVB" : "a living being",
            "PROC" : "a procedure",
            "FAC" : "a facility",
            "CHEM" : "a chemical",
            "DEVI" : "a medical device",
            "GEOG" : "a geographical zone",
            "OBJC" : "a non-medical object",
            "PHEN" : "a physiolocal phenomemon",
            "PHYS" : "a human physiology",
            "ACTI" : "an activity",
            "CONC" : "a concept",
            },
        'standard_ner_tags_description' : {
            'PER' : "These are words that refer to the name of a real or fictional person.",
            'DISO' : "These are words that refer to an alteration or abnormality of the functions or health of the body.",
            'LOC' : "These are words that refer to the name of a place.",
            'ORG' : "These are words that refer to the name of an organization.",
            'ANAT' : "These are words that refer to a part of the human body.",
            "LIVB" : "These are words that refer to a living being.",
            "PROC" : "These are words that refer to a medical procedure.",
            "FAC" : "These are words that refer to a facility made by humans.",
            "CHEM" : "These are words that refer to a drug or a chemical substance.",
            "DEVI" : "These are words that refer to a medical device or a medical instrument.",
            "GEOG" : "These are words that refer to a geographical zone.",
            "OBJC" : "These are words that refer to an object that is not necessarily medical.",
            "PHEN" : "These are words that refer to a physiolocal phenomemon or a physiolocal function.",
            "PHYS" : "These are words that refer to a human physiology.",
            "ACTI" : "These are words that refer to an activity.",
            "CONC" : "These are words that refer to a medical concept.",
            },
    },
    "fr":{
        "task_introduction_objective": "La tâche est d'identifier toutes les mentions {ner_tag_name} dans une phrase. {ner_tag_description} Il faut aussi les mettre en forme. Voici quelques exemples :\n",
        "task_introduction_subjective_annotator": "Je suis un excellent annotateur. Je peux identifier toutes les mentions {ner_tag_name} dans la phrase suivante, en les mettant en forme. {ner_tag_description} Voici quelques exemples que je peux traiter :\n",
        "task_introduction_subjective_linguist": "Je suis un excellent linguiste. Je peux identifier toutes les mentions {ner_tag_name} dans la phrase suivante, en les mettant en forme. {ner_tag_description} Voici quelques exemples que je peux traiter :\n",
        "task_introduction_subjective_clinician": "Je suis un excellent clinicien. Je peux identifier toutes les mentions {ner_tag_name} dans la phrase suivante, en les mettant en forme. {ner_tag_description} Voici quelques exemples que je peux traiter :\n",
        "ask_objective": "Identifie les mentions {ner_tag_name} dans la phrase suivante, en mettant \"{begin_tag}\" devant et un \"{end_tag}\" derrière la mention dans la phrase suivante.\n",
        "ask_subjective": "Imite-moi. Identifie les mentions {ner_tag_name} dans la phrase suivante, en mettant \"{begin_tag}\" devant et un \"{end_tag}\" derrière la mention dans la phrase suivante.\n",
        "dont_ask": "",
        "input_word": "Entrée : ",
        "input_dash": "- ",
        "output_word": "Sortie : ",
        "output_dash": "- ",
        "task_introduction_self_verif_objective": "La tâche est de vérifier si un mot est une mention d'{ner_tag_name} dans une phrase. {ner_tag_description} Voici quelques exemples :\n",
        "task_introduction_self_verif_subjective_annotator": "Je suis un excellent annotateur. Je peux vérifier si un mot est une mention d'{ner_tag_name} dans une phrase. {ner_tag_description} Voici quelques exemples :\n",
        "task_introduction_self_verif_subjective_linguist": "Je suis un excellent linguiste. Je peux vérifier si un mot est une mention d'{ner_tag_name} dans une phrase. {ner_tag_description} Voici quelques exemples :\n",
        "task_introduction_self_verif_subjective_clinician": "Je suis un excellent clinicien. Je peux vérifier si un mot est une mention d'{ner_tag_name} dans une phrase. {ner_tag_description} Voici quelques exemples :\n",
        "self_verif_template": "Dans la phrase \"{{sentence}}\", le mot \"{{word}}\" désigne-t-il {ner_tag_sing} ?\n",
        "yes_short": "Oui",
        "no_short": "Non",
        "yes_long": "{word} est {ner_tag_sing}, oui.",
        "no_long": "{word} n'est pas {ner_tag_sing}, non.",
        "standard_ner_tags_names_in_plural" : {
            'PER' : "de noms de personnes",
            'DISO' : "de maladies et symptômes",
            'LOC' : "de lieux",
            'ORG' : "d'organisations",
            'ANAT' : "de parties du corps",
            'LIVB' : "d'êtres vivants",
            'PROC' : "de procédures médicales",
            "FAC" : "d'installations",
            "CHEM" : "de substances chimiques",
            "DEVI" : "d'appareils médicaux",
            "GEOG" : "de zones géographiques",
            "OBJC" : "d'objets non médicaux",
            "PHEN" : "de phénomènes physiologiques",
            "PHYS" : "de physiologie humaine",
        },
        'standard_ner_tags_names' : {
            'PER' : "un nom de personne",
            'DISO' : "une altération des fonctions du corps",
            'LOC' : "un lieu",
            'ORG' : "une organisation",
            'ANAT' : "une partie du corps",
            'LIVB' : "un être vivant",
            'PROC' : "une procédure médicale",
            "FAC" : "une installation",
            "CHEM" : "une substance chimique",
            "DEVI" : "un appareil médical",
            "GEOG" : "une zone géographique",
            "OBJC" : "un objet non médical",
            "PHEN" : "un phénomène physiologique",
            "PHYS" : "une physiologie humaine",
        },
        'standard_ner_tags_description' : {
            'PER' : "Il s'agit des mots faisant mention du nom d'un personne qu'elle soit réelle ou fictive.",
            'DISO' : "Il s'agit des mots faisant mention d'une altération ou une anormalité des fonctions ou de la santé du corps.",
            'LOC' : "Il s'agit des mots faisant mention du nom d'un lieu.",
            'ORG' : "Il s'agit des mots faisant mention du nom d'une organisation.",
            'ANAT' : "Il s'agit des mots faisant mention d'une partie du corps humain.",
            'LIVB' : "Il s'agit des mots faisant mention d'un être vivant.",
            'PROC' : "Il s'agit des mots faisant mention d'une procédure médicale.",
            "FAC" : "Il s'agit des mots faisant mention d'une installation faite/construite par les humains.",
            "CHEM" : "Il s'agit des mots faisant mention d'un médicament ou d'une substance chimique.",
            "DEVI" : "Il s'agit des mots faisant mention d'un appareil médical ou d'un instrument médical.",
            "GEOG" : "Il s'agit des mots faisant mention d'une zone géographique.",
            "OBJC" : "Il s'agit des mots faisant mention d'un objet qui n'est pas nécessairement médical.",
            "PHEN" : "Il s'agit des mots faisant mention d'un phénomène physiologique ou d'une fonction physiologique.",
            "PHYS" : "Il s'agit des mots faisant mention d'une physiologie humaine.",
        },
        "quaero_ner_tags_names_in_plural" : {
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
        "quaero_ner_tags_names" : {
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
        "quaero_ner_tags_description" : {
            "ANAT" : "Il s'agit d'une entité se rapportant à la structure du corps humain, ses organes et leur position. Il s’agit principalement des parties du corpus ou organes (intestin grèle, pouce), des appareils (système digestif ), des tissus (épithélium), des cellules (Cellules épithéliales), des substances corporelles (sueur) et des organismes embryonaires (foetus).",
            "CHEM" : "Il s'agit d'une substance ou composition présentant des propriétés chimiques caractéristiques, en particulier des propriétés curatives ou préventives à l’égard des maladies humaines ou animales. Il s’agit principalement des médicaments disponibles en pharmacie (aspirine, risperidone, viagra), des antibiotiques (pénicilline, vancomycine), des proteines (myoprotéine, immunoglobuline) des hormones (insuline), des substances dangereuses (toxine botulinique, venin), des enzymes (uricase, cytochrome c).",
            "DEVI" : "Il s'agit d'un matériel utilisé pour administrer des soins (Pompe à insuline, Pacemaker, Seringue) ou effectuer des recherches médicales (Micropuce ADN).",
            "DISO" : "Il s'agit d'une altération de la morphologie, des fonctions, ou de la santé d’un organisme vivant, animal ou végétal. Il peut s’agir de malformations (pied bot, dysplasie rétinienne, malformation cardiovasculaire), de maladies (grippe, schizophrénie, syndrome de Marfan), de blessure (fracture du tibia), de signe ou symptome (lombalgie) ou d’une observation (fièvre).",
            "GEOG" : "Il s'agit d'un pays (France), une région (Nouvelle Angleterre), ou une ville (Paris).",
            "LIVB" : "Il s'agit d'un être vivant ou groupe d’êtres vivants. Il peut s’agir d’une personne (Patient, Nouveau-né) ou d’un groupe de personnes (Travailleurs volontaires, Hispaniques, Parents, Transsexuels), d’une plante (Rhubarbe) ou d’une catégorie de végétaux (Algues), d’un animal (Rat) ou d’une catégorie d’animaux (Reptiles).",
            "OBJC" : "Il s'agit de tout ce qui, animé ou inanimé, affecte les sens. Ici, il s’agit principalement d’objets physiques manufacturés(lunettes, véhicule).",
            "PHEN" : "Il s'agit d'un phénomène qui se produit naturellement ou à la suite d’une activité. Il s’agit principalement de fonctions biologiques (digestion, gestation).",
            "PHYS" : "Il s'agit de tout élément contribuant au fonctionnement ou à l’organisation mécanique, physique et biochimique des organismes vivants et de leurs composants (organes, tissus, cellules et organites cellulaires).",
            "PROC" : "Il s'agit d'une activité ou procédure contribuant au diagnostic ou au traitement des patients, à l’information des patients, la formation du personnel médical ou à la recherche biomédicale.",
        },
    }
}

def get_prompt_strings(
        language : str, #source ou non en booléen
        subjective : str,
        ner_tag_source : str,
        ask : bool,
        long_answer : bool,
        dash : bool,
):
    """Parameters:
    - language : str
        "en" or "fr"
    - subjective : str
        speak in the first person as an ("annotator", "linguist", "clinician"), if None, speak objectively
    - ner_tag_source : str
        source of the NER tags names and descriptions, either ["standard" or "quaero"]
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
    if subjective is not None and subjective not in ["annotator", "linguist", "clinician"]:
        raise ValueError("subjective must be None, 'annotator', 'linguist' or 'clinician'")
    if ner_tag_source not in ["standard", "quaero"]:
        raise ValueError("ner_tag_source must be 'standard' or 'quaero'")
    result = {}
    if subjective is None:
        result["task_introduction"] = strings[language]["task_introduction_objective"]
        result["task_introduction_self_verif"] = strings[language]["task_introduction_self_verif_objective"]
    else:
        result["task_introduction"] = strings[language]["task_introduction_subjective_"+subjective]
        result["task_introduction_self_verif"] = strings[language]["task_introduction_self_verif_subjective_"+subjective]
    result["self_verif_template"] = strings[language]["self_verif_template"]
    if ask:
        result["ask"] = strings[language]["ask_"+("objective" if subjective is None else "subjective")]
    else:
        result["ask"] = strings[language]["dont_ask"]
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
    result["ner_tags_names"] = strings[language][ner_tag_source+"_ner_tags_names"]
    result["ner_tags_names_in_plural"] = strings[language][ner_tag_source+"_ner_tags_names_in_plural"]
    result["ner_tags_description"] = strings[language][ner_tag_source+"_ner_tags_description"]

    return result