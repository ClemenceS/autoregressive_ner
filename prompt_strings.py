strings={
    "en":{
        "task_introduction" : "The task is to label all mentions of {ner_tag_plural} in a sentence, by putting them in a specific format. {ner_tag_description} Here are some examples:\n",
        "task_introduction_youre_a_specialist" : "You are an excellent {specialist}. You can identify all the mentions of {ner_tag_plural} in a sentence, by putting them in a specific format. {ner_tag_description} Here are some examples you can handle:\n",
        "ask": "Identify all the mentions of {ner_tag_plural} in the following sentence, by putting \"{begin_tag}\" in front and a \"{end_tag}\" behind each of them.\n",
        "ask_youre_a_specialist": "Continue. Identify all the mentions of {ner_tag_plural} in the following sentence, by putting \"{begin_tag}\" in front and a \"{end_tag}\" behind each of them.\n",
        "ask_listing": "Identify all the mentions of {ner_tag_plural} in the following sentence, by listing them, separated by \"{list_separator}\".\n",
        "ask_listing_youre_a_specialist": "Continue. Identify all the mentions of {ner_tag_plural} in the following sentence, by listing them, separated by \"{list_separator}\".\n",
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
            'FUNC' : "functions and jobs",
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
            "EVENT" : "events",
            "TIMEX3" : "time expressions",
            "RML" : "results and measurements",
            "ACTOR" : "actors",
            "Abbreviation" : "abbreviations",
            "Body_Part" : "body parts",
            "Clinical_Finding" : "clinical findings",
            "Diagnostic_Procedure" : "diagnostic procedures",
            "Disease" : "diseases",
            "Family_Member" : "family members",
            "Laboratory_or_Test_Result" : "laboratory or test results",
            "Laboratory_Procedure" : "laboratory procedures",
            "Medication" : "medications",
            "Procedure" : "procedures",
            "Sign_or_Symptom" : "signs or symptoms",
            "Therapeutic_Procedure" : "therapeutic procedures",
            "CompositeMention" : "composite mentions of diseases",
            "DiseaseClass" : "disease classes",
            "Modifier" : "modifiers",
            "SpecificDisease" : "diseases",
            },
        'ner_tags_names' : {
            'PER' : "a person's name",
            "FAC" : "a facility",
            'LOC' : "a location",
            'ORG' : "an organization",
            'FUNC' : "a function or a job",
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
            "EVENT" : "an event",
            "TIMEX3" : "a time expression",
            "RML" : "a result or a measurement",
            "ACTOR" : "an actor",
            "Abbreviation" : "an abbreviation",
            "Body_Part" : "a body part",
            "Clinical_Finding" : "a clinical finding",
            "Diagnostic_Procedure" : "a diagnostic procedure",
            "Disease" : "a disease",
            "Family_Member" : "a family member",
            "Laboratory_or_Test_Result" : "a laboratory or test result",
            "Laboratory_Procedure" : "a laboratory procedure",
            "Medication" : "a medication",
            "Procedure" : "a procedure",
            "Sign_or_Symptom" : "a sign or symptom",
            "Therapeutic_Procedure" : "a therapeutic procedure",
            "CompositeMention" : "a composite mention of diseases",
            "DiseaseClass" : "a disease class",
            "Modifier" : "a modifier of diseases",
            "SpecificDisease" : "a disease",

            },
        'ner_tags_description' : {
            'PER' : "These are names of persons such as real people or fictional characters.",
            "FAC" : "These are names of man-made structures such as infrastructure, buildings and monuments.",
            "LOC" : "These are names of geographical locations such as landmarks, cities, countries and regions.",
            "ORG" : "These are names of organizations such as companies, agencies and political parties.",
            "FUNC" : "These are words that refer to a profession or a job.",
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
            "OCCU" : "These are words that refer to a professional occupation or discipline.",
            "ORGA" : "These are words that refer to an organization such as healthcare related organizations.",
            "PHEN" : "These are words that refer to a phenomenon that occurs naturally or as a result of an activity, such as a biologic function.",
            "PHYS" : "These are words that refer to any element that contributes to the mechanical, physical and biochemical functioning or organization of living organisms and their components.",
            "PROC" : "These are words that refer to an activity or a procedure that contributes to the diagnosis or treatment of patients, the information of patients, the training of medical personnel or biomedical research.",
            "EVENT" : "These are words that refer to actions, states, and circumstances that are relevant to the clinical history of a patient such as pathologies and symptoms, or more generally words like \"enters\", \"reports\" or \"continue\".",
            "TIMEX3" : "These are time expressions such as dates, times, durations, frequencies, or intervals.",
            "RML" : "These are test results, results of laboratory analyses, formulaic measurements, and measure values.",
            "ACTOR" : "These are words that refer patients, healthcare professionals, or other actors relevant to the clinical history of a patient.",
            "Abbreviation" : "These are words that refer to abbreviations.",
            "Body_Part" : "These are words that refer to organs and anatomical parts of persons.",
            "Clinical_Finding" : "These are words that refer to observations, judgments or evaluations made about patients.",
            "Diagnostic_Procedure" : "These are words that refer to tests that allow determining the condition of the individual.",
            "Disease" : "These are words that describe an alteration of the physiological state in one or several parts of the body, due to generally known causes, manifested by characteristic symptoms and signs, and whose evolution is more or less predictable.",
            "Family_Member" : "These are words that refer to family members.",
            "Laboratory_or_Test_Result" : "These are words that refer to any measurement or evaluation obtained from a diagnostic support examination.",
            "Laboratory_Procedure" : "These are words that refer to tests that are performed on various patient samples that allow diagnosing diseases by detecting biomarkers and other parameters. Blood, urine, and other fluids and tissues that use biochemical, microbiological and/or cytological methods are considered.",
            "Medication" : "These are words that refer to medications or drugs used in the treatment and/or prevention of diseases, including brand names and generics, as well as names for groups of medications.",
            "Procedure" : "These are words that refer to activities derived from the care and care of patients.",
            "Sign_or_Symptom" : "These are words that refer to manifestations of a disease, determined by medical examination or perceived and expressed by the patient.",
            "Therapeutic_Procedure" : "These are words that refer to activities or treatments that are used to prevent, repair, eliminate or cure the individual's disease.",
            "CompositeMention" : "These are words that refer to mentions of multiple diseases, such as \"colorectal, endometrial, and ovarian cancers\".",
            "DiseaseClass" : "These are words that refer to classes of diseases, such as \"an autosomal recessive disease\".",
            "Modifier" : "These are words that refer to modifiers of diseases, such as \"primary\" or \"C7-deficient\".",
            "SpecificDisease" : "These are words that refer to specific diseases, such as \"diastrophic dysplasia\".",
            }
    },
    "fr":{
        "task_introduction" : "La tâche est d'identifier toutes les mentions {ner_tag_plural} dans une phrase, en les mettant en forme. {ner_tag_description} Voici quelques exemples :\n",
        "task_introduction_youre_a_specialist" : "Tu es un excellent {specialist}. Tu sais identifier toutes les mentions {ner_tag_plural} dans une phrase, en les mettant en forme. {ner_tag_description} Voici quelques exemples que tu peux traiter :\n",
        "ask": "Identifie toutes les mentions {ner_tag_plural} dans la phrase suivante, en mettant \"{begin_tag}\" devant et \"{end_tag}\" derrière chacune d'entre elles.\n",
        "ask_youre_a_specialist": "Continue. Identifie toutes les mentions {ner_tag_plural} dans la phrase suivante, en mettant \"{begin_tag}\" devant et \"{end_tag}\" derrière chacune d'entre elles.\n",
        "ask_listing": "Identifie toutes les mentions {ner_tag_plural} dans la phrase suivante, en les listant, séparées par \"{list_separator}\".\n",
        "ask_listing_youre_a_specialist": "Continue. Identifie toutes les mentions {ner_tag_plural} dans la phrase suivante, en les listant, séparées par \"{list_separator}\".\n",
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
            "FUNC" : "de fonctions et métiers",
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
            "EVENT" : "d'événements",
            "TIMEX3" : "d'expressions temporelles",
            "RML" : "de résultats et mesures",
            "ACTOR" : "d'acteurs",
        },
        'ner_tags_names' : {
            'PER' : "un nom de personne",
            'LOC' : "un lieu",
            'ORG' : "une organisation",
            "FAC" : "une production humaine",
            "FUNC" : "une fonction ou un métier",
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
            "EVENT" : "un événement",
            "TIMEX3" : "une expression temporelle",
            "RML" : "un résultat ou une mesure",
            "ACTOR" : "un acteur",
        },
        'ner_tags_description' : {
            'PER' : "Il s'agit des noms de personnes, qu'elles soient réelles ou fictives.",
            "LOC" : "Il s'agit des noms de lieux comme des endroits, villes, pays ou régions.",
            "ORG" : "Il s'agit des noms d'organisations comme des entreprises, des agences ou des partis politiques.",
            "FAC" : "Il s'agit des noms de structures faites par les humains comme des infrastructures, des bâtiments ou des monuments.",
            "FUNC" : "Il s'agit de mots qui se rapportent à une activité professionnelle.",
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
            "EVENT" : "Il s'agit d'une action, d’un état ou d’une circonstance qui est pertinent pour l’histoire clinique d’un patient. Il peut s’agir de pathologies et symptômes, ou plus généralement de mots comme \"entre\", \"rapporte\" ou \"continue\".",
            "TIMEX3" : "Il s'agit d’expressions temporelles comme des dates, heures, durées, fréquences, ou intervalles.",
            "RML" : "Il s'agit de résultats d’analyses de laboratoire, de mesures formelles, et de valeurs de mesure.",
            "ACTOR" : "Il s'agit de patients, de professionnels de santé, ou d’autres acteurs pertinents pour l’histoire clinique d’un patient.",
        },
    },
    "es":{
        "task_introduction" : "La tarea es identificar todas las menciones de {ner_tag_plural} en una oración, poniéndolas en un formato específico. {ner_tag_description} Aquí hay algunos ejemplos:\n",
        "task_introduction_youre_a_specialist" : "Eres un excelente {specialist}. Puedes identificar todas las menciones de {ner_tag_plural} en una oración, poniéndolas en un formato específico. {ner_tag_description} Aquí hay algunos ejemplos que puedes manejar:\n",
        "ask": "Identifica todas las menciones de {ner_tag_plural} en la siguiente oración, poniendo \"{begin_tag}\" delante y \"{end_tag}\" detrás de cada una de ellas.\n",
        "ask_youre_a_specialist": "Continúa. Identifica todas las menciones de {ner_tag_plural} en la siguiente oración, poniendo \"{begin_tag}\" delante y \"{end_tag}\" detrás de cada una de ellas.\n",
        "ask_listing": "Identifica todas las menciones de {ner_tag_plural} en la siguiente oración, listándolas, separadas por \"{list_separator}\".\n",
        "ask_listing_youre_a_specialist": "Continúa. Identifica todas las menciones de {ner_tag_plural} en la siguiente oración, listándolas, separadas por \"{list_separator}\".\n",
        "input_word": "Entrada: ",
        "input_dash": "- ",
        "output_word": "Salida: ",
        "output_dash": "- ",
        "task_introduction_self_verif": "La tarea es verificar si una palabra es una mención de {ner_tag_sing} en una oración. {ner_tag_description} Aquí hay algunos ejemplos:\n",
        "task_introduction_self_verif_youre_a_specialist": "Eres un excelente {specialist}. Puedes verificar si una palabra es una mención de {ner_tag_sing} en una oración. {ner_tag_description} Aquí hay algunos ejemplos que puedes manejar:\n",
        "self_verif_template": "En la oración \"{{sentence}}\", ¿es \"{{word}}\" {ner_tag_sing}?\n",
        "yes_short": "Sí",
        "no_short": "No",
        "yes_long": "{word} es {ner_tag_sing}, sí.",
        "no_long": "{word} no es {ner_tag_sing}, no.",
        "ner_tags_names_in_plural" : {
            'PER' : "nombres de personas",
            "FAC" : "instalaciones",
            'LOC' : "lugares",
            'ORG' : "organizaciones",
            "ACTI" : "actividades y comportamientos",
            'ANAT' : "anatomía",
            "CHEM" : "productos químicos y medicamentos",
            "CONC" : "conceptos e ideas",
            "DEVI" : "dispositivos médicos",
            'DISO' : "trastornos",
            'GENE' : "genes y secuencias moleculares",
            "GEOG" : "áreas geográficas",
            "LIVB" : "seres vivos",
            "OBJC" : "objetos",
            "OCCU" : "ocupaciones",
            "ORGA" : "organizaciones",
            "PHEN" : "fenómenos",
            "PHYS" : "fisiología",
            "PROC" : "procedimientos",
            "EVENT" : "eventos",
            "TIMEX3" : "expresiones de tiempo",
            "RML" : "resultados y mediciones",
            "ACTOR" : "actores",
            "Abbreviation" : "abreviaciones",
            "Body_Part" : "partes del cuerpo",
            "Clinical_Finding" : "hallazgos clínicos",
            "Diagnostic_Procedure" : "procedimientos diagnósticos",
            "Disease" : "enfermedades",
            "Family_Member" : "miembros de la familia",
            "Laboratory_or_Test_Result" : "resultados de exámenes de laboratorio u otras pruebas",
            "Laboratory_Procedure" : "procedimientos de laboratorio",
            "Medication" : "medicamentos o drogas",
            "Procedure" : "procedimientos",
            "Sign_or_Symptom" : "signos o síntomas",
            "Therapeutic_Procedure" : "procedimientos terapéuticos",
            },
        'ner_tags_names' : {
            'PER' : "un nombre de persona",
            "FAC" : "una instalación",
            'LOC' : "un lugar",
            'ORG' : "una organización",
            "ACTI" : "una actividad o comportamiento",
            'ANAT' : "una anatomía",
            "CHEM" : "un producto químico o un medicamento",
            "CONC" : "un concepto o una idea",
            "DEVI" : "un dispositivo",
            'DISO' : "un trastorno",
            'GENE' : "un gen o una secuencia molecular",
            "GEOG" : "un área geográfica",
            "LIVB" : "un ser vivo",
            "OBJC" : "un objeto",
            "OCCU" : "una ocupación",
            "ORGA" : "una organización",
            "PHEN" : "un fenómeno",
            "PHYS" : "una fisiología",
            "PROC" : "un procedimiento",
            "EVENT" : "un evento",
            "TIMEX3" : "una expresión de tiempo",
            "RML" : "un resultado o una medida",
            "ACTOR" : "un actor",
            "Abbreviation" : "una abreviación",
            "Body_Part" : "una parte del cuerpo",
            "Clinical_Finding" : "un hallazgo clínico",
            "Diagnostic_Procedure" : "un procedimiento diagnóstico",
            "Disease" : "una enfermedad",
            "Family_Member" : "un miembro de la familia",
            "Laboratory_or_Test_Result" : "un resultado de un examen de laboratorio u otra prueba",
            "Laboratory_Procedure" : "un procedimiento de laboratorio",
            "Medication" : "un medicamento o una droga",
            "Procedure" : "un procedimiento",
            "Sign_or_Symptom" : "un signo o un síntoma",
            "Therapeutic_Procedure" : "un procedimiento terapéutico",
            },
        'ner_tags_description' : {
            'PER' : "Estos son nombres de personas, ya sean reales o personajes ficticios.",
            "FAC" : "Estos son nombres de estructuras hechas por el hombre como infraestructura, edificios y monumentos.",
            "LOC" : "Estos son nombres de ubicaciones geográficas como hitos, ciudades, países y regiones.",
            "ORG" : "Estos son nombres de organizaciones como empresas, agencias y partidos políticos.",
            "ACTI" : "Estas son palabras que se refieren a actividades humanas, comportamientos o eventos, así como actividades gubernamentales o regulatorias.",
            "ANAT" : "Estas son palabras que se refieren a la estructura del cuerpo humano, sus órganos y su posición, como partes del cuerpo u órganos, sistemas, tejidos, células, sustancias corporales y estructuras embrionarias.",
            "CHEM" : "Estas son palabras que se refieren a una sustancia o composición que tiene una característica química, especialmente una propiedad curativa o preventiva con respecto a las enfermedades humanas o animales, como medicamentos, antibióticos, proteínas, hormonas, enzimas y sustancias peligrosas o venenosas.",
            "CONC" : "Estas son palabras que se refieren a un concepto o una idea, como una clasificación, un producto intelectual, un idioma, una ley o un reglamento.",
            "DEVI" : "Estas son palabras que se refieren a un dispositivo médico utilizado para administrar atención o realizar investigaciones médicas.",
            "DISO" : "Estas son palabras que se refieren a una alteración de la morfología, la función o la salud de un organismo vivo, animal o vegetal, como anomalías congénitas, disfunción, lesiones, signos o síntomas u observaciones.",
            "GENE" : "Estas son palabras que se refieren a un gen, un genoma o una secuencia molecular.",
            "GEOG" : "Estas son palabras que se refieren a un país, una región o una ciudad.",
            "LIVB" : "Estas son palabras que se refieren a un ser vivo o un grupo de seres vivos, como una persona o un grupo de personas, una planta o una categoría de plantas, un animal o una categoría de animales.",
            "OBJC" : "Estas son palabras que se refieren a cualquier cosa animada o inanimada que afecte los sentidos, como objetos físicos fabricados.",
            "OCCU" : "Estas son palabras que se refieren a una ocupación o disciplina profesional.",
            "ORGA" : "Estas son palabras que se refieren a una organización, por ejemplo organizaciones relacionadas con la salud.",            
            "PHEN" : "Estas son palabras que se refieren a un fenómeno que ocurre naturalmente o como resultado de una actividad, por ejemplo una función biológica.",
            "PHYS" : "Estas son palabras que se refieren a cualquier elemento que contribuya al funcionamiento mecánico, físico y bioquímico o la organización de los organismos vivos y sus componentes.",
            "PROC" : "Estas son palabras que se refieren a una actividad o un procedimiento que contribuye al diagnóstico o tratamiento de pacientes, la información de pacientes, la capacitación del personal médico o la investigación biomédica.",
            "EVENT" : "Estas son palabras que se refieren a acciones, estados y circunstancias que son relevantes para la historia clínica de un paciente, como patologías y síntomas, o más generalmente palabras como \"entra\", \"reporta\" o \"continúa\".",
            "TIMEX3" : "Estas son expresiones de tiempo como fechas, horas, duraciones, frecuencias o intervalos.",
            "RML" : "Estos son resultados de análisis de laboratorio, mediciones formales y valores de medición.",
            "ACTOR" : "Estas son palabras que se refieren a pacientes, profesionales de la salud u otros actores relevantes para la historia clínica de un paciente.",
            "Abbreviation" : "Estas son los casos de siglas y acrónimos.",
            "Body_Part" : "Estas son palabras que se refieren a òrganos y partes anatómicas de personas.",
            "Clinical_Finding" : "Estas son palabras que se refieren a observaciones, juicios o evaluaciones que se hacen sobre los pacientes.",
            "Diagnostic_Procedure" : "Estas son palabras que se refieren a exámenes que permiten determinar la condición del individuo.",
            "Disease" : "Estas son palabras que describen una alteración del estado fisiológico en una o varias partes del cuerpo, por causas en general conocidas, manifestada por síntomas y signos característicos, y cuya evolución es más o menos previsible.",
            "Family_Member" : "Estas son palabras que se refieren a miembros de la familia.",
            "Laboratory_or_Test_Result" : "Estas son palabras que se refieren a cualquier medición o evaluación obtenida a partir de un exámen de apoyo diagnóstico.",
            "Laboratory_Procedure" : "Estas son palabras que se refieren a exámenes que se realizan en diversas muestras de pacientes que permiten diagnosticar enfermedades mediante la detección de biomarcadores y otros parámetros. Se consideran los análisis de sangre, orina, y otros fluidos y tejidos que emplean métodos bioquímicos, microbiológicos y/o citológicos.",
            "Medication" : "Estas son palabras que se refieren a medicamentos o drogas empleados en el tratamiento y/o prevención de enfermedades, incluyendo marcas comerciales y genéricos, así como también nombres para grupos de medicamentos.",
            "Procedure" : "Estas son palabras que se refieren a actividades derivadas de la atención y el cuidado de los pacientes.",
            "Sign_or_Symptom" : "Estas son palabras que se refieren a manifestaciones de una enfermedad, determinadas mediante la exploración médica o percibidas y expresadas por el paciente.",
            "Therapeutic_Procedure" : "Estas son palabras que se refieren a actividades o tratamientos que es empleado para prevenir, reparar, eliminar o curar la enfrmedad del individuo.",
            },
    },
}

def get_prompt_strings(
        language : str,
        youre_a_specialist : bool,
        label_description : bool,
        ask : bool,
        long_answer : bool,
        dash : bool,
        listing : bool,
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
        if youre_a_specialist:
            if listing:
                result["ask"] = strings[language]["ask_listing_youre_a_specialist"]
            else:
                result["ask"] = strings[language]["ask_youre_a_specialist"]
        else:
            if listing:
                result["ask"] = strings[language]["ask_listing"]
            else:
                result["ask"] = strings[language]["ask"]
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