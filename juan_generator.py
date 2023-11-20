from prompt_strings import strings

en_strings = strings['en']
es_strings = strings['es']
fr_strings = strings['fr']

i = 1
sentences = []
words = []
for (name_en, string_en), (name_es, string_es), (name_fr, string_fr) in zip(en_strings.items(), es_strings.items(), fr_strings.items()):
    if "ner_tag" not in name_en:
        word = len(string_en.split()) < 7
        s = f"==================== {len((words if word else sentences))+1} ====================\n"
        s += string_en.strip() + "\n"
        s += '---------------------\n'
        s += string_fr.strip() + "\n"
        s += '---------------------\n'
        s += string_es.strip() + "\n"
        (words if word else sentences).append(s)
    else:
        #string_en/es/fr are dicts of strings
        # for (ner_en, ner_string_en), (ner_es, ner_string_es), (ner_fr, ner_string_fr) in zip(string_en.items(), string_es.items(), string_fr.items()):
        for ner_es, ner_string_es in string_es.items():
            ner_string_en = string_en[ner_es]
            ner_string_fr = string_fr[ner_es] if ner_es in string_fr else ""
            word = len(ner_string_en.split()) < 7
            s = f"==================== {len((words if word else sentences))+1} ====================\n"
            s += ner_string_en.strip() + "\n"
            s += '---------------------\n'
            s += ner_string_fr.strip() + "\n"
            s += '---------------------\n'
            s += ner_string_es.strip() + "\n"
            (words if word else sentences).append(s)

with open("strings.txt", "w") as f:
    f.write("\n".join(words))
    f.write("\n")
    f.write("\n".join(sentences))