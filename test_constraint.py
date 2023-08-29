# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

text = """Je suis un linguiste expert, je sais identifier les mentions des noms de personnes dans une phrase. Il s'agit des mots faisant mention du nom d'un personne qu'elle soit réelle ou fictive. Je peux aussi les mettre en forme. Voici quelques exemples de phrases que je peux traiter :
Entrée : Théodore Maunoir se lie d' amitié avec Louis Appia .
Sortie : @@Théodore Maunoir## se lie d' amitié avec @@Louis Appia## .
Entrée : Député RPR de Seine-et-Marne depuis 1986 et maire de Coulommiers de 1992 à 2008 , Guy Drut est un proche de Jacques Chirac , qui l' a employé comme chargé de mission auprès du premier ministre en 1975-1976 .
Sortie : Député RPR de Seine-et-Marne depuis 1986 et maire de Coulommiers de 1992 à 2008 , @@Guy Drut## est un proche de @@Jacques Chirac## , qui l' a employé comme chargé de mission auprès du premier ministre en 1975-1976 .
Entrée : Simon Caboche réussit cependant à prendre la fuite avec Jean sans Peur .
Sortie : @@Simon Caboche## réussit cependant à prendre la fuite avec @@Jean sans Peur## .
Entrée : Boris Spassky -- Mikhail Tal , Tallinn , 1973 :
Sortie : @@Boris Spassky## -- @@Mikhail Tal## , Tallinn , 1973 :
Entrée : Rabbi Ishmaël , développant les sept règles d' Hillel , exposa treize principes .
Sortie : @@Rabbi Ishmaël## , développant les sept règles d' @@Hillel## , exposa treize principes .
Imite-moi. Identifie les mentions de noms de personnes dans la phrase suivante, en mettant "@@" devant et un "##" derrière la mention dans la phrase suivante.
Entrée : Dublin City University est l' université la plus récente créée en Irlande .
Sortie :"""

input_ids = tokenizer.encode(text, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)


print(output)