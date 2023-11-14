# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM") 

text = "The patient was prescribed 100mg of aspirin daily for 3 days."
input_ids = tokenizer.encode(text, return_tensors="pt")

output = model.generate(input_ids, max_new_tokens=40)

print(tokenizer.decode(output[0], skip_special_tokens=True))