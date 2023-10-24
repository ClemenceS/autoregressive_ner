from fastchat.model import get_conversation_template
    
def fastchat_template(model_name, prompts):
    if 'bloom' in model_name:
        return prompts
    prompts_for_model = []
    for prompt in prompts:
        conv = get_conversation_template(model_name)
        if conv.name != "one_shot":
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompts_for_model.append(conv.get_prompt())
        else:
            prompts_for_model.append(prompt)
    return prompts_for_model

def hf_template(model_name, prompts):
    prompts_for_model = []
    for prompt in prompts:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        chat = [
            {"role": "user", "content": prompt},
        ]
        prompts_for_model.append(tokenizer.apply_chat_template(chat, tokenize=False))
    return prompts_for_model

models = [
    "lmsys/vicuna-13b-v1.5",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "tiiuae/falcon-40b",
]
s = "hello world"
for model in models:
    print(model)
    print("fastchat")
    print(fastchat_template(model, [s]))
    print("hf")
    print(hf_template(model, [s]))
    print()