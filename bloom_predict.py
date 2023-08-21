import re
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch
import requests
from tqdm import tqdm

def bloom_predict(prompts, api_inference, model_name, batch_size, begin_tag, end_tag, logger, self_verif_template, yes_no, self_verification, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    
    logger.info("Getting last line lengths...")
    line_lengths = [len(tokenizer.encode(prompt.split('\n')[-2])) for prompt in prompts]

    if api_inference:
        #use huggingface inference API
        logger.info("Generating...")
        API_URL = "https://api-inference.huggingface.co/models/"+model_name
        headers = {"Authorization": "Bearer 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"}
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            try:
                return response.json()
            except:
                return {"error":response.text}
        
        outputs = []
        for i in tqdm(range(len(prompts))):
            output = query({"inputs":prompts[i],"parameters":{**kwargs, "max_new_tokens":line_lengths[i]+25, "return_full_text":False}})
            nb_retries = 0
            while 'error' in output and nb_retries < 10:
                logger.info("Retrying...")
                output = query({"inputs":prompts[i],"parameters":{**kwargs, "max_new_tokens":line_lengths[i]+25, "return_full_text":False}})
                nb_retries += 1
            if 'error' in output:
                outputs.append('')
                logger.error("API inference failed for prompt: {}\nAnswer: {}".format(prompts[i], output))
            else:
                outputs.append(output[0]['generated_text'])               
    elif 'bloom' in model_name:
        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Generating...")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        outputs = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            input_ids_batch = input_ids[i:i+batch_size].to(device)
            output = model.generate(input_ids_batch, max_new_tokens=max(line_lengths[i:i+batch_size])+25, **kwargs)
            output = output[:,input_ids_batch.size(1):]
            outputs += output.tolist()
            
        logger.info("Decoding...")
        outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    elif 'vicuna' in model_name:
        from fastchat.model import load_model, get_conversation_template, add_model_args
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_model(
        model_name,
        device=device,
        num_gpus=1,
        load_8bit=True,
        debug=False,
        )
        #input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids
        logger.info("Converting prompts to conversations...")
        logger.info("Generating...")
        outputs = []
        for i in tqdm(range(len(prompts))):
            # print(prompts[i])
            # print('----------------')
            conv = get_conversation_template(model_name)
            conv.append_message(conv.roles[0], prompts[i])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        
            input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            output = model.generate(input_ids, max_new_tokens=line_lengths[i]+25, **kwargs)

            output = output[:,len(input_ids[0])-10:]
            output = tokenizer.decode(output[0], skip_special_tokens=True, skip_spaces_between_tokens=False)
            outputs.append(output)
            # print(output)
            # print('================')
    else:
        raise NotImplementedError("Model not supported")
            
    predictions = []
    #get predicted entities
    regex_begin_tag = re.escape(begin_tag)
    regex_end_tag = re.escape(end_tag)
    for o in outputs:
        first_line = o.split('\n')[0]
        entities = re.findall(regex_begin_tag+'(.*?)'+regex_end_tag, first_line)
        predictions.append(entities)
    
    if self_verification:
        logger.info("Self verifying...")
        verified_predictions = []
        for prompt, prompt_predictions in tqdm(zip(prompts, predictions)):
            sentence = prompt.split('\n')[-2].split(':')[1].strip()
            prompt_verified_predictions = []
            for pred in prompt_predictions:
                verification_prompt = self_verif_template.format(sentence=sentence, word=pred)
                if api_inference:
                    answer = query({"inputs":verification_prompt,"parameters":{'max_new_tokens':10, "return_full_text":False}})
                    if 'error' in answer:
                        logger.error("Self verification failed for prompt: {}\nAnswer: {}".format(verification_prompt, answer))
                    else:
                        answer = answer[0]['generated_text']
                elif 'bloom' in model_name:
                    input_ids = tokenizer(verification_prompt, padding=True, return_tensors="pt").input_ids
                    input_ids = input_ids.to(device)
                    answer = model.generate(input_ids, max_new_tokens=10)
                    answer = answer[:,input_ids.size(1):]
                    answer = tokenizer.decode(answer[0], skip_special_tokens=True)
                elif 'vicuna' in model_name:
                    conv = get_conversation_template(model_name)
                    conv.append_message(conv.roles[0], verification_prompt)
                    conv.append_message(conv.roles[1], None)
                    verification_prompt = conv.get_prompt()
                    input_ids = tokenizer(verification_prompt, padding=True, return_tensors="pt").input_ids
                    input_ids = input_ids.to(device)
                    answer = model.generate(input_ids, max_new_tokens=10)
                    answer = answer[:,len(input_ids[0])-10:]
                    answer = tokenizer.decode(answer[0], skip_special_tokens=True, skip_spaces_between_tokens=False)
                if yes_no[0] in answer:
                    prompt_verified_predictions.append(pred)
                elif yes_no[1] not in answer:
                    logger.warning("Self verification failed for prompt: {}\nAnswer: {}".format(verification_prompt, answer))
            verified_predictions.append(prompt_verified_predictions)
        predictions = verified_predictions



    return predictions, outputs
        
    