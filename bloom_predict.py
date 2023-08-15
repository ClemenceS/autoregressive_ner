import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
from tqdm import tqdm

def bloom_predict(prompts, api_inference, model_name, batch_size, begin_tag, end_tag, logger, self_verif_template, yes_no, self_verification, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if api_inference or model_name == "bigscience/bloom":
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
            last_line = prompts[i].split('\n')[-2]
            prompt_length = len(tokenizer.encode(last_line))
            
            output = query({"inputs":prompts[i],"parameters":{**kwargs, "max_new_tokens":prompt_length+25, "return_full_text":False}})
            nb_retries = 0
            while 'error' in output and nb_retries < 10:
                output = query({"inputs":prompts[i],"parameters":{**kwargs, "max_new_tokens":prompt_length+25, "return_full_text":False}})
                nb_retries += 1
            if 'error' in output:
                outputs.append('')
                print(output)
            else:
                outputs.append(output[0]['generated_text'])               
    else:
        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Generating...")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        outputs = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            input_ids_batch = input_ids[i:i+batch_size].to(device)
            output = model.generate(input_ids_batch, max_new_tokens=100, do_sample=True, return_full_text=False, **kwargs)
            output = output[:,input_ids_batch.size(1):]
            outputs += output.tolist()
            
        logger.info("Decoding...")
        outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    predictions = []
    #get predicted entities
    regex_begin_tag = re.escape(begin_tag)
    regex_end_tag = re.escape(end_tag)
    for o in outputs:
        first_line = o.split('\n')[0]
        entities = re.findall(regex_begin_tag+'(.*?)'+regex_end_tag, first_line)
        predictions.append(entities)
    
    if self_verification:
        verified_predictions = []
        for prompt, prompt_predictions in zip(prompts, predictions):
            sentence = prompt.split('\n')[-2].split(':')[1].strip()
            prompt_verified_predictions = []
            for pred in prompt_predictions:
                verification_prompt = self_verif_template.format(sentence, pred)
                answer = query({"inputs":verification_prompt,"parameters":{'max_new_tokens':1, 'return_full_text':False}})
                if 'error' in answer:
                    print(answer)
                else:
                    answer = answer[0]['generated_text']
                # print(verification_prompt)
                # print("----------")
                # print(answer)
                # print("==========")
                if yes_no[0] in answer:
                    prompt_verified_predictions.append(pred)
                elif yes_no[1] not in answer:
                    logger.warning("Self verification failed for prompt: {}".format(prompt))
            verified_predictions.append(prompt_verified_predictions)
        predictions = verified_predictions



    return predictions, outputs
        
    