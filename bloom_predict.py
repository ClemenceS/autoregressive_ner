import re
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch
import requests
from tqdm import tqdm
from fastchat.model import get_conversation_template

def bloom_predict(prompts, api_inference, model_name, batch_size, begin_tag, end_tag, logger, self_verif_template, yes_no, self_verification,
                  model, tokenizer, control, prompt_dict, **kwargs):
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids
        logger.info("Generating...")
        outputs = []
        for i in tqdm(range(len(prompts))):
            conv = get_conversation_template(model_name)
            conv.append_message(conv.roles[0], prompts[i])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        
            input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            if not control:
                output = model.generate(input_ids, max_new_tokens=line_lengths[i]+25, **kwargs)
                output = output[:,len(input_ids[0])-10:]
                output = tokenizer.decode(output[0], skip_special_tokens=True, skip_spaces_between_tokens=False)                
            else:
                entry = tokenizer.encode(prompts[i].split('\n')[-2].replace(prompt_dict['input_intro'],prompt_dict['output_intro']).strip()+'\n', add_special_tokens=False)
                sticked = True
                
                begin_tag_toks = tokenizer.encode(begin_tag,add_special_tokens=False)
                if sticked:
                    end_tag_toks = tokenizer.encode('@'+end_tag,add_special_tokens=False)[1:]
                else :
                    end_tag_toks = tokenizer.encode(end_tag,add_special_tokens=False)
                nb_open_entites = 0
                while input_ids[0,-1]!=tokenizer.eos_token_id and len(entry)>1:
                    output = model.generate(input_ids,max_new_tokens=1,output_scores=True, return_dict_in_generate=True, **kwargs)
                    scores = output.scores
                    sequences = output.sequences
                    next_entry_id = entry[0]
                    #print('================')
                    #print('next_entry id: ',next_entry_id,' word: ',tokenizer.decode(next_entry_id))
                    allowed_tokens = [next_entry_id,begin_tag_toks[0]]
                    if nb_open_entites>0:
                        allowed_tokens.append(end_tag_toks[0])
                    next_scores = {k:v for k,v in zip(allowed_tokens, scores[0][0][[allowed_tokens]])}
                    #next_scores_readble = {k:v for k,v in zip([tokenizer.convert_ids_to_tokens(t) for t in allowed_tokens], [x.item() for x in scores[0][0][[allowed_tokens]]])}
                    #print('----- scores ---------')
                    #print(next_scores_readble)

                    #print('------- generation -------------')    
                    generated_id = max(next_scores, key=next_scores.get)

                    #print(next_entry_id)
                    if generated_id==next_entry_id:
                        entry = entry[1:]
                        all_generated_ids = [generated_id]
                    elif generated_id==begin_tag_toks[0]:
                        nb_open_entites+=1
                        all_generated_ids = begin_tag_toks
                        if sticked:
                            entry = tokenizer.encode("@"+tokenizer.decode(entry), add_special_tokens=False)[1:]
                    else:
                        assert generated_id==end_tag_toks[0], generated_id
                        nb_open_entites-=1
                        all_generated_ids = end_tag_toks

                    generated_words = tokenizer.convert_ids_to_tokens(all_generated_ids)
                    #print('generated id: ',all_generated_ids, 'word:',generated_words)
                    input_ids = torch.cat([input_ids,torch.tensor(all_generated_ids).unsqueeze(0).to(device)], dim=1)


                output = tokenizer.decode(sequences[0],skip_special_tokens=True).replace(prompt,'').strip()
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
        entities = re.findall(regex_begin_tag+'(.*?)'+regex_end_tag, o)
        predictions.append(entities)
    
    if self_verification:
        logger.info("Self verifying...")
        verified_predictions = []
        for prompt, prompt_predictions in tqdm(zip(prompts, predictions)):
            sentence = prompt.split('\n')[-2].split(':')[1].strip()
            prompt_verified_predictions = []
            for pred in prompt_predictions:
                verification_prompt = self_verif_template.format(sentence=sentence, word=pred)
                conv = get_conversation_template(model_name)
                conv.append_message(conv.roles[0], verification_prompt)
                conv.append_message(conv.roles[1], None)
                verification_prompt = conv.get_prompt()
                input_ids = tokenizer(verification_prompt, padding=True, return_tensors="pt").input_ids
                input_ids = input_ids.to(device)
                answer = model.generate(input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                yes_tok = tokenizer.encode(yes_no[0],add_special_tokens=False)[0]
                no_tok = tokenizer.encode(yes_no[1],add_special_tokens=False)[0]
                allowed_tokens = [yes_tok, no_tok]
                scores = answer.scores[0][0][[allowed_tokens]]
                if scores[0] > scores[1]:
                    prompt_verified_predictions.append(pred)
                #answer = tokenizer.decode(answer[0], skip_special_tokens=True, skip_spaces_between_tokens=False)
                #if yes_no[0] in answer:
                #    prompt_verified_predictions.append(pred)
                #elif yes_no[1] not in answer:
                #    print("Self verification failed for prompt: {}\nAnswer: {}".format(verification_prompt, answer))
            verified_predictions.append(prompt_verified_predictions)
        predictions = verified_predictions



    return predictions, outputs
        
    