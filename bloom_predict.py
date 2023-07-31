from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
from tqdm import tqdm


def bloom_predict(prompts, args, logger, params):
    if args.api_inference or args.model_name == "bigscience/bloom":
        #use huggingface inference API
        logger.info("Generating...")
        API_URL = "https://api-inference.huggingface.co/models/"+args.model_name
        headers = {"Authorization": "Bearer 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"}
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            try:
                return response.json()
            except:
                return {"error":response.text}
        outputs = []
        for i in tqdm(range(len(prompts))):
            output = query({"inputs":prompts[i],"parameters":params})
            nb_retries = 0
            while 'error' in output and nb_retries < 10:
                output = query({"inputs":prompts[i],"parameters":params})
                nb_retries += 1
            if 'error' in output:
                outputs.append('')
                print(output)
            else:
                outputs.append(output[0]['generated_text'])               
    else:
        logger.info("Tokenizing...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Generating...")
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        model.eval()

        outputs = []
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            input_ids_batch = input_ids[i:i+args.batch_size].to(device)
            output = model.generate(input_ids_batch, max_new_tokens=100, do_sample=True, top_p=params['top_p'], top_k=params['top_k'], temperature=params['temperature'], return_full_text=params['return_full_text'])
            output = output[:,input_ids_batch.size(1):]
            outputs += output.tolist()
            
        logger.info("Decoding...")
        outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return outputs