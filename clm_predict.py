import os
import re
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from prompt_maker import example2string, make_prompts, get_yes_no_words
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams
import logging
import datetime


MODEL_INSTRUCTION_TEMPLATES = {
    "lmsys/vicuna-13b-v1.5" : "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:",
    "mistralai/Mistral-7B-Instruct-v0.1" : "<s>[INST] {} [/INST]",
    "tiiuae/falcon-40b-instruct":"<|im_start|>user\n{}<|im_end|>\n",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict")

def get_prompts_for_model(model_name, prompts):
    if model_name not in MODEL_INSTRUCTION_TEMPLATES:
        return prompts
    return [MODEL_INSTRUCTION_TEMPLATES[model_name].format(prompt) for prompt in prompts]

def validate_sentence(s, begin_tag, end_tag):
    return (
        begin_tag*(s.count(end_tag)-s.count(begin_tag))
        +s
        +end_tag*(s.count(begin_tag)-s.count(end_tag))
        )

def remove_1st_level_ents(s, begin_tag, end_tag):
    s_res = s
    for e in get_1st_level_ents(s, begin_tag=begin_tag, end_tag=end_tag):
        s_res = s_res.replace(begin_tag+e+end_tag, e)
    return s_res

def get_1st_level_ents(s, begin_tag, end_tag):
    regex_begin_tag = re.escape(begin_tag)
    regex_end_tag = re.escape(end_tag)
    return re.findall(regex_begin_tag+f'([^{begin_tag[0]}]*?)'+regex_end_tag, s)

def get_all_ents(s, begin_tag, end_tag):
    s = validate_sentence(s, begin_tag=begin_tag, end_tag=end_tag)
    entities = []
    entities_this_level = [-1]
    while len(entities_this_level):
        entities_this_level = get_1st_level_ents(s, begin_tag=begin_tag, end_tag=end_tag)
        entities.extend(entities_this_level)
        s = remove_1st_level_ents(s, begin_tag=begin_tag, end_tag=end_tag)
    return entities

def get_indices(ref_sentence, s, begin_tag, end_tag, list_separator=", ", listing=False):
    if not listing:
        # s is a sentence where all entities are surrounded by begin_tag and end_tag
        entities_indices = []
        for e in get_all_ents(s, begin_tag=begin_tag, end_tag=end_tag):
            index = ref_sentence.find(e)
            if index!=-1:
                entities_indices.append((index, index+len(e)))
        return list(set(entities_indices))
    else:
        # s is a list_separator-separated list of entities
        entities_indices = []
        for e in s.split(list_separator):
            index = ref_sentence.find(e)
            if index!=-1:
                entities_indices.append((index, index+len(e)))
        return list(set(entities_indices))

class Newline(StoppingCriteria):
    def __init__(self, check_start, newline_token):
        self.check_start = check_start
        self.newline_token = newline_token
    
    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        return self.newline_token in input_ids[0, self.check_start:]


def predict_for_dataset(
        llm,
        model,
        tokenizer,
        training_data,
        testing_data,
        ner_tags,
        model_name,
        one_step,
        control,
        begin_tag,
        end_tag,
        model_kwargs,
        random_seed,
        listing,
        list_separator,
        **kwargs):
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    first_prompts = []
    self_verif_templates = {}
    if testing_data is None:
        logger.info(f"Making a {len(training_data)}-fold cross validation over the training data for each tag...")
        for ner_tag in ner_tags:
            kf = KFold(n_splits=len(training_data), shuffle=False)
            for i, (train_indices, dev_indices) in enumerate(kf.split(training_data)):
                dev_dataset = [training_data[i] for i in dev_indices]
                train_dataset = [training_data[i] for i in train_indices]
                first_prompts_fold, self_verif_template_fold = make_prompts(
                    train_dataset,
                    dev_dataset,
                    ner_tag,
                    begin_tag=begin_tag,
                    end_tag=end_tag,
                    one_step=one_step,
                    random_seed=random_seed,
                    listing=listing,
                    list_separator=list_separator,
                    **kwargs
                )
                first_prompts.extend(first_prompts_fold)
                self_verif_templates[ner_tag] = self_verif_template_fold
            logger.debug("Here is an example of a {} tag prompt :\n{}".format(ner_tag, first_prompts[-1]))
            logger.debug("Here is an example of a self verification template :\n{}".format(self_verif_templates[ner_tag]))
    else:
        logger.info("{} examples in train set".format(len(training_data)))
        logger.info("{} examples in test set".format(len(testing_data)))
        for ner_tag in ner_tags:
            first_prompts_ner_tag, self_verif_template_ner_tag = make_prompts(
                training_data,
                testing_data,
                ner_tag,
                begin_tag=begin_tag,
                end_tag=end_tag,
                one_step=one_step,
                listing=listing,
                list_separator=list_separator,
                random_seed=random_seed,
                **kwargs
            )
            first_prompts.extend(first_prompts_ner_tag)
            self_verif_templates[ner_tag] = self_verif_template_ner_tag
            logger.debug("Here is an example of a {} tag prompt :\n{}".format(ner_tag, first_prompts[-1]))
            logger.debug("Here is an example of a self verification template :\n{}".format(self_verif_templates[ner_tag]))
    
    reference = testing_data if testing_data is not None else training_data
    newline_token = tokenizer.encode('\n', add_special_tokens=False)[-1]
    eos_token = tokenizer.eos_token_id
    yes_no = get_yes_no_words(prompt_language=kwargs['prompt_language'])
    # yes_tok = tokenizer.encode(yes_no[0],add_special_tokens=False)[0]
    # no_tok = tokenizer.encode(yes_no[1],add_special_tokens=False)[0]
    sticked = True
    begin_tag_toks = tokenizer.encode("@@",add_special_tokens=False)
    if sticked:
        end_tag_toks = tokenizer.encode('@##',add_special_tokens=False)[1:]
    else :
        end_tag_toks = tokenizer.encode("##",add_special_tokens=False)
    if control:
        entries = tokenizer([example['text'].strip()+'\n' for example in reference], add_special_tokens=False).input_ids

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = []
    predictions = [
        {
            'doc_id': example['doc_id'],
            'text': example['text'],
            'entities': [],
        }
        for example in reference
    ]
    if not control:
        model_prompts = get_prompts_for_model(model_name, first_prompts)
        if llm:
            sampling_params = SamplingParams(
                best_of=1,
                stop=['\n'],
                temperature=0.0,
                top_k=-1,
                top_p=1,
                max_tokens=128,
            )
            pre_outputs = llm.generate(model_prompts, sampling_params)
            outputs = [o.outputs[0].text for o in pre_outputs]
        else:
            outputs = []
            for i in tqdm(range(0,len(model_prompts))):
                input_tokens = tokenizer.batch_encode_plus(model_prompts[i:i+1],return_tensors="pt", padding=True)
                for t in input_tokens:
                    if torch.is_tensor(input_tokens[t]):
                        input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
                stopping_criteria = [Newline(check_start=len(input_tokens.input_ids[0]), newline_token=newline_token)]
                generation_config = GenerationConfig.from_dict(model_kwargs)
                output_batch = model.generate(**input_tokens, stopping_criteria=stopping_criteria, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, generation_config=generation_config)
                output_batch = tokenizer.batch_decode(output_batch, skip_special_tokens=True)
                cropped_outputs = [o[len(model_prompts[i]):] for o in output_batch]
                outputs.extend(cropped_outputs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model_prompts = get_prompts_for_model(model_name, first_prompts)
        for i in tqdm(range(0,len(first_prompts))):
            prompt = model_prompts[i]
            input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids
            input_ids = input_ids.to(model.device)
            entry = entries[i%len(reference)]

            nb_open_entites = 0
            all_generated_ids = []
            num_new_tokens = 0
            while input_ids[0,-1] not in [newline_token, eos_token] and len(entry)>1 and num_new_tokens<512:
                output = model.generate(input_ids,max_new_tokens=1,output_scores=True, return_dict_in_generate=True, **model_kwargs)
                scores = output.scores
                next_entry_id = entry[0]
                if nb_open_entites<=0:
                    allowed_tokens = torch.tensor([
                        next_entry_id, 
                        begin_tag_toks[0],
                        eos_token,
                        ]).to(device)
                else:
                    allowed_tokens = torch.tensor([next_entry_id, begin_tag_toks[0], end_tag_toks[0]]).to(device)
                next_scores = scores[0][0][[allowed_tokens]]
                generated_id = allowed_tokens[torch.argmax(next_scores)]
                if generated_id in [next_entry_id, eos_token]:
                    entry = entry[1:]
                    all_generated_ids = [generated_id]
                elif generated_id==begin_tag_toks[0]:
                    nb_open_entites+=1
                    all_generated_ids = begin_tag_toks
                    if sticked:
                        entry = tokenizer.encode("@"+tokenizer.decode(entry), add_special_tokens=False)[1:]
                else:
                    nb_open_entites-=1
                    all_generated_ids = end_tag_toks

                num_new_tokens+=len(all_generated_ids)

                input_ids = torch.cat([input_ids,torch.tensor(all_generated_ids).unsqueeze(0).to(device)], dim=1)
            output_text = tokenizer.decode(input_ids[0],skip_special_tokens=True).replace(prompt,'').strip()
            outputs.append(output_text)
        
    for i, output in enumerate(outputs):
        if i//len(reference)>=len(ner_tags):
            continue
        predictions[i%len(reference)]['entities'].extend([
            {
                'entity_id': 'T{}'.format(len(predictions[i%len(reference)]['entities'])+ent_idx+1),
                'label': ner_tags[i//len(reference)],
                'fragments': [
                    {
                        'begin': begin,
                        'end': end,
                    }],
                'text': reference[i%len(reference)]['text'][begin:end],
            }
            for ent_idx, (begin, end) in enumerate(get_indices(reference[i%len(reference)]['text'], output, begin_tag, end_tag, listing=listing, list_separator=list_separator))
        ])
    verif_prompts = []
    if not one_step:
        sentences = []
        addresses = []
        for i,predicted_example in enumerate(predictions):
            for pred in predicted_example['entities']:
                type = pred['label']
                id = pred['entity_id']
                prompting_sentence = example2string(predicted_example, type, begin_tag, end_tag, sticked=True, tagged=False, listing=listing)
                verification_sentence = self_verif_templates[type].format(word=pred['text'], sentence=prompting_sentence)
                sentences.append(verification_sentence)
                addresses.append((i,id))
        verif_prompts = get_prompts_for_model(model_name, sentences)
        logger.info(f"{len(verif_prompts)} prompts generated for self verification")
        
        sampling_params = SamplingParams(
            stop=['\n'],
            temperature=0.0,
            max_tokens=128,
            top_k=-1,
            top_p=1,
        )
        if llm:
            pre_outputs = llm.generate(verif_prompts, sampling_params)
            verif_outputs = [o.outputs[0].text for o in pre_outputs]
            for i, output in enumerate(verif_outputs):
                if i < len(addresses):
                    if yes_no[1].lower() in output.lower():
                        sent_idx, ent_id = addresses[i]
                        predictions[sent_idx]['entities'] = [ent for ent in predictions[sent_idx]['entities'] if ent['entity_id']!=ent_id]
        else:
            batch_size = 4
            for i in tqdm(range(0,len(verif_prompts),batch_size)):
                batch = verif_prompts[i:i+batch_size]
                input_tokens = tokenizer.batch_encode_plus(batch,return_tensors="pt", padding=True)
                for t in input_tokens:
                    if torch.is_tensor(input_tokens[t]):
                        input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
                stopping_criteria = [Newline(check_start=len(input_tokens.input_ids[0]), newline_token=newline_token)]
                generation_config = GenerationConfig.from_dict(model_kwargs)
                output_batch = model.generate(**input_tokens, stopping_criteria=stopping_criteria, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, generation_config=generation_config)
                output_batch = tokenizer.batch_decode(output_batch, skip_special_tokens=True)
                for j, output in enumerate(output_batch):
                    if yes_no[1].lower() in output.lower():
                        sent_idx, ent_id = addresses[i+j]
                        predictions[sent_idx]['entities'] = [ent for ent in predictions[sent_idx]['entities'] if ent['entity_id']!=ent_id]

    return outputs, predictions, model_prompts[0], (verif_prompts[0] if len(verif_prompts)>0 else None)
    