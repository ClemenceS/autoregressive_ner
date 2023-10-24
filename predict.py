import re
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from prompt_maker import example2string, make_prompts, get_yes_no_words
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

MODEL_INSTRUCTION_TEMPLATES = {
    "lmsys/vicuna-13b-v1.5" : "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:",
    "mistralai/Mistral-7B-Instruct-v0.1" : "<s>[INST] {} [/INST]",
    "tiiuae/falcon-40b":"<|im_start|>user\n{}<|im_end|>\n",
}

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

def get_indices(ref_sentence, s, begin_tag, end_tag):
    entities_indices = []
    for e in get_all_ents(s, begin_tag=begin_tag, end_tag=end_tag):
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


def predict_for_dataset(training_data, testing_data, ner_tags, model_name, logger, control, self_verification, begin_tag, end_tag, model_kwargs, n_gpus,  **kwargs):
    llm = LLM(model_name, tensor_parallel_size=n_gpus)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')

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
                    self_verification=self_verification,
                    **kwargs
                )
                first_prompts.extend(first_prompts_fold)
                self_verif_templates[ner_tag] = self_verif_template_fold
            # logger.info("Here is an example of a {} tag prompt :\n{}".format(ner_tag, first_prompts[-1]))
            # logger.info("Here is an example of a self verification template :\n{}".format(self_verif_templates[ner_tag]))
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
                self_verification=self_verification,
                **kwargs
            )
            first_prompts.extend(first_prompts_ner_tag)
            self_verif_templates[ner_tag] = self_verif_template_ner_tag
        # logger.info("Here is an example of a {} tag prompt :\n{}".format(ner_tag, first_prompts[-1]))
        # logger.info("Here is an example of a self verification template :\n{}".format(self_verif_templates[ner_tag]))
    
    reference = testing_data if testing_data is not None else training_data
    newline_token = tokenizer.encode('\n', add_special_tokens=False)[-1]
    eos_token = tokenizer.eos_token_id
    yes_no = get_yes_no_words(keywords=kwargs['keywords'])
    yes_tok = tokenizer.encode(yes_no[0],add_special_tokens=False)[0]
    no_tok = tokenizer.encode(yes_no[1],add_special_tokens=False)[0]
    sticked = True
    begin_tag_toks = tokenizer.encode("@@",add_special_tokens=False)
    if sticked:
        end_tag_toks = tokenizer.encode('@##',add_special_tokens=False)[1:]
    else :
        end_tag_toks = tokenizer.encode("##",add_special_tokens=False)
    if control:
        entries = tokenizer([example['text'].strip()+'\n' for example in reference], add_special_tokens=False).input_ids

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Generating...")
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
        sampling_params = SamplingParams(
            use_beam_search=model_kwargs["num_beams"]>1,
            best_of=model_kwargs["num_beams"],
            # stop=['\n'],
            temperature=0,
            #no tested yet...
            # temperature=model_kwargs["temperature"] if model_kwargs["do_sample"] else 0,
            # top_p=model_kwargs["top_p"] if model_kwargs["do_sample"] else None,
            # top_k=model_kwargs["top_k"] if model_kwargs["do_sample"] else None,
            max_tokens=128,
        )
        pre_outputs = llm.generate(model_prompts, sampling_params)
        outputs = [o.outputs[0].text for o in pre_outputs]
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model_prompts = get_prompts_for_model(model_name, first_prompts)
        for i in tqdm(range(0,len(first_prompts))):
            prompt = model_prompts[i]
            input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
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
            for ent_idx, (begin, end) in enumerate(get_indices(reference[i%len(reference)]['text'], output, begin_tag, end_tag))
        ])

    if self_verification:
        sentences = []
        addresses = []
        for i,predicted_example in enumerate(predictions):
            for pred in predicted_example['entities']:
                type = pred['label']
                id = pred['entity_id']
                prompting_sentence = example2string(predicted_example, type, begin_tag, end_tag, sticked=True, tagged=False)
                verification_sentence = self_verif_templates[type].format(word=pred['text'], sentence=prompting_sentence)
                sentences.append(verification_sentence)
                addresses.append((i,id))
        prompts = get_prompts_for_model(model_name, sentences)
        print(f"{len(prompts)} prompts generated for self verification")
        
        sampling_params = SamplingParams(
            # stop=['\n'],
            temperature=0,
            max_tokens=2,
        )
        pre_outputs = llm.generate(prompts, sampling_params)
        verif_outputs = [o.outputs[0].text for o in pre_outputs]
        for i, output in enumerate(verif_outputs):
            if yes_no[1].lower() in output.lower():
                sent_idx, ent_id = addresses[i]
                predictions[sent_idx]['entities'] = [ent for ent in predictions[sent_idx]['entities'] if ent['entity_id']!=ent_id]
        # batch_size = 4
        # for i in tqdm(range(0,len(prompts),batch_size)):
        #     prompts_batch = prompts[i:i+batch_size]
        #     input_ids = tokenizer(prompts_batch, padding=True, return_tensors="pt").input_ids
        #     input_ids = input_ids.to(device)
        #     output_batch = model.generate(input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
        #     for j in range(len(prompts_batch)):
        #         scores = output_batch.scores[0][j][[yes_tok, no_tok]]
        #         sent_idx, ent_id = addresses[i+j]
        #         if scores[0]<scores[1]:
        #             predictions[sent_idx]['entities'] = [ent for ent in predictions[sent_idx]['entities'] if ent['entity_id']!=ent_id]

    return outputs, predictions   
    