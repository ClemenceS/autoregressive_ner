import re
import torch
from tqdm import tqdm
from fastchat.model import get_conversation_template
from sklearn.model_selection import KFold
from prompt_maker import make_prompts, get_yes_no_words
from transformers import StoppingCriteria

def get_prompt_for_model(model_name, prompt):
    if 'vicuna' in model_name:
        conv = get_conversation_template(model_name)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif 'bloom' in model_name:    
        pass
    else:
        raise NotImplementedError("Model not supported")
    return prompt

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


def bloom_predict(training_data, testing_data, ner_tags, model_name, logger, model, tokenizer, control, self_verification, begin_tag, end_tag, model_kwargs, **kwargs):
    first_prompts = []
    self_verif_templates = []
    if testing_data is None:
        for ner_tag in ner_tags:
            kf = KFold(n_splits=len(training_data), shuffle=False)
            logger.info(f"Making a {len(training_data)}-fold cross validation over the training data.")
            for i, (train_indices, dev_indices) in enumerate(kf.split(training_data)):
                dev_dataset = [training_data[i] for i in dev_indices]
                train_dataset = [training_data[i] for i in train_indices]
                first_prompts_fold, self_verif_templates_fold = make_prompts(
                    train_dataset,
                    dev_dataset,
                    ner_tag,
                    begin_tag=begin_tag,
                    end_tag=end_tag,
                    self_verification=self_verification,
                    **kwargs
                )
                first_prompts.extend(first_prompts_fold)
                self_verif_templates.extend(self_verif_templates_fold)
    else:
        logger.info("{} examples in train set".format(len(training_data)))
        logger.info("{} examples in test set".format(len(testing_data)))
        for ner_tag in ner_tags:
            first_prompts_ner_tag, self_verif_templates_ner_tag= make_prompts(
                training_data,
                testing_data,
                ner_tag,
                begin_tag=begin_tag,
                end_tag=end_tag,
                self_verification=self_verification,
                **kwargs
            )
            first_prompts.extend(first_prompts_ner_tag)
            self_verif_templates.extend(self_verif_templates_ner_tag)
    logger.info("Number of prompts : {}".format(len(first_prompts)))
    logger.info("Here is an example prompt :\n{}".format(first_prompts[0]))

    logger.info("Getting last line lengths...")
    #line_lengths = [len(tokenizer.encode(prompt.split('\n')[-2])) for prompt in first_prompts]
    #line_lengths = [len(tokenizer.encode(text)) for text in [example['text'] for example in testing_data]] if testing_data is not None else [len(tokenizer.encode(text)) for text in [example['text'] for example in training_data]]
    newline_token = tokenizer.encode('\n', add_special_tokens=False)[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Generating...")
    outputs = []
    reference = testing_data if testing_data is not None else training_data
    predictions = [
                    {
                        'doc_id': example['doc_id'],
                        'text': example['text'],
                        'entities': [],
                    }
                    for example in reference
                ]
    for i in tqdm(range(len(first_prompts))):
        prompt = get_prompt_for_model(model_name, first_prompts[i])
        input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        if not control:
            criteria = Newline(check_start=len(input_ids[i]), newline_token=newline_token)
            output = model.generate(input_ids, max_new_tokens=512, stopping_criteria=[criteria], output_scores=True, return_dict_in_generate=True, **model_kwargs)
            output = output[:,len(input_ids[0]):]
            output = tokenizer.decode(output[0], skip_special_tokens=True, skip_spaces_between_tokens=False)                
        else:
            entry = tokenizer.encode(first_prompts[i].split('\n')[-2].split(':',2)[1].strip()+'\n', add_special_tokens=False)
            sticked = True
            
            begin_tag_toks = tokenizer.encode("@@",add_special_tokens=False)
            if sticked:
                end_tag_toks = tokenizer.encode('@##',add_special_tokens=False)[1:]
            else :
                end_tag_toks = tokenizer.encode("##",add_special_tokens=False)
            nb_open_entites = 0
            while input_ids[0,-1]!=tokenizer.eos_token_id and len(entry)>1:
                output = model.generate(input_ids,max_new_tokens=1,output_scores=True, return_dict_in_generate=True, **model_kwargs)
                scores = output.scores
                sequences = output.sequences
                next_entry_id = entry[0]
                allowed_tokens = [next_entry_id,begin_tag_toks[0]]
                if nb_open_entites>0:
                    allowed_tokens.append(end_tag_toks[0])
                next_scores = {k:v for k,v in zip(allowed_tokens, scores[0][0][[allowed_tokens]])}
                generated_id = max(next_scores, key=next_scores.get)
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

                input_ids = torch.cat([input_ids,torch.tensor(all_generated_ids).unsqueeze(0).to(device)], dim=1)

            output = tokenizer.decode(sequences[0],skip_special_tokens=True).replace(prompt,'').strip()
        outputs.append(output)
        
        predictions[i%len(reference)]['entities'].extend([
            {
                'entity_id': 'T{}'.format(ent_idx),
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
            yes_no = get_yes_no_words(keywords=kwargs['keywords'])
            yes_tok = tokenizer.encode(yes_no[0],add_special_tokens=False)[0]
            no_tok = tokenizer.encode(yes_no[1],add_special_tokens=False)[0]
            allowed_tokens = [yes_tok, no_tok]
            verified_entities = []
            for pred in predictions[i%len(reference)]['entities']:
                if pred['label']!=ner_tags[i//len(reference)]:
                    verified_entities.append(pred)
                    continue
                verification_prompt = self_verif_templates[i].format(word=pred['text'])
                # print(verification_prompt)
                verification_prompt = get_prompt_for_model(model_name, verification_prompt)
                input_ids = tokenizer(verification_prompt, padding=True, return_tensors="pt").input_ids
                input_ids = input_ids.to(device)
                answer = model.generate(input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                scores = answer.scores[0][0][allowed_tokens]
                # print('---')
                # print(tokenizer.decode(answer.sequences[-1], skip_special_tokens=True))
                # print('====')
                if scores[0]>scores[1]:
                    verified_entities.append(pred)
            predictions[i%len(reference)]['entities'] = verified_entities
        
    
    return outputs, predictions   
    