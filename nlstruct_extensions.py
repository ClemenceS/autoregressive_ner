import torch
from torchmetrics import Metric

import os
from nlstruct.datasets.base import NERDataset
from datasets import load_dataset, load_from_disk

from nlstruct.data_utils import regex_tokenize, split_spans, dedup
from nlstruct.torch_utils import pad_to_tensor
from collections import defaultdict

def entity_match_filter(labels, matcher):
    labels = labels if isinstance(labels, (tuple, list)) else (labels,)
    if isinstance(matcher, (tuple, list)):
        return any(label in matcher for label in labels)
    eval_locals = defaultdict(lambda: False)
    eval_locals.update({k: True for k in labels})
    return eval(matcher, None, eval_locals)


class DocumentEntityMetricPerLabel(Metric):
    def __init__(
          self,
          add_label_specific_metrics=[],
          word_regex=r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]',
          filter_entities=None,
          joint_matching=False,
          binarize_label_threshold=1.,
          binarize_tag_threshold=0.5,
          eval_attributes=False,
          eval_fragments_label=False,
          compute_on_step=True,
          dist_sync_on_step=False,
          process_group=None,
          dist_sync_fn=None,
          explode_fragments=False,
          prefix="",
    ):
        # `compute_on_step` was removed from torchmetrics v0.9
        # keep the argument in signature for compatibility
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self._compute_on_step = compute_on_step
        self.joint_matching = joint_matching
        self.filter_entities = filter_entities
        self.prefix = prefix
        self.eval_attributes = eval_attributes
        self.eval_fragments_label = eval_fragments_label
        self.explode_fragments = explode_fragments
        self.word_regex = word_regex
        self.add_label_specific_metrics = add_label_specific_metrics
        self.binarize_label_threshold = float(binarize_label_threshold) if binarize_label_threshold is not False else binarize_label_threshold
        self.binarize_tag_threshold = float(binarize_tag_threshold) if binarize_tag_threshold is not False else binarize_tag_threshold
        self.add_state("true_positive", default=torch.tensor(0., device="cuda"), dist_reduce_fx="sum")
        self.add_state("pred_count", default=torch.tensor(0., device="cuda"), dist_reduce_fx="sum")
        self.add_state("gold_count", default=torch.tensor(0., device="cuda"), dist_reduce_fx="sum")
        for label in self.add_label_specific_metrics:
           self.add_state(f"{label}_true_positive", default=torch.tensor(0., device="cuda"), dist_reduce_fx="sum")
           self.add_state(f"{label}_pred_count", default=torch.tensor(0., device="cuda"), dist_reduce_fx="sum")
           self.add_state(f"{label}_gold_count", default=torch.tensor(0., device="cuda"), dist_reduce_fx="sum")

    def increment(self, name, by=1):
        """Increments a counter specified by the 'name' argument."""
        self.__dict__[name] += by
   
    def update(self, preds, targets):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        for pred_doc, gold_doc in zip(preds, targets):
            for label,(tp, pc, gc) in self.compare_two_samples(pred_doc, gold_doc).items():
                self.increment(f"true_positive", by=tp)
                self.increment(f"pred_count", by=pc)
                self.increment(f"gold_count", by=gc)
                if label in self.add_label_specific_metrics:
                    self.increment(f"{label}_true_positive", by=tp)
                    self.increment(f"{label}_pred_count", by=pc)
                    self.increment(f"{label}_gold_count", by=gc)

    def compare_two_samples(self, pred_doc, gold_doc, return_match_scores=False):
        assert pred_doc["text"] == gold_doc["text"], f'Mismatch:\n{pred_doc["text"]}\nvs.\n{gold_doc["text"]}'
        pred_doc_entities = list(pred_doc["entities"])
        gold_doc_entities = list(gold_doc["entities"])

        pred_doc_entities = [entity for entity in pred_doc_entities
                             if self.filter_entities is None
                             or entity_match_filter(entity["label"], self.filter_entities)]
        gold_doc_entities = [entity for entity in gold_doc_entities
                             if self.filter_entities is None
                             or entity_match_filter(entity["label"], self.filter_entities)]
        if self.explode_fragments:
            pred_doc_entities = [{"label": f.get("label", "main"), "fragments": [f]} for f in
                                 dedup((f for entity in pred_doc_entities for f in entity["fragments"]), key=lambda x: (x['begin'], x['end'], x.get('label', None)))]
            gold_doc_entities = [{"label": f.get("label", "main"), "fragments": [f]} for f in
                                 dedup((f for entity in gold_doc_entities for f in entity["fragments"]), key=lambda x: (x['begin'], x['end'], x.get('label', None)))]

        words = regex_tokenize(gold_doc["text"], reg=self.word_regex, do_unidecode=True, return_offsets_mapping=True)

        all_fragment_labels = set()
        all_entity_labels = set()
        fragments_begin = []
        fragments_end = []
        pred_entities_fragments = []
        for entity in pred_doc_entities:
            all_entity_labels.update(set(entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],)))
            #                                          *(("{}:{}".format(att["name"], att["label"]) for att in entity["attributes"]) if self.eval_attributes else ()))))
            pred_entities_fragments.append([])
            for fragment in entity["fragments"]:
                pred_entities_fragments[-1].append(len(fragments_begin))
                fragments_begin.append(fragment["begin"])
                fragments_end.append(fragment["end"])
                all_fragment_labels.add(fragment["label"] if self.eval_fragments_label else "main")

        gold_entities_fragments = []
        for entity in gold_doc_entities:
            all_entity_labels.update(set(entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],)))
            #                                          *(("{}:{}".format(att["name"], att["value"]) for att in entity["attributes"]) if self.eval_attributes else ()))))
            gold_entities_fragments.append([])
            for fragment in entity["fragments"]:
                gold_entities_fragments[-1].append(len(fragments_begin))
                fragments_begin.append(fragment["begin"])
                fragments_end.append(fragment["end"])
                all_fragment_labels.add(fragment["label"] if self.eval_fragments_label else "main")
        all_fragment_labels = list(all_fragment_labels)
        all_entity_labels = list(all_entity_labels)
        if len(all_fragment_labels) == 0:
            all_fragment_labels = ["main"]
        if len(all_entity_labels) == 0:
            all_entity_labels = ["main"]

        fragments_begin, fragments_end = split_spans(fragments_begin, fragments_end, words["begin"], words["end"])
        pred_entities_labels = [[False] * len(all_entity_labels)] * max(len(pred_doc_entities), 1)
        pred_tags = [[[False] * len(words["begin"]) for _ in range(len(all_fragment_labels))] for _ in range(max(len(pred_doc_entities), 1))]  # n_entities * n_token_labels * n_tokens
        gold_entities_labels = [[False] * len(all_entity_labels)] * max(len(gold_doc_entities), 1)
        gold_entities_optional_labels = [[False] * len(all_entity_labels)] * max(len(gold_doc_entities), 1)
        gold_tags = [[[False] * len(words["begin"]) for _ in range(len(all_fragment_labels))] for _ in range(max(len(gold_doc_entities), 1))]  # n_entities * n_token_labels * n_tokens

        for entity_idx, (entity_fragments, entity) in enumerate(zip(pred_entities_fragments, pred_doc_entities)):
            for fragment_idx, fragment in zip(entity_fragments, entity["fragments"]):
                begin = fragments_begin[fragment_idx]
                end = fragments_end[fragment_idx]
                label = all_fragment_labels.index(fragment["label"] if self.eval_fragments_label else "main")
                pred_tags[entity_idx][label][begin:end] = [True] * (end - begin)
            entity_labels = list(entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],))
            # *(("{}:{}".format(att["name"], att["label"]) for att in entity["attributes"]) if self.eval_attributes else ())]
            pred_entities_labels[entity_idx] = [label in entity_labels for label in all_entity_labels]

        for entity_idx, (entity_fragments, entity) in enumerate(zip(gold_entities_fragments, gold_doc_entities)):
            for fragment_idx, fragment in zip(entity_fragments, entity["fragments"]):
                begin = fragments_begin[fragment_idx]
                end = fragments_end[fragment_idx]
                label = all_fragment_labels.index(fragment["label"] if self.eval_fragments_label else "main")
                gold_tags[entity_idx][label][begin:end] = [True] * (end - begin)
            entity_labels = list(entity["label"] if isinstance(entity["label"], (tuple, list)) else (entity["label"],))
            entity_optional_labels = entity.get("complete_labels", entity["label"])
            if not isinstance(entity_optional_labels, (tuple, list)):
                entity_optional_labels = [entity_optional_labels]
            # *(("{}:{}".format(att["name"], att["value"]) for att in entity["attributes"]) if self.eval_attributes else ())]
            gold_entities_labels[entity_idx] = [label in entity_labels for label in all_entity_labels]
            gold_entities_optional_labels[entity_idx] = [label in entity_optional_labels for label in all_entity_labels]

        gold_tags = pad_to_tensor(gold_tags)
        pred_tags = pad_to_tensor(pred_tags)
        gold_entities_labels = pad_to_tensor(gold_entities_labels)
        gold_entities_optional_labels = pad_to_tensor(gold_entities_optional_labels)
        pred_entities_labels = pad_to_tensor(pred_entities_labels)

        # score = 0.
        tag_denom_match_scores = (
              pred_tags.float().sum(-1).sum(-1).unsqueeze(1) +
              gold_tags.float().sum(-1).sum(-1).unsqueeze(0)
        )
        tag_match_scores = 2 * torch.einsum("pkt,gkt->pg", pred_tags.float(), gold_tags.float()) / tag_denom_match_scores.clamp_min(1)
        #tag_match_scores[(tag_denom_match_scores == 0.) & (tag_match_scores == 0.)] = 1.


        # tag_denom_match_scores = (
        #      pred_tags.float().sum(-1).unsqueeze(1) + # pkt -> p:k
        #      gold_tags.float().sum(-1).unsqueeze(0)   # gkt -> :gk
        # )
        # tag_match_scores = (2 * torch.einsum("pkt,gkt->pgk", pred_tags.float(), gold_tags.float()) / tag_denom_match_scores.clamp_min(1)) > 0.5
        # tag_match_scores[tag_denom_match_scores == 0.] = 1.

        # label_match_scores = 2 * torch.einsum("pk,gk->pg", pred_entities_labels.float(), gold_entities_labels.float()) / (
        #      pred_entities_labels.float().sum(-1).unsqueeze(1) +
        #      gold_entities_labels.float().sum(-1).unsqueeze(0)
        # ).clamp_min(1)

        label_match_precision = torch.einsum("pk,gk->pg", pred_entities_labels.float(), gold_entities_optional_labels.float()) / pred_entities_labels.float().sum(-1).unsqueeze(1).clamp_min(1.)
        label_match_recall = torch.einsum("pk,gk->pg", pred_entities_labels.float(), gold_entities_labels.float()) / gold_entities_labels.float().sum(-1).unsqueeze(0).clamp_min(1.)
        label_match_scores = 2 / (1. / label_match_precision + 1. / label_match_recall)
        match_scores = label_match_scores * tag_match_scores
        if self.binarize_tag_threshold is not False:
            tag_match_scores = (tag_match_scores >= self.binarize_tag_threshold).float()
        if self.binarize_label_threshold is not False:
            label_match_scores = (label_match_scores >= self.binarize_tag_threshold).float()
        effective_scores = tag_match_scores * label_match_scores
        
        results={}
        
        pred_values = [p['label'] for p in pred_doc_entities]
        gold_values = [g['label'] for g in gold_doc_entities]
        
        score_per_label = {l:0. for l in all_entity_labels}
        matched_scores = torch.zeros_like(match_scores) - 1.
        for pred_idx in range(match_scores.shape[0]):
            if self.joint_matching:
                pred_idx = match_scores.max(-1).values.argmax()
            gold_idx = match_scores[pred_idx].argmax()
            if not any(gold_entities_labels[gold_idx]):
                continue
            ent_label = all_entity_labels[gold_entities_labels[gold_idx].nonzero().squeeze()]
            match_score = match_scores[pred_idx, gold_idx].float()
            effective_score = effective_scores[pred_idx, gold_idx].float()
            matched_scores[pred_idx, gold_idx] = max(matched_scores[pred_idx, gold_idx], effective_score)
            if match_score >= 0 and effective_score > 0:
                score_per_label[ent_label] += effective_score
                match_scores[:, gold_idx] = -1
                match_scores[pred_idx, :] = -1
        on_cuda = True
        # return {l : (float(score_per_label[l]), pred_values.count(l), gold_values.count(l))
        #             for l in all_entity_labels}
        if on_cuda :
            #make sure to return a tuple of tensors
            return {l : (torch.tensor(score_per_label[l]).cuda(), torch.tensor(pred_values.count(l)).cuda(), torch.tensor(gold_values.count(l)).cuda())
                        for l in all_entity_labels}
        else :
            return {l : (torch.tensor(score_per_label[l]), torch.tensor(pred_values.count(l)), torch.tensor(gold_values.count(l)))
                        for l in all_entity_labels}

    def compute(self):
        """
        Computes accuracy over state.
        """
        results={}
        if self.gold_count == 0 and self.pred_count == 0:
           results[self.prefix + f"tp"]= 0
           results[self.prefix + f"precision"]= 1
           results[self.prefix + f"_recall"]= 1
           results[self.prefix + f"f1"]= 1
        else :
           results[self.prefix + "tp"] = self.true_positive
           results[self.prefix + "precision"]= self.true_positive / max(1, self.pred_count)
           results[self.prefix + "recall"]= self.true_positive/ max(1, self.gold_count)
           results[self.prefix + "f1"]= (self.true_positive * 2) / (self.pred_count + self.gold_count)
        for label in self.add_label_specific_metrics:
            l_true_positive, l_gold_count, l_pred_count = getattr(self, f"{label}_true_positive"), getattr(self, f"{label}_gold_count"), getattr(self, f"{label}_pred_count")
            if l_gold_count == 0 and l_pred_count == 0:
               results[self.prefix + f"{label}_tp"]= 0
               results[self.prefix + f"{label}_precision"]= 1
               results[self.prefix + f"{label}_recall"]= 1
               results[self.prefix + f"{label}_f1"]= 1
            else :
               results[self.prefix + f"{label}_tp"] = l_true_positive
               results[self.prefix + f"{label}_precision"]= l_true_positive / max(1, l_pred_count)
               results[self.prefix + f"{label}_recall"]= l_true_positive/ max(1, l_gold_count)
               results[self.prefix + f"{label}_f1"]= (l_true_positive * 2) / (l_pred_count + l_gold_count)
        return results



def tags_to_entities(words, ner_tags, tag_map):
    #this function takes a list of words and a list of tags and returns a list of entities
    #the tags are 0 for O, and 1 for type '1' and 2 for type '2' and so on
    #each entity is a dictionary with the following keys: 'id', 'type', 'begin', 'end', 'text'
    ann = []
    i=0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag != 0:
            j = i+1
            while j < len(ner_tags) and tag_map[ner_tags[j]] == tag_map[tag]:
                j += 1
            ent_type = tag_map[tag].split('-')[-1]
            ent_id = f"T{len(ann)+1}"
            ent_text = ' '.join(words[i:j])
            ent_begin = len(' '.join(words[:i])) + 1 if i > 0 else 0
            ent_end = ent_begin + len(ent_text)
            ann.append({
                'entity_id': ent_id,
                'label': ent_type,
                'fragments': [{
                    'begin': ent_begin,
                    'end': ent_end
                }],
                'text': ent_text
            })
            i = j
        else:
            i += 1
    return ann

    
def load_from_hf(dataset, tag_map, doc_id_colname, words_colname='words', ner_tags_colname='ner_tags'):
    examples = []
    # Load a brat dataset into a Dataset object
    for e in dataset:
        examples.append({
            'doc_id': e[doc_id_colname],
            'text': ' '.join(e[words_colname]),
            'entities': tags_to_entities(e[words_colname], e[ner_tags_colname], tag_map),
        })
    return examples

class HuggingfaceNERDataset(NERDataset):
    def __init__(self, dataset_name: str, tag_map: dict, preprocess_fn=None, doc_id_colname='doc_id', words_colname='words', ner_tags_colname='ner_tags', load_from_disk=False):
        self.load_from_disk = load_from_disk
        train_data, val_data, test_data = self.extract(dataset_name, tag_map, doc_id_colname=doc_id_colname, words_colname=words_colname, ner_tags_colname=ner_tags_colname)
        super().__init__(train_data, val_data, test_data, preprocess_fn=preprocess_fn)
    
    def extract(self, dataset_name, tag_map, doc_id_colname, words_colname, ner_tags_colname):
        try :
            if self.load_from_disk:
                self.dataset = load_from_disk(dataset_name)
            else:
                try:
                    base = os.path.basename(dataset_name)
                    dir = os.path.dirname(dataset_name)
                    self.dataset = load_dataset(dir, base)
                except:
                    self.dataset = load_dataset(dataset_name)
        except ValueError:
            raise ValueError(f"Dataset {dataset_name} does not exist. Please check the name of the dataset.")
        train_data = load_from_hf(self.dataset["train"], tag_map, doc_id_colname=doc_id_colname, words_colname=words_colname, ner_tags_colname=ner_tags_colname)
        test_data = load_from_hf(self.dataset["test"], tag_map, doc_id_colname=doc_id_colname, words_colname=words_colname, ner_tags_colname=ner_tags_colname)
        val_data = load_from_hf([], tag_map, doc_id_colname=doc_id_colname, words_colname=words_colname, ner_tags_colname=ner_tags_colname)
        return train_data, val_data, test_data
    