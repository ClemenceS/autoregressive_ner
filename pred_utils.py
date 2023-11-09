def full_preds_string(textual_outputs, predicted_dataset, reference_dataset, ner_tags):
    full_preds = ""
    for i, (o, pred, gold) in enumerate(zip(textual_outputs, predicted_dataset, reference_dataset)):
            full_preds += '='*50+'\n'
            full_preds += 'input: '+pred['text']+'\n'
            for j,tag in enumerate(ner_tags):
                full_preds += '-'*50+'\n'
                full_preds += tag+' output: '+textual_outputs[j*len(predicted_dataset)+i]+'\n'
                full_preds += 'final: '+str([p['text'] for p in pred['entities'] if p['label']==tag])+'\n'
                full_preds += 'gold: '+str([g['text'] for g in gold['entities'] if g['label']==tag])+'\n'
    return full_preds

def get_metrics_string(metrics_dict, ner_tags):
    s_metrics = ""
    for metric_name, metric in metrics_dict.items():
        s_metrics+="="*20+metric_name+"="*20+'\n'
        s_metrics+=f'ALL    tp: {metric["tp"]}    precision: {metric["precision"]}    recall: {metric["recall"]}    f1: {metric["f1"]}\n'
        for tag in ner_tags:
            s_metrics+=f'{tag}    tp: {metric[tag+"_tp"]}    precision: {metric[tag+"_precision"]}    recall: {metric[tag+"_recall"]}    f1: {metric[tag+"_f1"]}\n'
    return s_metrics