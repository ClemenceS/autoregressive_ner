#this script takes a brat annotation file and a text file and outputs a specific format for evaluation
#
#the brat annotation file is a .ann file
#the text file is a .txt file
#
#the ann file is a tab delimited file with the following columns:
#1. annotation id
#2. annotation type and span
#3. annotation text
#
#span is in the format: start char end char
#
#the text file is a text file with the text of the document
#
#the output is a json file with the following format:
#{
#	"docid": "docid",
#	"words": ["word1", "word2", "word3", ...],
#	"ner_tags": ["0", "0", "0", "0", "0", "0", "1", "1", "1", "0", ...]
#}
#whe ner_tags is a list of numbers, with each number corresponding to a label type (eg. 0 = O, 1 = PERSON, 2 = LOCATION, etc.)
#the list is the same length as the list of words, and each number corresponds to the word at the same index in the words list
#

import sys
import json
import os
import argparse
import re

#parse arguments
parser = argparse.ArgumentParser(description='Convert brat annotation files to json format for evaluation')
parser.add_argument('-d','--dir', help='directory of brat annotation files', required=True)

args = parser.parse_args()


def sentencize(example):
    #this function takes an example and splits it into a list of examples, each with a single sentence
    #the example is a dictionary with the following format:
    #{
    #	"docid": "docid",
    #	"words": ["word1", "word2", "word3", ...],
    #	"ner_tags": ["0", "0", "0", "0", "0", "0", "1", "1", "1", "0", ...]
    #}
    #the ner_tags are a list of numbers, with each number corresponding to a label type (eg. 0 = O, 1 = PERSON, 2 = LOCATION, etc.)
    #the list is the same length as the list of words, and each number corresponds to the word at the same index in the words list
    
    #we start by finding the indices of the words that are sentence boundaries
    sentence_separator = '.'
    indices = [i for i, x in enumerate(example['words']) if x == sentence_separator]
    #we then split the words and ner_tags into a list of lists, with each list corresponding to a sentence
    words = [example['words'][i+1:(j+1 if j else None)] for i, j in zip([-1]+indices, indices+[None])]
    ner_tags = [example['ner_tags'][i+1:(j+1 if j else None)] for i, j in zip([-1]+indices, indices+[None])]
    #we then create a list of examples, each with a single sentence
    examples = []
    for i in range(len(words)):
        if len(words[i]) > 0:
            examples.append({'docid': example['docid']+'_'+str(len(examples)), 'words': words[i], 'ner_tags': ner_tags[i]})
    return examples

label2id = {'O': 0}
#loop through all directories in directory
for dir in os.listdir(args.dir):
    #verify that dir is a directory
    if os.path.isdir(os.path.join(args.dir, dir)):
        all_outputs = []
        #loop through all .ann files in directory
        for filename in os.listdir(os.path.join(args.dir, dir)):
            if filename.endswith('.ann'):
                #get file names
                ann_filename = os.path.join(args.dir, dir, filename)
                txt_filename = ann_filename.replace('.ann', '.txt')
                
                #open files
                ann_file = open(ann_filename, 'r')
                txt_file = open(txt_filename, 'r')

                #read in text file
                text = txt_file.read()

                #read in annotation file
                anns = ann_file.readlines()

                #initialize output
                output = {}
                output['docid'] = filename.replace('.ann', '')
                output['words'] = []
                output['ner_tags'] = []
                #split text into words using the french conventions
                WORD_RE = re.compile(r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]')
                words = re.findall(WORD_RE, text)
                #for each word, get the start and end char in text
                word_idx_to_char_span = {}
                char_idx = 0
                for i, w in enumerate(words):
                    #make sure char_idx is at the start of the word
                    while char_idx < len(text) and text[char_idx] != w[0]:
                        char_idx += 1
                    word_idx_to_char_span[i] = (char_idx, char_idx+len(w))
                    char_idx += len(w)
                    #make sure char_idx is now at the start of the next word
                    while char_idx < len(text) and i+1 < len(words) and text[char_idx] != words[i+1][0]:
                        char_idx += 1
                
                #assert that the char spans match the words
                for i, w in enumerate(words):
                    assert text[word_idx_to_char_span[i][0]:word_idx_to_char_span[i][1]] == w, (w, word_idx_to_char_span[i], text[word_idx_to_char_span[i][0]:word_idx_to_char_span[i][1]])

                span2label = {}
                for ann in anns:
                    if ann.startswith('T'):
                        ann = ann.strip()
                        ann = ann.split('\t')
                        ann_id = ann[0]
                        ann_type = ann[1].split()[0]
                        ann_span = ann[1].split()[1:]
                        ann_span = ann_span[:1]+ann_span[-1:]
                        ann_span = [int(x) for x in ann_span]
                        ann_text = ann[2]
                        span2label[(ann_span[0], ann_span[1])] = ann_type
                

                #sort span2label by start char then by descending end char
                span2label = sorted(span2label.items(), key=lambda x: (x[0][0], -x[0][1]))
                
                

                #remove spans that are subsets of other spans
                i = 0
                while i < len(span2label)-1:
                    if span2label[i][0][0] >= span2label[i+1][0][0] and span2label[i+1][0][1] <= span2label[i][0][1]:
                        span2label.pop(i)
                    else:
                        i += 1

                
                # if len(span2label) > 0:               
                #     if "Analyse minéralogique" in text:
                #         print(words)
                #         print(word_idx_to_char_span)
                #         print(span2label)
                #     next_span = span2label.pop(0)
                #     for i, w in enumerate(words):
                #         output['words'].append(w)
                #         if word_idx_to_char_span[i][0] >= next_span[0][0] and word_idx_to_char_span[i][1] <= next_span[0][1]:
                #             #word is in span
                #             #if type is not in label2id, add it
                #             if next_span[1] not in label2id:
                #                 label2id[next_span[1]] = len(label2id)
                #             output['ner_tags'].append(label2id[next_span[1]])
                #             #make sure next_span is not empty yet and that next word is not in actual next_span
                #             # if len(span2label) > 0 :
                #             #     if word_idx_to_char_span[i][1] >= next_span[0][1]:
                #             #         next_span = span2label.pop(0)
                #             # else:
                                
                #         else:
                #             #word is not in span
                #             output['ner_tags'].append(0)

                # else:
                #     for i, w in enumerate(words):
                #         output['words'].append(w)
                #         output['ner_tags'].append(0)

                for i, w in enumerate(words):
                    output['words'].append(w)
                    for span in span2label:
                        if span[0][0] <= word_idx_to_char_span[i][0] and word_idx_to_char_span[i][1] <= span[0][1]:
                            #word is in span
                            #if type is not in label2id, add it
                            if span[1] not in label2id:
                                label2id[span[1]] = len(label2id)
                            output['ner_tags'].append(label2id[span[1]])
                            break
                    else:
                        #word is not in span
                        output['ner_tags'].append(0)
                
                #write output to file, making sure encoding allows for french characters
                #json.dump(output, out_file, ensure_ascii=False)
                if len(output['docid'].split('_')[0]) <4:
                    all_outputs.extend(sentencize(output))
                else:
                    all_outputs.append(output)
                
                #close files
                ann_file.close()
                txt_file.close()


        #print the first output in a nice format
        print("========")
        print("Example output:")
        for i, w in enumerate(all_outputs):
            if len(w['docid'].split('_')[0]) <4:
                break
        print(json.dumps(all_outputs[0], indent=4, ensure_ascii=False))
        # print(json.dumps(all_outputs[0], indent=4, ensure_ascii=False))
        #write all_outputs to Parquet file inside directory
        import pandas as pd
        df = pd.DataFrame(all_outputs)
        print("========")
        splitname = "train" if dir.startswith("train") else "test"
        output_filename = os.path.join("quaero_test/data", splitname+'.parquet')
        df.to_parquet(output_filename)
        for k,v in label2id.items():
            print("{}: \"{}\"".format(v, k))

        
        #with open(os.path.join(args.dir, dir+'.json'), 'w', encoding='utf-8') as out_file:
        #    json.dump(all_outputs, out_file, ensure_ascii=False)

        



