from csv import reader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch
import numpy as np
import tensorflow as tf


def run(data_fold_nr, data_name, eval_data_name):
    tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512)

    bonus_target_name='NLRside'
    seq_cutoff=39

    test_sequences = []
    test_labels = []

    #with open('./data/'+data_name+'/'+bonus_target_name+'/tst_prepared_shuffled.csv', 'r') as read_obj:
    with open('./data/'+data_name+'/'+bonus_target_name+'/nlrside_tst_prepared.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if ('0' in row[1]):
                test_labels.append(0)
            else:
                test_labels.append(1)
            test_sequences.append(row[0])



    model = RobertaForSequenceClassification.from_pretrained(
        "./Am-RoBERTa/"+eval_data_name+"_m_"+data_fold_nr+"(final)")
    

    current_label = 0
    for seq in test_sequences:
        print('Processing sequence '+str(test_sequences.index(seq)+1)+'/'+str(len(test_sequences)))
        predictions = []
        positive_sequences = []
        pos_probs = []
        neg_probs = []
        if len(seq)>seq_cutoff:
            splits = len(seq)-seq_cutoff
            for i in range(splits):
                subseq = seq[i:seq_cutoff+i+1]
                inputs = tokenizer(subseq, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.softmax(logits, axis=-1).detach().numpy()
                if prediction[0][1]>prediction[0][0]:
                    positive_sequences.append(subseq)
                    predictions.append('1')
                    pos_probs.append(prediction[0][1])
                else:
                    predictions.append('0')
                    neg_probs.append(prediction[0][0])

        if '1' in predictions:
            with open("./data/window/current_test.txt", 'a') as out_f:
                out_f.write(eval_data_name + data_fold_nr+'-'+ bonus_target_name + '\n1,'+str(test_labels[current_label])+','+str(max(pos_probs))+','+str(positive_sequences)+'\n')
        else:
            with open("./data/window/current_test.txt", 'a') as out_f:
                out_f.write(eval_data_name + data_fold_nr+'-'+ bonus_target_name + '\n0,'+str(test_labels[current_label])+','+str(min(neg_probs))+'\n')            
        
        current_label += 1





run(str(1), 'NLReff_for_PRoBERTa', 'PRoBERTa')