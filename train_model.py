from csv import reader
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments, LineByLineTextDataset, DataCollator, RobertaForMaskedLM, pipeline
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve
import numpy as np


tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512)
data_fold_nr = '4'


train_sequences = []
train_labels = []
eval_sequences = []
eval_labels = []
test_sequences = []
test_labels = []

# open file in read mode
with open('./data/variable_length_proberta/f'+data_fold_nr+'/trn_prepared_shuffled.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        if ('0' in row[1]):
            train_labels.append(0)
        else:
            train_labels.append(1)
        train_sequences.append(row[0])

with open('./data/variable_length_proberta/f'+data_fold_nr+'/val_prepared_shuffled.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        if ('0' in row[1]):
            eval_labels.append(0)
        else:
            eval_labels.append(1)
        eval_sequences.append(row[0])

with open('./data/variable_length_proberta/f'+data_fold_nr+'/val_prepared_shuffled.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        if ('0' in row[1]):
            test_labels.append(0)
        else:
            test_labels.append(1)
        test_sequences.append(row[0])

config = RobertaConfig(
    vocab_size=10_000,
    max_position_embeddings=514,
    num_attention_heads=8,
    num_hidden_layers=6,
    type_vocab_size=1
)

train_encodings = tokenizer(train_sequences, truncation=True, padding=True)
eval_encodings = tokenizer(eval_sequences, truncation=True, padding=True)
test_encodings = tokenizer(test_sequences, truncation=True, padding=True)


class AmDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = AmDataset(train_encodings, train_labels)
eval_dataset = AmDataset(eval_encodings, eval_labels)
test_dataset = AmDataset(test_encodings, test_labels)


model = RobertaForSequenceClassification(config=config)


training_args = TrainingArguments(
    output_dir="./Am-RoBERTa/m_f"+data_fold_nr,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_strategy='epoch',
    #logging_strategy="epoch",
    logging_steps=100,
    logging_first_step=True,
    evaluation_strategy="steps",
    eval_steps=100
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr[1]
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./Am-RoBERTa/m_f"+data_fold_nr)

model = RobertaForSequenceClassification.from_pretrained(
    './Am-RoBERTa/m_f'+data_fold_nr)
print(model.config)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print(trainer.evaluate(eval_dataset))
