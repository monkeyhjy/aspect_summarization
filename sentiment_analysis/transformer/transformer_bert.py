import torch
import random
import numpy as np
from sklearn.model_selection import KFold
import torch.nn as nn
from transformers import BertTokenizer, BertModel, IntervalStrategy
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import os


# model_path = "bert-base-uncased"
# model_name = "bert-base-uncased"
model_path = "F:/bert_model/test-mlm"
model_name = "test-mlm"

use_roberta = False
# model_path = "roberta-base"
# model_name = "roberta-base"

maxlen = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 3
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
SEED = int(random.random()*1000)
SEED = 93
BATCH_SIZE = 8
N_EPOCHS = 10
LEARNING_RATE = 0.0001


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class MultilabelTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        softmax = nn.Softmax(dim=1)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.MSELoss()
        loss = loss_fct(softmax(logits).float(),
                        labels)
        return (loss, outputs) if return_outputs else loss


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    # tokens = tokens[ :max_input_length-2]
    return tokens


def loadfile():
    f = open('test-all.txt', 'r', encoding='utf-8')

    combined = []
    y = []
    for line in f:
        combined.append(line.split('\t')[0])
        temp = line.split('\t')[2].strip()
        # y.append(temp)
        if temp == '1':
            y.append([1.0, 0.0, 0.0])
        elif temp == '0':
            y.append([0.0, 1.0, 0.0])
        else:
            y.append([0.0, 0.0, 1.0])

    return combined, y


class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BERTGRUSentiment, self).__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                          batch_first=True, dropout= dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else  hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):

        # text = [batch_size, sent_len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch_size, sent_len, emb_dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n_layers * n_directions, batch_size, emb_dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch_size, hidden_dim]

        output = self.out(hidden)
        output = self.softmax(output)



        # output = [batch_size, output_dim]

        return output


def binary_accuracy(preds, y):


    correct = 0
    for i in range(0,y.size()[0]):
        rounded_preds = np.argmax(preds[i].data.cpu().numpy())
        label = np.argmax(y[i].data.cpu().numpy())
        if rounded_preds==label:
            correct+=1

    acc = correct/ len(y)
    return acc


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class WholeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics_old(pred):
    labels = pred.label_ids.argmax(-1)
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def evaluate():
    total_acc = 0.0
    origin,y=loadfile() #[,  , ]

    y = np.array(y, dtype=np.float32)
    KF = KFold(n_splits=10, random_state=SEED, shuffle=True)

    for train_index, test_index in KF.split(origin):
        x_train = []
        x_test = []
        for i in train_index:
            x_train.append(origin[i])
        for i in test_index:
            x_test.append(origin[i])

        y_train, y_test = y[train_index], y[test_index]
        train_encodings = tokenizer(x_train, truncation=True, padding=True)
        test_encodings = tokenizer(x_test, truncation=True, padding=True)
        train_dataset = TrainDataset(train_encodings, y_train)
        test_dataset = TrainDataset(test_encodings, y_test)

        if not use_roberta:
            bert = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3).cuda()
        else:
            bert = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3).cuda()


        # 初始化
        # init_model(model, optimizer, criterion)
        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            learning_rate=1e-5,
            num_train_epochs=20,  # total number of training epochs
            per_device_train_batch_size=20,  # batch size per device during training
            per_device_eval_batch_size=200,  # batch size for evaluation
            logging_dir='./logs',  # directory for storing logs
            logging_steps=500,
            do_train=True,
            do_eval=True,
            no_cuda=False,
            load_best_model_at_end=True,
            # eval_steps=100,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_strategy=IntervalStrategy.STEPS
        )

        trainer = MultilabelTrainer(
            model=bert, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics_old
        )
        train_out = trainer.train()
        test_out = trainer.evaluate()
        total_acc += test_out['eval_accuracy']

    file = open("../../output/eval-SA.txt", "w", encoding="utf-8")
    print(f"Total Test Acc: {total_acc * 10:.2f}%")
    # file.write("seed:"+str(SEED))
    file.write(str(total_acc))
    file.close()


SEED = 93
if not use_roberta:
    tokenizer = BertTokenizer.from_pretrained(model_path)
else:
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

# evaluate()
# classify()

