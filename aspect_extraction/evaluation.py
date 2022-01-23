#!/usr/bin/env python
#  -*- coding: utf-8  -*-
# the methods to evaluate the effect of aspect_extraction

import codecs
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
# # cluster_map need to be specified manually according to the top words in each inferred aspect (save in aspect.log)

gold_map = {
    'community': 0,
    'compatibility': 1,
    'documentation': 2,
    'functional': 3,
    'performance': 4,
    'reliability': 5,
    'usability': 6,
    'other': 7
}

# ABAE_map = {
#     0: 0,
#     1: 2,
#     2: 1,
#     3: 6,
#     4: 4,
#     5: 5,
#     6: 2,
# }

ABAE_map = {
    0: 6,
    1: 4,
    2: 1,
    3: 0,
    4: 3,
    5: 5,
    6: 4,
}

ABAE_reversed_map = {
    gold_map['community']: '4',
    gold_map['compatibility']: '2',
    gold_map['documentation']: '4',
    gold_map['functional']: '1+7',
    gold_map['performance']: '2',
    gold_map['reliability']: '3',
    gold_map['usability']: '6',
    gold_map['other']: '0+5'
}

tool_names = [
    'Jenkins', 'Travis', 'TeamCity', 'CircleCi', 'GitlabCi'
]

start = {
    'TeamCity': 0,
    'GitlabCi': 347,
    'CircleCi': 693,
    'Jenkins': 1032,
    'Travis': 1414,
}

end = {
    'TeamCity': 347,
    'GitlabCi': 693,
    'CircleCi': 1032,
    'Jenkins': 1414,
    'Travis': 2000,
}

DKAAE_reversed_map = {
    gold_map['community']: '1',
    gold_map['compatibility']: '0',
    gold_map['documentation']: '1',
    gold_map['functional']: '2',
    gold_map['performance']: '3',
    gold_map['reliability']: '4',
    gold_map['usability']: '5',
    gold_map['other']: '6'
}

LDA_map = {

}


def evaluation(true, predict, domain, model, sub_class=False):
    true_label = []
    predict_label = []

    if not sub_class:
        if domain == 'SO':
            for id in true.keys():
                true_id = int(id) - 1
                true_label.append(int(true[id]))
                predict_label.append(int(predict[str(true_id)]))
                if int(true[id]) == 6:
                    print(true_id)
                    print(int(predict[str(true_id)]))

            if model == 'ABAE':
                predict_label_ABAE = predict_label
                predict_label = []
                for label in predict_label_ABAE:
                    predict_label.append(ABAE_map[int(label)])

        print(classification_report(true_label, predict_label,
                                    target_names=['community', 'compatibility', 'documentation', 'functional',
                                                  'performance', 'reliability', 'usability', 'other'], digits=3))
        print(f1_score(true_label, predict_label, labels=[0,1,2,3,4,5,6], average='micro'))
    else:
        if domain == 'SO':

            for id in true.keys():
                true_id = int(id) - 1
                temp = int(true[id])
                if temp == 0 or temp == 2:
                    temp = 6
                true_label.append(temp)
                predict_label.append(int(predict[str(true_id)]))

            if model == 'ABAE':
                predict_label_ABAE = predict_label
                predict_label = []
                for label in predict_label_ABAE:
                    predict_label.append(ABAE_map[int(label)])


        print(classification_report(true_label, predict_label,
                                    target_names=['compatibility', 'functional',
                                                  'performance', 'reliability', 'usability', 'other'], digits=3))


def evaluation_7cata(true, predict, model):
    true_label = []
    predict_label = []
    fout = open("out/wrong_label.txt", 'w', encoding='utf-8')
    for id in true.keys():
        # true_id = int(id) - 1
        true_id = int(id)
        try:
            true_label.append(int(
                DKAAE_reversed_map[
                    int(true[id])
                ]
            ))
        except ValueError:
            print(id)

        predict_label.append(int(predict[str(true_id)]))
        if int(predict[str(true_id)]) != int(DKAAE_reversed_map[int(true[id])]):
            fout.write(str(id+1)+'\t'+true[id]+'\t'+predict[str(true_id)]+'\n')

    if model == 'ABAE':
        predict_label_ABAE = predict_label
        predict_label = []
        for label in predict_label_ABAE:
            predict_label.append(ABAE_map[int(label)])

    report = classification_report(true_label, predict_label,
                                target_names=['portability', 'learnability', 'functional',
                                              'performance', 'reliability', 'usability', 'other'], digits=3)
    print(report)
    p = precision_score(true_label, predict_label, labels=[0,1,2,3,4,5,6], average='weighted')
    r = recall_score(true_label, predict_label, labels=[0,1,2,3,4,5,6], average='weighted')
    return p,r,f1_score(true_label, predict_label, labels=[0,1,2,3,4,5,6], average='weighted')


def prediction(test_labels, aspect_probs, cluster_map, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:, -1]
    predict_labels = [cluster_map[label_id] for label_id in label_ids]
    evaluation(open(test_labels), predict_labels, domain)


def load_test_file_travis(domain):
    f = open('../dataset/'+domain+'/test.txt', 'r', encoding='utf-8')
    labels = {}
    for line in f:
        segs = line.split('\t')
        labels[segs[0]] = segs[2].strip()
    return labels


def load_test_file(domain):
    f = open('../dataset/'+domain+'/test.txt', 'r', encoding='utf-8')
    labels = {}
    i=0
    for line in f:
        segs = line.split('\t')
        labels[i]=segs[1].strip()
        i=i+1
    return labels


def evaluate_tools(true, predict, model, start, end):
    true_label = []
    predict_label = []
    for id in true.keys():
        # true_id = int(id) - 1
        true_id = int(id)
        try:
            true_label.append(int(
                DKAAE_reversed_map[
                    int(true[id])
                ]
            ))
        except ValueError:
            print(id)

        predict_label.append(int(predict[str(true_id)]))

    if model == 'ABAE':
        predict_label_ABAE = predict_label
        predict_label = []
        for label in predict_label_ABAE:
            predict_label.append(ABAE_map[int(label)])

    true_label = true_label[start:end]
    predict_label = predict_label[start:end]
    report = classification_report(true_label, predict_label,
                                   target_names=['portability', 'learnability', 'functional',
                                                 'performance', 'reliability', 'usability', 'other'], digits=3)
    # print(report)

    p = precision_score(true_label, predict_label, labels=[0, 1, 2, 3, 4, 5, 6], average='weighted')
    r = recall_score(true_label, predict_label, labels=[0, 1, 2, 3, 4, 5, 6], average='weighted')
    return p, r, f1_score(true_label, predict_label, labels=[0, 1, 2, 3, 4, 5, 6], average='weighted')


def get_test_label():
    true_label = []
    labels = load_test_file('SO')
    for id in labels.keys():
        # true_id = int(id) - 1
        true_id = int(id)
        try:
            true_label.append(int(
                DKAAE_reversed_map[
                    int(labels[id])
                ]
            ))
        except ValueError:
            print(id)
    return true_label


def evaluation_size(true, predict, model):
    true_label = []
    predict_label = []
    for id in predict.keys():
        # true_id = int(id) - 1
        true_id = int(id)
        try:
            true_label.append(int(
                DKAAE_reversed_map[
                    int(true[true_id])
                ]
            ))
        except ValueError:
            print(id)

        predict_label.append(int(predict[str(true_id)]))

    report = classification_report(true_label, predict_label,
                                target_names=['portability', 'learnability', 'functional',
                                              'performance', 'reliability', 'usability', 'other'], digits=3)
    print(report)
    p = precision_score(true_label, predict_label, labels=[0,1,2,3,4,5,6], average='weighted')
    r = recall_score(true_label, predict_label, labels=[0,1,2,3,4,5,6], average='weighted')
    return p,r,f1_score(true_label, predict_label, labels=[0,1,2,3,4,5,6], average='weighted')


def load_results(domain, model, epoch):
    path = 'out/'+domain+'/'+model+'/sumout/'+str(epoch)+'.sum'
    f = open(path, 'r', encoding='utf-8')
    results = {}
    for line in f:
        segs = line.split('\t')
        results[segs[0]] = segs[1].strip()
    return results


if __name__ == "__main__":
    labels = load_test_file('SO')

    results = load_results('SO', 'ABAE', 10)
    evaluation_7cata(labels, results, 'ABAE')
    for t in tool_names:
        # print('\n' + t)
        evaluate_tools(labels, results, 'ABAE', start[t], end[t])


