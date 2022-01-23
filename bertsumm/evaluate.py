# coding:utf8
from rouge import Rouge

tool_names = [
    'Jenkins'
]

aspect_names = [
    'compatibility', 'documentation', 'functional', 'performance', 'reliability', 'usability', 'other'
]

rouge_methods = [
    "rouge-1",
    "rouge-2",
    "rouge-l"
]


def mean(arr):
    sum = 0
    for i in arr:
        sum += i
    return sum/len(arr)


def read_result(model, tool, aspect):
    positive = []
    negative = []
    ns=""
    ps=""
    curr = True
    path = "data/result/"+model+"/test/"+tool+"."+aspect+".txt"
    f = open(path, "r", encoding="utf-8")
    for line in f:
        line = line.strip()
        if line == "negative:":
            curr = False

        if len(line) != 0:
            if curr:
                ns += line+"\n"
            else:
                ps += line+"\n"
    positive.append(ps.lower())
    negative.append(ns.lower())
    return positive, negative


def read_label(num, tool, aspect):
    positive = []
    negative = []
    ns = ""
    ps = ""
    curr = True
    path = "data/test/" + tool + "/gold" + num + "/" + aspect + ".txt"
    f = open(path, "r", encoding="utf-8")
    for line in f:
        line = line.strip()
        if line == "Positive:":
            curr = False

        if len(line) != 0:
            if curr:
                ns += line + "\n"
            else:
                ps += line + "\n"
    positive.append(ps.lower())
    negative.append(ns.lower())
    return positive, negative


rouge_score1 = {
    "bert": {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
    },
    "bert_ori": {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
    }
}
rouge_score2 = {
    "bert": {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
    },
    "bert_ori": {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
    }
}
rouge = Rouge()
for t in tool_names:
    for a in aspect_names:
        # label1
        label1_positive, label1_negative = read_label("1", t, a)
        # label2
        label2_positive, label2_negative = read_label("2", t, a)

        # bert origin
        bert_ori_positive, bert_ori_negative = read_result("bert_ori", t, a)

        rouge_score1_p = rouge.get_scores(label1_positive, bert_ori_positive)
        rouge_score2_p = rouge.get_scores(label2_positive, bert_ori_positive)
        for r in rouge_methods:
            rouge_score1["bert_ori"][r].append(rouge_score1_p[0][r]['r'])
            rouge_score2["bert_ori"][r].append(rouge_score2_p[0][r]['r'])

        rouge_score1_n = rouge.get_scores(label1_negative, bert_ori_negative)
        rouge_score2_n = rouge.get_scores(label2_negative, bert_ori_negative)
        for r in rouge_methods:
            rouge_score1["bert_ori"][r].append(rouge_score1_n[0][r]['r'])
            rouge_score2["bert_ori"][r].append(rouge_score2_n[0][r]['r'])

        # bert
        bert_positive, bert_negative = read_result("bert", t, a)
        rouge_score1_p = rouge.get_scores(label1_positive, bert_positive)
        rouge_score2_p = rouge.get_scores(label2_positive, bert_positive)
        for r in rouge_methods:
            rouge_score1["bert"][r].append(rouge_score1_p[0][r]['r'])
            rouge_score2["bert"][r].append(rouge_score2_p[0][r]['r'])

        rouge_score1_n = rouge.get_scores(label1_negative, bert_ori_negative)
        rouge_score2_n = rouge.get_scores(label2_negative, bert_ori_negative)
        for r in rouge_methods:
            rouge_score1["bert"][r].append(rouge_score1_n[0][r]['r'])
            rouge_score2["bert"][r].append(rouge_score2_n[0][r]['r'])

fout = open('../output/eval-ES.txt', 'w', encoding='utf-8')
for r in rouge_methods:
    fout.write("***bert***\n")
    fout.write(r+":\n")
    fout.write("1: "+str(mean(rouge_score1["bert"][r]))+"\n")
    fout.write("2: " + str(mean(rouge_score2["bert"][r]))+"\n")

    fout.write("***bert_ori***\n")
    fout.write(r + ":\n")
    fout.write("1: " + str(mean(rouge_score1["bert_ori"][r]))+"\n")
    fout.write("2: " + str(mean(rouge_score2["bert_ori"][r]))+"\n")