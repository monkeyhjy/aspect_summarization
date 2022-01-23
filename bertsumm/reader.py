# read test data and output data that can be read by the backend
import os
import sys

gold_map = {
    'community': '0',
    'compatibility': '1',
    'documentation': '2',
    'functional': '3',
    'performance': '4',
    'reliability': '5',
    'usability': '6',
    'other': '7'
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

tool_names = [
    'Jenkins', 'Travis', 'TeamCity', 'CircleCi', 'GitlabCi'
]

aspect_names = [
    'compatibility', 'documentation', 'functional', 'performance', 'reliability', 'usability', 'other'
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

start_all = {
    'TeamCity': 70603,
    'GitlabCi': 2831,
    'CircleCi': 0,
    'Jenkins': 6285,
    'Travis': 74086,
}

end_all = {
    'TeamCity': 74086,
    'GitlabCi': 6285,
    'CircleCi': 2831,
    'Jenkins': 70603,
    'Travis': 79922,
}


def read_test():
    aspect_label = []
    senti_label = []
    origin_sentence = []
    file = open(os.path.dirname(__file__)+'/data/test/test-all.txt', 'r', encoding='utf-8')
    tools = {}

    for line in file:
        line = line.split('\t')
        origin_sentence.append(line[0])
        aspect_label.append(DKAAE_reversed_map[line[1].strip()])
        tmp = line[2].strip()
        senti_label.append(str(int(tmp) + 1))

    for t in tool_names:
        tool_detail = {}
        for a in aspect_names:
            tool_detail[a] = {'0': [], '1': [], '2': []}
        for i in range(0, 1775):
            if start[t] <= i < end[t]:
                tool_detail[aspect_names[int(aspect_label[i])]][senti_label[i]].append(origin_sentence[i].lower())
            else:
                continue
        tools[t] =tool_detail
    return tools


def read_all():
    print(os.path.dirname(__file__))

    aspect_label = []
    senti_label = []
    effect_id = []
    origin_sentence = []
    file_aspect = open(os.path.dirname(__file__)+'/data/all/10.sum', 'r', encoding='utf-8')
    file_senti = open(os.path.dirname(__file__)+'/data/all/train-all-out.txt', 'r', encoding='utf-8')
    # file_senti = open('software_output.txt', 'r', encoding='utf-8')
    tools = {}

    for line in file_senti:
        # line = line.replace("/code_segment/", "")
        # line = line.replace("/CODE_SEGMENT/", "")
        origin_sentence.append(line.split('\t')[0])
        senti_label.append(line.split('\t')[1].strip())

    for line in file_aspect:
        line = line.split('\t')
        effect_id.append(int(line[0].strip()))
        aspect_label.append(line[1].strip())

    for t in tool_names:

        tool_detail = {}
        for a in aspect_names:
            tool_detail[a] = {'0': [], '1': [], '2': []}

        j = 0
        for i in range(0, len(senti_label)):
        # for i in range(0, 2000):
            if i in effect_id:

                if start_all[t] <= i < end_all[t]:

                    tool_detail[aspect_names[int(aspect_label[j])]][senti_label[i]].append(origin_sentence[i].lower())
                    # print(str(j) + "\t"+ str(i)+"\t" +aspect_label[j]+ "\t" +senti_label[i])
                    j = j + 1
                else:
                    j = j + 1
                    continue

        tools[t] = tool_detail
    return tools


def read_and_output():
    result = read_all()
    output_path = "data/result/classify_result.txt"
    output_file = open(output_path, "w", encoding='utf-8')
    for tool in tool_names:
        for aspect in aspect_names:
            for s in result[tool][aspect]['0']:
                line = s + '\t' + tool + '\t' + aspect + '\t' + '0' +'\n'
                output_file.write(line)
            for s in result[tool][aspect]['1']:
                line = s + '\t' + tool + '\t' + aspect + '\t' + '1' +'\n'
                output_file.write(line)
            for s in result[tool][aspect]['2']:
                line = s + '\t' + tool + '\t' + aspect + '\t' + '2' +'\n'
                output_file.write(line)
    output_file.close()


def read_and_output_test():
    print("Loading data")
    result = read_test()
    output_path = "data/test/"

    for tool in tool_names:
        length = 0
        for aspect in aspect_names:
            output_file = open(output_path+tool+"/"+aspect+".txt", "w", encoding='utf-8')
            output_file.write("Negative:\n")
            for s in result[tool][aspect]['0']:
                output_file.write(s+"\n")

            output_file.write("\nPositive:\n")
            for s in result[tool][aspect]['2']:
                output_file.write(s + "\n")
            output_file.close()

            length += len(result[tool][aspect]['2'])
            length += len(result[tool][aspect]['0'])
        print(tool+":")
        print(length)


# output = open("sentiment_statistic.txt", "w", encoding="utf-8")
# result = read_all()
# for tool in tool_names:
#     output.write(tool+":\n")
#     for aspect in aspect_names:
#         sum = len(result[tool][aspect]['0'])+len(result[tool][aspect]['2'])
#         p1 = round(len(result[tool][aspect]['0'])/sum*100, 1)
#         output.write(str(p1)+"%\t")
#         p2 = round(len(result[tool][aspect]['2'])/sum*100, 1)
#         output.write(str(p2) + "%\n")

