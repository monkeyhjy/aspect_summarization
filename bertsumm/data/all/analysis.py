import os

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
    'GitlabCi': 246,
    'CircleCi': 480,
    'Jenkins': 716,
    'Travis': 985,
}

end = {
    'TeamCity': 246,
    'GitlabCi': 480,
    'CircleCi': 716,
    'Jenkins': 985,
    'Travis': 1240,
}

start_all = {
    'TeamCity': 44575,
    'GitlabCi': 1735,
    'CircleCi': 0,
    'Jenkins': 3440,
    'Travis': 47231,
}

end_all = {
    'TeamCity': 47231,
    'GitlabCi': 3440,
    'CircleCi': 1735,
    'Jenkins': 44575,
    'Travis': 51461,
}

def analysis_test():
    aspect_label = []
    senti_label = []
    file = open('test-all.txt', 'r', encoding='utf-8')
    tools = {}

    for line in file:
        line = line.split('\t')
        aspect_label.append(DKAAE_reversed_map[line[2].strip()])
        senti_label.append(line[1].strip())

    for t in tool_names:
        tool_detail = {}
        for a in aspect_names:
            tool_detail[a] = {'-1':0.0, '0':0.0, '1':0.0}
        for i in range(0, 1240):
            if start[t] <= i < end[t]:
                tool_detail[aspect_names[int(aspect_label[i])]][senti_label[i]] += 1
            else:
                continue
        for a in aspect_names:
            aspect_total = tool_detail[a]['-1']+tool_detail[a]['0']+tool_detail[a]['1']
            tool_detail[a]['total'] = aspect_total
            tool_detail[a]['1'] = tool_detail[a]['1']/aspect_total
            tool_detail[a]['0'] = tool_detail[a]['0'] / aspect_total
            tool_detail[a]['-1'] = tool_detail[a]['-1'] / aspect_total
        tools[t] =tool_detail
    print(tools)
    for t in tool_names:
        for a in aspect_names:
            print(str(round(tools[t][a]['1'], 3))+'\t'+str(round(tools[t][a]['0'], 3))+'\t'+str(round(tools[t][a]['-1'], 3)))


def analysis_all():
    aspect_label = []
    senti_label = []
    effect_id = []
    file_aspect = open('10.sum', 'r', encoding='utf-8')
    file_senti = open('train-all-out.txt', 'r', encoding='utf-8')
    # file_senti = open('software_output.txt', 'r', encoding='utf-8')
    tools = {}

    for line in file_senti:
        senti_label.append(line.split('\t')[2].strip())

    for line in file_aspect:
        line = line.split('\t')
        effect_id.append(int(line[0].strip()))
        aspect_label.append(line[1].strip())

    for t in tool_names:

        tool_detail = {}
        for a in aspect_names:
            tool_detail[a] = {'0': 0.0, '1': 0.0, '2': 0.0}

        j = 0
        for i in range(0, len(senti_label)):
        # for i in range(0, 2000):
            if i in effect_id:

                if start_all[t] <= i < end_all[t]:

                    tool_detail[aspect_names[int(aspect_label[j])]][senti_label[i]] += 1
                    # print(str(j) + "\t"+ str(i)+"\t" +aspect_label[j]+ "\t" +senti_label[i])
                    j = j + 1
                else:
                    j = j + 1
                    continue
        for a in aspect_names:
            aspect_total = tool_detail[a]['2'] + tool_detail[a]['0'] + tool_detail[a]['1']
            tool_detail[a]['total'] = aspect_total
            tool_detail[a]['1'] = tool_detail[a]['1'] / aspect_total
            tool_detail[a]['0'] = tool_detail[a]['0'] / aspect_total
            tool_detail[a]['2'] = tool_detail[a]['2'] / aspect_total
        tools[t] = tool_detail
        print(tools)
        os.system("pause")
    for t in tool_names:
        for a in aspect_names:
            print(str(round(tools[t][a]['0'], 3)) + '\t' + str(round(tools[t][a]['1'], 3)) + '\t' + str(
                round(tools[t][a]['2'], 3)))


analysis_all()