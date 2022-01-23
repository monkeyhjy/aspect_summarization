from summarizer import Summarizer
from transformers import AutoConfig, AutoTokenizer, AutoModel
import reader
import os
import numpy as np


def use_mlm(i, use_test=False):
    aspect_names = [
        'compatibility', 'documentation', 'functional', 'performance', 'reliability', 'usability', 'other'
    ]

    if use_test:
        tools = reader.read_test()
    else:
        tools = reader.read_all()
    # Please change the test_mlm_path to specific path of computer
    test_mlm_path = "F:/bert_model/test-mlm"
    custom_config = AutoConfig.from_pretrained(test_mlm_path)
    custom_config.output_hidden_states=True
    custom_tokenizer = AutoTokenizer.from_pretrained(test_mlm_path)
    custom_model = AutoModel.from_pretrained(test_mlm_path, config=custom_config)
    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer, random_state=i)
    print("load data complete")

    tool_names = [
        'Jenkins', 'Travis', 'TeamCity', 'CircleCi', 'GitlabCi'
    ]
    for aspect in aspect_names:

        for t in tool_names:
            length = len(tools[t][aspect]['0'])

            body = "\n".join(tools[t][aspect]['0'])

            print(i)
            print("\n")
            if length < 3:
                res = 1
            else:
                res = model.calculate_optimal_k(body, k_max=min(5, length - 1))

            if length == 0:
                result = []
            else:
                result = model(body, num_sentences=res, ratio=1.0, min_length=1, max_length=600, use_first=False)

            length = len(tools[t][aspect]['2'])
            body2 = "\n".join(tools[t][aspect]['2'])

            if length < 3:
                res = 1
            else:
                res = model.calculate_optimal_k(body2, k_max=min(5, length - 1))

            if length == 0:
                result2 = []
            else:
                result2 = model(body2, num_sentences=res)

            if use_test:
                path = "data/result/bert/test/"
            else:
                path = "data/result/bert/"
            fout = open(path + t + "." + aspect + ".txt", "w", encoding="utf-8")
            fout.write("positive:\n")
            for s in result:
                fout.write(s)
                fout.write("\n")
            fout.write("negative:\n")
            for s in result2:
                fout.write(s)
                fout.write("\n")


def use_bert(i, use_test=False):
    aspect_names = [
        'compatibility', 'documentation', 'functional', 'performance', 'reliability', 'usability', 'other'
    ]
    # aspect_names = [
    #     'performance', 'reliability', 'usability', 'other'
    # ]
    if use_test:
        tools = reader.read_test()
    else:
        tools = reader.read_all()
    print("load data complete")

    model = Summarizer(model='bert-base-uncased', random_state=i)
    tool_names = [
        'Jenkins', 'Travis', 'TeamCity', 'CircleCi', 'GitlabCi'
    ]
    for aspect in aspect_names:

        for t in tool_names:

            length = len(tools[t][aspect]['0'])

            body = "\n".join(tools[t][aspect]['0'])

            print(i)
            print("\n")
            if length < 3:
                res = 1
            else:
                res = model.calculate_optimal_k(body, k_max=min(5, length-1))

            if length == 0:
                result = []
            else:
                result = model(body, num_sentences=res, ratio=1.0, min_length=1, max_length=600, use_first=False)

            length = len(tools[t][aspect]['2'])
            body2 = "\n".join(tools[t][aspect]['2'])

            if length < 3:
                res = 1
            else:
                res = model.calculate_optimal_k(body2, k_max=min(5, length - 1))

            if length == 0:
                result2 = []
            else:
                result2 = model(body2, num_sentences=res)

            if use_test:
                path = "data/result/bert_ori/test/"
            else:
                path = "data/result/bert_ori/"
            fout = open(path + t + "." + aspect + ".txt", "w", encoding="utf-8")
            fout.write("positive:\n")
            for s in result:
                fout.write(s)
                fout.write("\n")
            fout.write("negative:\n")
            for s in result2:
                fout.write(s)
                fout.write("\n")


# seed = np.random.randint(0, 1000)
# use_bert(seed, use_test=True)
# use_mlm(seed, use_test=False)