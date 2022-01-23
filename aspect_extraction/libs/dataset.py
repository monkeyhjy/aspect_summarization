import gensim
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re
import random
from evaluation import get_test_label
num_regex = re.compile(r'^[+-]?[0-9]+\.?[0-9]*$')

def is_number(token):
    return bool(num_regex.match(token))


def parallel_chunks(l1, l2, l3, n):
    """
    Yields chunks of size n from 3 lists in parallel
    """
    if len(l1) != len(l2) or len(l2) != len(l3):
        print(len(l1), len(l2), len(l3))
        raise IndexError
    else:
        for i in range(0, len(l1), n):
            yield l1[i:i+n], l2[i:i+n], l3[i:i+n]


def get_vocab(stem, train=False):
    if train:
        if stem:
            f = open('../preprocessed_data/SO/vocab-all-stem', 'r', encoding='utf-8')
            print('opened vocab-all-stem!')
    else:
        if stem:
            f = open('../preprocessed_data/SO/vocab-full-stem', 'r', encoding='utf-8')
            print('opened vocab-full-stem!')

        else:
            f = open('../preprocessed_data/SO/vocab-full', 'r', encoding='utf-8')

    count = 1
    id = 1
    word2id = {}
    id2cnt = {}
    for line in f:
        tokens = line.split('\t')
        count = count + 1
        word2id[tokens[0]] = id
        id2cnt[id] = tokens[1]
        id = id + 1
    return count, word2id, id2cnt


def load_word2vec(fname, vocab):
    """
    Loads 200x1 word vecs from Stack Overflow word2vec
    """
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=False)

    # for word in model.wv.vocab:
    #     if word in vocab:
    #         word_vecs[word] = list(model.wv[word])
    for word in vocab:
        try:
            word_vecs[word] = list(model.get_vector(word))
        except Exception:
            continue

    return word_vecs


def get_data(config, word2id, product_dict, data_size=0, tool=None):
    padding = config['padding']  # 默认为0
    data = []
    products = ['travis', 'travis ci', 'travisci']

    scodes = []
    original = []
    
    if config['stem']:
        if tool is None:
            f = open('../preprocessed_data/SO/test-stem.txt', 'r', encoding='utf-8')
        else:
            f = open('../preprocessed_data/SO/tools/train-{0}-stem.txt'.format(tool), 'r', encoding='utf-8')
            # f = open('../preprocessed_data/SO/tools/test/{0}.txt'.format(tool), 'r', encoding='utf-8')
    else:
        f = open('../preprocessed_data/SO/train.txt', 'r', encoding='utf-8')

    index = 0
    for line in f:
        # if len(line.split(' ')) > 50:
        #     index = index + 1
        #     continue
        # line.replace("/CODE_SEGMENT/", "")
        # line.replace("/pre/", "")
        # line.replace("/url/", "")
        original.append(line)
        scodes.append(index)
        ids = []
        line = line.replace('\n', '')
        words = line.split(' ')
        for word in words:
            # if word not in products:
            #     try:
            #         ids.append(word2id[word])
            #     except KeyError:
            #         # print(word)
            #         continue
            # else:
            #     print(word)
            #     continue
            if is_number(word):
                word = 'num'

            try:
                ids.append(word2id[word])
            except KeyError:
                # print(word)
                continue

        ids = [0] * padding + ids + [0] * padding
        data.append(ids)
        index = index + 1
    if data_size != 0:
        index = random.sample(range(0, 1240), data_size)
        index.sort()
        data_new = []
        scodes_new = []
        original_new = []
        test_new = []
        test = get_test_label()
        for i in index:
            data_new.append(data[i])
            scodes_new.append(scodes[i])
            original_new.append(original[i])
            test_new.append(test[i])
        return data_new, scodes_new, original_new, test_new
    return data, scodes, original, []


def get_test_data(config, word2id, tool=None):
    padding = config['padding']  # 默认为0
    data = []

    scodes = []
    original = []

    if config['stem']:
        f = open('../preprocessed_data/SO/train-all-stem.txt', 'r', encoding='utf-8')
        if tool is not None:
            f = open('../preprocessed_data/SO/tools/test/{0}.txt'.format(tool), 'r', encoding='utf-8')
    else:
        f = open('../preprocessed_data/SO/test.txt', 'r', encoding='utf-8')

    index = 0
    for line in f:
        if len(line.split(' ')) > 50:
            index = index + 1
            continue

        original.append(line)
        scodes.append(index)
        ids = []
        line = line.replace('\n', '')
        words = line.split(' ')
        for word in words:
            if is_number(word):
                word = 'num'
            try:
                ids.append(word2id[word])
            except KeyError:
                # print(word)
                continue

        ids = [0] * padding + ids + [0] * padding
        data.append(ids)
        index = index + 1
    return data, scodes, original


def load_keywords(path, word2id, project_list):
    project_doc = []
    project_aspect = []
    for p in project_list:
        f = open(path, 'r', encoding='utf-8')
        id2name = {}
        documents = []

        for line in f:
            document = []
            text_token = re.split(" ", line)

            for word in text_token:
                try:
                    temp = word.split(":")
                    document.append(word2id[temp[0].strip()])
                except KeyError:
                    print(temp[0])
                    continue
            documents.append(document)
        project_doc.append(documents)
        project_aspect.append(id2name)
    return project_doc, project_aspect


def load_documents(path, word2id, project_list):
    project_doc = []
    project_aspect = []
    for p in project_list:
        f = open(path, 'r', encoding='utf-8')
        first_line = True
        id2name = {}
        documents = []
        aspect_name = ''

        for line in f:
            if first_line:
                id, name = line.split(' ')
                id2name[id] = name.strip()
                aspect_name = name.split(' ')
                first_line = False
            else:
                if len(line.strip()) == 0:
                    print('in empty line!!!')
                    first_line = True
                else:
                    document = []
                    text_token = re.split(" |\t", line)

                    for word in text_token:
                        try:
                            document.append(word2id[word.strip()])
                        except KeyError:
                            print(word)
                            continue
                    for a in aspect_name:
                        document.append(word2id[a.strip()])
                    documents.append(document)
        project_doc.append(documents)
        project_aspect.append(id2name)
    return project_doc, project_aspect




