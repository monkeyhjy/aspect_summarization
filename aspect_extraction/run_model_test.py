import torch
from libs.my_torch import AspectAutoencoder, DocumentKnowledgeAssistedAE
from libs.dataset import parallel_chunks, get_vocab, load_word2vec, get_data, load_documents, load_keywords, get_test_data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from time import time
import numpy as np
from numpy.random import permutation, seed
from libs.loss import TripletMarginCosineLoss, OrthogonalityLoss
import libs.configuration as configuration
import gensim
from scipy.cluster.vq import kmeans
import torch.nn as nn
import torch.nn.functional as F
from evaluation import load_test_file, load_results, evaluation_7cata, start, end, tool_names, evaluate_tools, evaluation_size
import pandas as pd


def cos_sim(vector_a, vector_b):
    """
    Calculate cos similarity of two vectors
    :param vector_a: vector a
    :param vector_b: vector b
    :return: similarity
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def run_model(config, net, w2v_model, model, word2id, project_list, use_documents=True, use_test=False, data_size=0, tool=None):
    for epoch in range(config['epochs']):
        fsem = open('out/SO/{0}/semeval/{1}.key'.format(model, epoch + 1), 'w', encoding='utf-8')
        fsem.write('')
        fsem.close()

        fsum = open('out/SO/{0}/sumout/{1}.sum'.format(model, epoch + 1), 'w', encoding='utf-8')
        fsum.write('')
        fsum.close()
    print('Loading data...')
    id2word = {}
    world_vector = []
    data, scodes, original, test_labels = get_data(config, word2id, configuration.get_so_product_dict(), data_size, tool=tool)

    # data is the array of word ids in a sentence, the length of each ids in data are different
    batches = []
    batch_original = []
    batch_scodes = []
    for i, (segments, codes, segs_o), in enumerate(parallel_chunks(data, scodes, original, config['batch_size'])):
        max_len_batch = len(max(segments, key=len))
        batch_id = str(i)

        for j in range(len(segments)):
            # turn segments into the same length
            segments[j].extend([0] * (max_len_batch - len(segments[j])))
        batches.append(Variable(torch.from_numpy(np.array(segments, dtype=np.int32)).long()))

        batch_original.append(segs_o)
        batch_scodes.append(codes)

    if torch.cuda.is_available():
        net = net.cuda()

    # Print initial aspect
    temp = net.get_aspects().cpu()

    for a in temp:
        words = w2v_model.similar_by_vector(a.detach().numpy(), topn=10)
        print(words)
    if model == 'DKAAE':
        net.set_linear()
        world_vector = net.get_world_vector().cpu()
        words = w2v_model.similar_by_vector(world_vector.detach().numpy(), topn=10)
        print(words)

    rec_loss = TripletMarginCosineLoss()
    crossent_loss = nn.CrossEntropyLoss()
    orth_loss = OrthogonalityLoss()

    params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    print('Starting training...')
    print('\n')
    start_all = time()

    for epoch in range(config['epochs']):
        print('Epoch', str(epoch + 1))

        start = time()
        perm = permutation(len(batches))
        semout = ''
        sumout = ''

        for i in range(len(batches)):
            inputs = batches[i]
            sc = batch_scodes[i]
            orig = batch_original[i]

            if inputs.shape[1] < config['min_len']:
                continue

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                if model == 'DKAAE-mi':
                    sub_labels = sub_labels.cuda()
            if model == 'DKAAE':
                world_vector = net.get_world_vector()
                world_vector = world_vector / world_vector.norm(p=2, dim=0, keepdim=True)
                enc, out, a_probs = net(inputs, config['aspect_encoder'], world_vector.detach(), perm[i])
                # If you do not use attention, restore the following two lines and annotate above three lines
                # world_vector = None
                # enc, out, a_probs = net(inputs, config['aspect_encoder'], None, perm[i])
            if model == 'ABAE':
                out, a_probs = net(inputs, perm[i])
            if model == 'MATE':
                enc, out, a_probs = net(inputs, config['aspect_encoder'], None, perm[i])
            positives, negatives = net.get_targets()

            loss = rec_loss(out, positives, negatives)
            aspects = net.get_aspects()
            if not config['fix_a_emb']:
                loss += orth_loss(aspects)

            optimizer.zero_grad()
            loss.backward()
            print('loss:', str(loss))
            optimizer.step()

            for j in range(a_probs.size()[0]):

                semout += 'nan {0}'.format(sc[j])
                for a in range(a_probs.size()[1]):
                    semout += ' c{0}/{1:.4f}'.format(a, a_probs.data[j][a])
                semout += '\n'

                sumout += str(sc[j])
                sumout += '\t' + str(np.argmax(a_probs.data.cpu().numpy()[j]))
                sumout += '\t' + str(orig[j])

            fsem = open('out/SO/{0}/semeval/{1}.key'.format(model, epoch + 1), 'w', encoding='utf-8')
            fsem.write(semout)
            fsem.close()

            fsum = open('out/SO/{0}/sumout/{1}.sum'.format(model, epoch + 1), 'w', encoding='utf-8')
            fsum.write(sumout)
            fsum.close()

        # net.eval()
        net.train()

        print('({0:6.2f}sec)'.format(time() - start))

    labels = load_test_file('SO')
    results = load_results('SO', model_name, 10)
    p, r, f1 = evaluation_7cata(labels, results, model_name)
    print(f1)
    fout = open('../output/eval-AE.txt', 'w', encoding='utf-8')
    fout.write(str(f1))
    fout.close()
    if use_test:
        tests, t_scodes, t_original = get_test_data(config, word2id, tool)
        net.eval()
        max_len = len(max(tests, key=len))

        for j in range(len(tests)):
            tests[j].extend([0] * (max_len - len(tests[j])))
        tests = torch.from_numpy(np.array(tests, dtype=np.int32)).long()
        length = len(tests)

        world_vector = net.get_world_vector()
        world_vector = world_vector / world_vector.norm(p=2, dim=0, keepdim=True)
        enc, out, a_probs = net(tests, config['aspect_encoder'], world_vector.detach(), length)
        ftest = open('out/SO/{0}/test/10.sum'.format(model), 'w', encoding='utf-8')
        testout = ''
        for j in range(a_probs.size()[0]):

            testout += str(t_scodes[j])
            # for a in range(a_probs.size()[1]):
            #     sumout += '\t{0:.6f}'.format(a_probs.data[j][a])
            testout += '\t' + str(np.argmax(a_probs.data.cpu().numpy()[j]))
            testout += '\t' + str(t_original[j])
        ftest.write(testout)
        ftest.close()

    temp = net.get_aspects().cpu()
    topic_file = open(config['topic_file_path'], 'w', encoding='utf-8')
    for a in temp:
        words = w2v_model.similar_by_vector(a.detach().numpy(), topn=50)
        line = []
        for word in words:
            line.append(word[0])
        topic_file.write(' '.join(line))
        topic_file.write('\n')
    print('Finished training... ({0:.2f}sec)'.format(time() - start_all))
    print('\n')

    if config['savemodel'] != '':
        print('Saving model...')
        torch.save(net.state_dict(), config['savemodel'])


# Average 5 run
def run_5_times(word_count, document):
    config = configuration.get_config_DKAEE()
    fout = open('out/final.txt', 'w', encoding='utf-8')
    p_tools = {}
    r_tools = {}
    f1_tools = {}
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    time = 2
    for t in tool_names:
        p_tools[t] = 0.0
        r_tools[t] = 0.0
        f1_tools[t] = 0.0
    for i in range(0, time):

        net = DocumentKnowledgeAssistedAE(word_count, config['emb_size'], num_aspects=config['aspects'],
                                          neg_samples=config['negative'], w_emb=w_emb, a_weight=None,
                                          recon_method=config['recon_method'], attention=config['attention'],
                                          fix_w_emb=config['fix_w_emb'], instance_num=config['instance_num'],
                                          aspect_encoder=config['aspect_encoder'], aspect_size=config['aspect_size'],
                                          M=M, document=document, fix_a_emb=config['fix_a_emb'])

        run_model(config, net, model, model_name, word2id, project_list, use_documents=True, use_test=False, tool=None, data_size=0)
        labels = load_test_file('SO')
        results = load_results('SO', model_name, 10)
        p, r, f1 = evaluation_7cata(labels, results, model_name)

        precision += p
        recall += r
        f1_score += f1

        for t in tool_names:
            p, r, f1 = evaluate_tools(labels, results, model_name, start[t], end[t])
            p_tools[t] += p
            r_tools[t] += r
            f1_tools[t] += f1
    pre = f1_score / time
    fout.write("Precision:")
    fout.write(str(precision / time))
    fout.write('\n')

    fout.write("Recall:")
    fout.write(str(recall / time))
    fout.write('\n')

    fout.write("f1:")
    fout.write(str(f1_score / time))
    fout.write('\n')

    for t in tool_names:
        fout.write(t + '\n')
        fout.write(str(p_tools[t] / time))
        fout.write('\t')

        fout.write(str(r_tools[t] / time))
        fout.write('\t')

        fout.write(str(f1_tools[t] / time))
        fout.write('\n')
    return pre


if __name__ == "__main__":

    model_name = 'DKAAE'
    if model_name == 'ABAE':
        config = configuration.get_config_ABAE()
    elif model_name == 'DKAAE':
        config = configuration.get_config_DKAEE()
    else:
        config = configuration.get_config_MATE()

    model = gensim.models.KeyedVectors.load_word2vec_format(config['w2v_path'], binary=False, )

    # Test word vector
    words = model.similar_by_word('continuous')
    print(words)
    words = model.similar_by_word('integration')
    print(words)
    # Test word vector
    word_count, word2id, id2cnt = get_vocab(config['stem'])
    print(word2id)
    w2v = load_word2vec(config['w2v_path'], word2id)
    embed = np.random.uniform(-0.25, 0.25, (word_count, 200))
    embed[0] = 0
    for word, vec in w2v.items():
        embed[word2id[word]] = vec
    w_emb = torch.from_numpy(np.array(embed))
    project_list = ['travis']
    if model_name == 'ABAE':
        # kmeans initialization (ABAE)
        print('Running k-means...')
        a_emb, _ = kmeans(embed, config['aspects'], iter=20)
        a_emb = torch.from_numpy(a_emb)
        net = AspectAutoencoder(word_count, config['emb_size'],
                                num_aspects=config['aspects'], neg_samples=config['negative'],
                                w_emb=w_emb, a_emb=a_emb, recon_method=config['recon_method'], seed_w=None,
                                num_seeds=None, attention=config['attention'], fix_w_emb=config['fix_w_emb'],
                                fix_a_emb=config['fix_a_emb'])
        run_model(config, net, model, model_name, word2id, project_list, use_documents=False)
    else:

        M = None
        if config['attention_weight'] == 'eye':
            M = torch.eye(200, 200)
        if config['use_keyword']:
            project_documents, _ = load_keywords(config['document_path'], word2id, project_list)
        else:
            project_documents, _ = load_documents(config['document_path'], word2id, project_list)

        if config['aspect_encoder'] == 'FixedEncoder':
            a_weight = configuration.get_keyword_weights()
            a_weight = torch.from_numpy(np.array(a_weight, dtype=np.float))
        # project_documents: The document formed by the replaced document of each project
        documents = project_documents[0]
        print(documents)
        # add general aspect to documents arrays
        max_len_doc = len(max(documents, key=len))

        for j in range(len(documents)):
            # turn segments into the same length
            documents[j].extend([0] * (max_len_doc - len(documents[j])))
        # documents is the array of word ids in a document, the length of each ids in documents are different
        document = torch.from_numpy(np.array(documents, dtype=np.int32)).long()

        # net = DocumentKnowledgeAssistedAE(word_count, config['emb_size'], num_aspects=config['aspects'],
        #                                   neg_samples=config['negative'], w_emb=w_emb, a_weight=None,
        #                                   recon_method=config['recon_method'], attention=config['attention'],
        #                                   fix_w_emb=config['fix_w_emb'], instance_num=config['instance_num'],
        #                                   aspect_encoder=config['aspect_encoder'], aspect_size=config['aspect_size'],
        #                                   M=M, document=document, fix_a_emb=config['fix_a_emb'])
        # run_model(config, net, model, model_name, word2id, project_list, use_documents=True, use_test=False, tool=None, data_size=0)

        # Average result of 5 run. When running 5 times, please annotate above net and run_model
        run_5_times(word_count, document)
