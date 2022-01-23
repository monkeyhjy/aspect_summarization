current_model = 'DKAAE'


def get_config_ABAE():
    config = {
        'emb_size': 200,
        'aspects': 7,
        'neg_samples': 20,
        'w2v_path': 'libs/SO_vectors_small_full-stem.txt',
        'stem': True,
        'recon_method': 'centr',
        'attention': True,
        'fix_w_emb': True,
        'fix_a_emb': False,
        'learning_rate': 0.001,
        'epochs': 15,
        'padding': 0,
        'min_len': 1,
        'max_len': 300,
        'batch_size': 50,
        'savemodel': '',
        'negative': 20,
        'topic_file_path': 'out/SO/ABAE/topics.txt',
        'attention_weight': None
    }
    return config


def get_config_MATE():
    config = {
        'emb_size': 200,
        'aspects': 7,
        'w2v_path': 'libs/SO_vectors_small_full-stem.txt',
        'stem': True,
        'recon_method': 'centr',
        'attention': True,
        'fix_w_emb': True,
        'fix_a_emb': False,
        'activate_func': 'tanh',
        'learning_rate': 0.005,
        'epochs': 10,
        'padding': 0,
        'min_len': 1,
        'max_len': 300,
        'batch_size': 50,
        'savemodel': '',
        'negative': 20,
        'aspect_encoder': 'FixedEncoder',
        'instance_num': 1,
        'aspect_size': 10,
        'document_path': '../preprocessed_data/keywords-lin-iso-no-doc-2.txt',
        'topic_file_path': 'out/SO/MATE/topics.txt',
        'attention_weight': 'Random',
        # 'attention_weight': None
        'use_keyword': True
    }
    return config


def get_config_DKAEE():
    config = {
        'emb_size': 200,
        'aspects': 7,
        'w2v_path': 'libs/SO_vectors_small_full-stem.txt',
        'stem': True,
        'recon_method': 'centr',
        'attention': True,
        'fix_w_emb': True,
        'fix_a_emb': True,
        'activate_func': 'sigmod',
        'learning_rate': 0.005,
        'epochs': 10,
        'padding': 0,
        'min_len': 1,
        'max_len': 300,
        'batch_size': 50,
        'savemodel': '',
        'negative': 20,
        'aspect_encoder': 'AttentionEncoder',
        'instance_num': 1,
        'aspect_size': 10,
        'document_path': '../preprocessed_data/keywords-lin-iso-no-doc-2.txt',
        'topic_file_path': 'out/SO/DKAAE/topics.txt',
        'attention_weight': 'eye',
        'use_keyword': True
    }
    return config


def get_so_product_dict():
    product_dict = {
        'travis': 0
    }
    return product_dict


def get_app_product_dict():
    product_dict = {

    }
    return product_dict

def get_keyword_weights():
    weights = [
        [0.25734,0.18572,0.11059,0.10284,0.10284,0.10284,0.10284,0.10284,0.10284,0.10284],
        [0.15414,0.10587,0.10587,0.10587,0.10587,0.10587,0.10587,0.10587,0.10587,0.10587],
        [0.04262,0.02910,0.02526,0.02333,0.02333,0.02333,0.01866,0.01866,0.01866,0.01866],
        [0.31319,0.22614,0.21674,0.17296,0.17296,0.12529,0.12529,0.12529,0.12529,0.12529],
        [0.22519,0.22519,0.09008,0.09008,0.09008,0.09008,0.09008,0.09008,0.09008,0.09008],
        [0.13582,0.13582,0.13132,0.07525,0.07525,0.07525,0.07525,0.07525,0.07525,0.07525],
        [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    ]