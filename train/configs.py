import copy
import torch

default_config = {
    'batch_size': 24,
    'max_sent_cnt': 8, # number of sentences to embedd into memory_vect
    'max_sent_len': 32, # length of single memory_vect sentence
    'max_sent_len_decoder': 16,
    'word_edim': 1024, # embeding dimension of the memory_vect sentence
    's2v_dim': 4*1024,
    'mem_vect_dim': 4*1024,
    'use_memory': True,
    'name': '',
    'restore_name': '',

    'sentence_encoder': {
        'input_drop': 0.0,
        'gating': {
            'gate1': {
                'gate_network': [{'network': 'Mha', 'input_dim': 1024, 'output_dim': 1024, 'output_activation': None}, 
                                 {'network': 'Fc', 'input_dim': 1024, 'output_dim': 12*1024, 'output_activation': torch.sigmoid}],
                'main_network': [{'network': 'Fc', 'input_dim': 1024, 'output_dim': 12*1024, 'output_activation': None}]
            },
            'gate2': {
                'gate_network': [{'network': 'Fc', 'input_dim': 12*1024, 'output_dim': 4*1024, 'output_activation': torch.sigmoid}],
                'main_network': [{'network': 'Fc', 'input_dim': 12*1024, 'output_dim': 4*1024, 'output_activation': None}]
            },
        },
        'transformer': {
            'word_dim': 1024,
            'num_layers': 0,
            'num_heads': 0,
            'ffn_dim': 0,
            'dropout': 0.0,
            'gate': False,
            'res_activation': None
        },
        'pooling': {
            'pooling_method': '',
            'mha': {
                'attention_dropout': 0.0,
                'num_heads': 128,
                'dropout': 0.0
            },
            'pooling_function': 'max',
        },
    },

    's2v_encoder': {
        'input_drop': 0.0,
        'transformer': {
            'num_layers': 0,
            'num_heads': 0,
            'ffn_dim': 0,
            'dropout': 0.0,
            'res_activation': None
        },
        'pooling': {
            'pooling_method': '',
            'mha': {
                'num_heads': 128,
                'dropout': 0.0
            },
            'pooling_function': 'max',
        },
    },

    'training': {
        'optimizer': 'Adam',
        'SAM_optimizer': False,
        'clipnorm': 1.,
        'lr': 2e-4,
        'lr_gamma': 1,
        'lr_step': 1000,
        'epochs': 1000,
        'log': True
    }
}

configs = []
for i in range(1000):
    configs.append(copy.deepcopy(default_config))

i = 0
# -----------------------------------------------------------------------------
for _ in range(0, 100):
    if i == 0:
        word_edim = 768
        configs[i]['word_edim'] = word_edim
        configs[i]['training']['lr'] = 1e-6
        configs[i]['training']['lr_step'] = 1
        configs[i]['training']['lr_gamma'] = 0.9
        configs[i]['training']['epochs'] = 20000
        configs[i]['batch_size'] = 1
        configs[i]['max_sent_len'] = 24

    configs[i]['name'] = 'b' + str(configs[i]['batch_size']) + \
        '_' + configs[i]['training']['optimizer'] + 'lr' + str(configs[i]['training']['lr']) + \
        's' + str(configs[i]['training']['lr_step']) + 'g' + str(configs[i]['training']['lr_gamma']) + \
        '_sC' + str(configs[i]['max_sent_cnt']) + '.sL' + str(configs[i]['max_sent_len']) + \
        '.s2vD' + str(configs[i]['s2v_dim']) + \
        '.mvD' + str(configs[i]['mem_vect_dim']) + \
        '_s.dr' + str(configs[i]['sentence_encoder']['input_drop']) + \
        '.Trx' + str(configs[i]['sentence_encoder']['transformer']['num_layers']) + \
        '.g' + str(configs[i]['sentence_encoder']['transformer']['gate'])[0] + \
        '.ffn' + str(configs[i]['sentence_encoder']['transformer']['ffn_dim']) + \
        '.' + str(configs[i]['sentence_encoder']['pooling']['pooling_method']) + \
        '.h' + str(configs[i]['sentence_encoder']['pooling']['mha']['num_heads']) + \
        '.' + str(configs[i]['sentence_encoder']['pooling']['pooling_function']) + \
        '_s2v.dr' + str(configs[i]['s2v_encoder']['input_drop']) + \
        '.gTrx' + str(configs[i]['s2v_encoder']['transformer']['num_layers']) + \
        '.h' + str(configs[i]['s2v_encoder']['transformer']['num_heads']) + \
        '.ffn' + str(configs[i]['s2v_encoder']['transformer']['ffn_dim']) + \
        '.' + str(configs[i]['s2v_encoder']['pooling']['pooling_method']) + \
        '.' + str(configs[i]['s2v_encoder']['pooling']['pooling_function']) + \
        '_v039_LongformerBase_MemV.postNorm.sin_6xTr.FfnMha_context_word_snli_qa_' + str(i)
        # '_v038_RobertaBase_MemV.postNorm.sin_6xTr.FfnMha_context_word_snli_zsre_' + str(i)
    i += 1