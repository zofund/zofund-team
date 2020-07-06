import os
from os.path import join
#  --------------------------路径相关------------------------------
# 项目目录
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# source目录
SRC_PATH = join(PROJECT_PATH, 'source/')
# 配置文件目录
SETTING_PATH = join(PROJECT_PATH, 'Setting.conf')
# Input Path
INPUT_PATH = join(PROJECT_PATH, 'data', 'input')
# Output Path
OUTPUT_PATH = join(PROJECT_PATH, 'data', 'output')
# BERT Path
BERT_PATH = join(PROJECT_PATH, 'data', 'BERT')
# Bert-Base-Chinese Path
BERT_BASE_CN_PATH = join(BERT_PATH, 'bert-base-chinese')
# Log Dir
LOG_DIR = join(PROJECT_PATH, 'data', 'logging')
# Saved Model Dir
SAVED_MODEL_DIR = join(PROJECT_PATH, 'data', 'saved_model')
# Saved Bert Path
SAVED_BERT_PATH = join(SAVED_MODEL_DIR, 'm01bert/')
if not os.path.exists(SAVED_BERT_PATH):
    os.mkdir(SAVED_BERT_PATH)
# Saved LSTM Pah
SAVED_LSTM_PATH = join(SAVED_MODEL_DIR, 'm01lstm/')
if not os.path.exists(SAVED_LSTM_PATH):
    os.mkdir(SAVED_LSTM_PATH)


#  ----------------------------模型相关-----------------------
# Token最大长度
MAX_LENGTH_DEFAULT = 500
# Labels, 请勿更改
LABEL_COLS = [
    'prosperity',
    'occupancy',
    'transition',
    'product_price',
    'exceed_expectation',
    'score_0',
    'score_1',
    'score_2',
    'score_3',
    'score_4',
    'score_5',
    'score_6',
    'score_7',
    'score_8',
    'score_9',
    'score_10'
]
# Label List
LABEL_LIST = [0, 1]
# random seed
RANDOM_SEED = 100
