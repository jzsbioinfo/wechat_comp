

import pandas as pd

import numpy as np

from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from lightgbm.sklearn import LGBMClassifier

from sklearn.externals import joblib

from collections import defaultdict

import gc

import time
import pickle
import os 
from pathlib import Path
import logging

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)##打印加上时间等信息
        logger.addHandler(file_handler)
    return logger
logger = init_logger( 'log_b.txt')
logger.info('-------------------')
pd.set_option('display.max_columns', None)



def reduce_mem(df, cols):

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in tqdm(cols):

        col_type = df[col].dtypes

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    gc.collect()

    return df



## 从官方baseline里面抽出来的评测函数

def uAUC(labels, preds, user_id_list):

    """Calculate user AUC"""

    user_pred = defaultdict(lambda: [])

    user_truth = defaultdict(lambda: [])

    for idx, truth in enumerate(labels):

        user_id = user_id_list[idx]

        pred = preds[idx]

        truth = labels[idx]

        user_pred[user_id].append(pred)

        user_truth[user_id].append(truth)



    user_flag = defaultdict(lambda: False)

    for user_id in set(user_id_list):

        truths = user_truth[user_id]

        flag = False

        # 若全是正样本或全是负样本，则flag为False

        for i in range(len(truths) - 1):

            if truths[i] != truths[i + 1]:

                flag = True

                break

        user_flag[user_id] = flag



    total_auc = 0.0

    size = 0.0

    for user_id in user_flag:

        if user_flag[user_id]:

            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))

            total_auc += auc 

            size += 1.0

    user_auc = float(total_auc)/size

    return user_auc


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
#     import Path
#     if isinstance(file_path, Path):
#         file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data

# emb.drop('feed_embedding',inplace = True,axis = 1)
# save_pickle(emb,'data/wechat_algo_data1/feed_embeddings_512.pkl')




y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15







## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）

n_day = 5


test = load_pickle('./train/data/wechat_algo_data1/jin_test.pkl')
cols = load_pickle('./train/data/wechat_algo_data1/jin_cols.pkl')
cols = [i for i in cols if i!='user_date_weight_emd']

test[y_list[:4]] = 0.0
for seed in [2021,10086,13579,9394,456]:

    ##################### 全量训练 #####################

    #r_dict = dict(zip(y_list[:4], r_list))

    for y in y_list[:4]:

        print('=========', y, '=========')
        t = time.time()
        clf = load_pickle('../data/model/jin/seed_{}_day{}_train_model_{}.m'.format(seed,n_day,y))
        test[y] += clf.predict_proba(test[cols])[:, 1]/5
        print('runtime: {}\n'.format(time.time() - t))
        logger.info('runtime: {}\n'.format(time.time() - t))

if not os.path.exists('../data/submission/jin'):os.makedirs('../data/submission/jin')
test[['userid', 'feedid'] + y_list[:4]].to_csv(
    './lgb3.csv',
    index=False

)




