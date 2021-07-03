#!/usr/bin/env python
# coding: utf-8

# # 导包

# In[2]:


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
import pickle
import random
import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm
def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
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
seed_everything()


###feed_embedding
emd = pd.read_csv('../../data/wedata/feed_embeddings.csv')
for i in tqdm(range(512)):
    emd['emd_{}'.format(i) ] = emd['feed_embedding'].apply(lambda x:float(x.split(' ')[i]))

from sklearn.decomposition import PCA
dim = 32
pca = PCA(n_components=dim)
pca_result = pca.fit_transform(emd[['emd_{}'.format(i) for i in range(512)]].values)
emx = pd.read_csv('../../data/wedata/feed_embeddings.csv')
for i in range(dim):
   emx['emd_{}'.format(i)] =  pca_result[:,i]
emx.drop('feed_embedding',axis = 1,inplace = True)
save_pickle(emx,'../../data/wedata/feed_embeddings_{}.pkl'.format(dim))
del emd,emx

logger.info('feed_embedding pca over')


###user_weight_emd
train = pd.read_csv('../../data/wedata/user_action.csv')
user_feed_date_list = train.groupby(['userid','date_']).apply(lambda x: [str(i) for i in list(x['feedid'])]).reset_index()
user_feed_date_sum = user_feed_date_list.groupby(['userid']).apply(lambda x:x['date_'].sum()).reset_index()
user_feed_date_sum.columns = ['userid','date_sum']
user_feed_date_list = user_feed_date_list.merge(user_feed_date_sum,on = 'userid')
user_feed_date_list['rate'] = user_feed_date_list['date_']/user_feed_date_list['date_sum']
emd = pd.read_csv('../../data/wedata/feed_embeddings.csv')
emd.index = emd.feedid
emd_dict = {}
for index in tqdm(emd.index):
    emd_dict[index] = [float(i) for i in emd.loc[int(index),'feed_embedding'].split(' ')[0:-1]]
    
def getemb(x):
    emds = []
    for i in x:
        emds.append(emd_dict[int(i)])
    emds = np.array(emds)
    emds = np.mean(emds,axis = 0)
    return list(emds)
user_feed_date_list['emd'] = user_feed_date_list[0].apply(getemb)
user_feed_date_list['emd'] = user_feed_date_list['emd'].apply(lambda x:np.array(x))
user_feed_date_list['date_emd'] = user_feed_date_list['emd']*user_feed_date_list['rate']
user_feed_date_list['emd'] = user_feed_date_list['emd'].apply(lambda x:np.array(x))
user_feed_date_list['date_emd'] = user_feed_date_list['emd']*user_feed_date_list['rate']
user_date_emd = user_feed_date_list.groupby('userid').apply(lambda x:sum(x['date_emd'])).reset_index()
user_date_emd.columns = ['userid','user_date_weight_emd']
emb = user_date_emd.copy()
save_pickle(user_date_emd,'../../data/wedata/user_date_weigh_emd.pkl')
for i in range(512):
    user_date_emd['user_date_weight_emd_{}'.format(i)] = user_date_emd['user_date_weight_emd'].apply(lambda x:x[i])

from sklearn.decomposition import PCA
dim = 32
pca = PCA(n_components=dim)
pca_result = pca.fit_transform(user_date_emd[['user_date_weight_emd_{}'.format(i) for i in range(512)]].values)
for i in range(dim):
    emb['user_weight_emd_{}'.format(i)] =  pca_result[:,i]
# emb.drop('user_date_weight_emd',axis = 1,inplace = True)
save_pickle(emb,'../../data/wedata/user_weight_emd_{}.pkl'.format(dim))
del train,emd,user_feed_date_list
logger.info('user_wight_emb pca over')
# # 定义工具函数

# In[3]:


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


# # 读取数据

# In[4]:


y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15



## 读取训练集

train = pd.read_csv('../../data/wedata/user_action.csv')
print('train.shape:',train.shape)
for y in y_list:
    print(y, train[y].mean())



## 读取测试集

test = pd.read_csv('../../data/wedata/test_b.csv')
test['date_'] = max_day
print('test.shape',test.shape)



## 合并处理
df = pd.concat([train, test], axis=0, ignore_index=True)





## 读取视频信息表

feed_info = pd.read_csv('../../data/wedata/feed_info.csv')
## 此份baseline只保留这三列

feed_info = feed_info[[
    'feedid', 'authorid', 'videoplayseconds'
]]



df = df.merge(feed_info, on='feedid', how='left')


# # 特征工程

# In[5]:


## 视频时长是秒，转换成毫秒，才能与play、stay做运算

df['videoplayseconds'] *= 1000

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）

df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')

df['play_times'] = df['play'] / df['videoplayseconds']

play_cols = [

    'is_finish', 'play_times', 'play', 'stay'

]



## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）

n_day = 5

for stat_cols in tqdm([

    ['userid'],

    ['feedid'],

    ['authorid'],

    ['userid', 'authorid']

]):

    f = '_'.join(stat_cols)

    stat_df = pd.DataFrame()

    for target_day in range(2, max_day + 1):

        left, right = max(target_day - n_day, 1), target_day - 1



        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)

        tmp['date_'] = target_day



        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')



        g = tmp.groupby(stat_cols)

        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')



        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]



        for x in play_cols[1:]:

            for stat in ['max', 'mean']:

                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)

                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))



        for y in y_list[:4]:

            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')

            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')

            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])



        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)

        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)

        del g, tmp

    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')

    del stat_df

    gc.collect()





## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

for f in tqdm(['userid', 'feedid', 'authorid']):

    df[f + '_count'] = df[f].map(df[f].value_counts())

for f1, f2 in tqdm([

    ['userid', 'feedid'],

    ['userid', 'authorid']

]):

    df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')

    df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

for f1, f2 in tqdm([

    ['userid', 'authorid']

]):

    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')

    df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)

    df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')

df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')

df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')


# # 训练

# In[ ]:



## 内存够用的不需要做这一步
emb = load_pickle('../../data/wedata/feed_embeddings_32.pkl')
df = df.merge(emb, on='feedid', how='left')

user_feed_emd = load_pickle('../../data/wedata/graph_walk_emb_32.pkl')
df = df.merge(user_feed_emd, on='userid', how='left')

#user_feed_emd = load_pickle('data/wechat_algo_data1/user_author_emd.pkl')
#df = df.merge(user_feed_emd, on='userid', how='left')

user_weight_emd = load_pickle('../../data/wedata/user_weight_emd_32.pkl')
df = df.merge(user_weight_emd, on='userid', how='left')

keyword_tag = load_pickle('../../data/wedata/keyword_tag_8.pkl')
df = df.merge(keyword_tag, on='feedid', how='left')

df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])



train = df[~df['read_comment'].isna()].reset_index(drop=True)

test = df[df['read_comment'].isna()].reset_index(drop=True)

cols = [f for f in df.columns if f not in ['date_'] + play_cols + y_list]

print(train[cols].shape)
if not os.path.exists('../train/data/wechat_algo_data1/'):os.makedirs('../train/data/wechat_algo_data1/')
save_pickle(train,'../train/data/wechat_algo_data1/jin_train.pkl')
save_pickle(test,'../train/data/wechat_algo_data1/jin_test.pkl')
save_pickle(cols,'../train/data/wechat_algo_data1/jin_cols.pkl')
logger.info('feat  over')




