
# -*- coding: utf-8 -*-
from tqdm import tqdm

import gc
import numpy as np
import lightgbm as lgb
from evaluation import uAUC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
# import seaborn as sns
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# 存储数据的根目录
ROOT_PATH = "../../data/wedata"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_b.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']



def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
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
    print('start memory {:.2f} Mb, end memory {:.2f} Mb reduced ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


feed_info = reduce_mem(pd.read_csv(FEED_INFO))
user_action_df = reduce_mem(pd.read_csv(USER_ACTION))
feed_embed = pd.read_csv(FEED_EMBEDDINGS)
test = reduce_mem(pd.read_csv(TEST_FILE))
test['date_'] = 15


feed_use_feature = ['feedid','authorid','videoplayseconds','bgm_song_id', 'bgm_singer_id']
feed_info_use = feed_info[feed_use_feature] # feed_info
# feed_info_use.loc[feed_info_use['videoplayseconds']>=300,'videoplayseconds'] = 300



data_df = pd.concat([user_action_df, test], axis=0, ignore_index=True)
data_df = pd.merge(data_df, feed_info_use, on=['feedid'],how='left',copy=False)




## user feed embed -----------------------------------------------------------------


userid_feedid_d2v_max_wd = pd.read_pickle('./userid_feedid_d2v_b.pkl')

data_df = pd.merge(data_df, userid_feedid_d2v_max_wd, on=['userid'],how='left',copy=False)


## user author embed -----------------------------------------------------------------

userid_authorid_d2v_max_wd = pd.read_pickle('./userid_authorid_d2v_b.pkl')

data_df = pd.merge(data_df, userid_authorid_d2v_max_wd, on=['userid'],how='left',copy=False)



del userid_feedid_d2v_max_wd, userid_authorid_d2v_max_wd






data_df[['bgm_song_id','bgm_singer_id']]=data_df[['bgm_song_id','bgm_singer_id']].fillna(0)
data_df[['bgm_song_id','bgm_singer_id']]=data_df[['bgm_song_id','bgm_singer_id']].fillna(0)

## 视频时长是秒，转换成毫秒，才能与play、stay做运算 -----------------------------------------------------

data_df['videoplayseconds'] *= 1000
## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
data_df['is_finish'] = (data_df['play'] >= 0.9*data_df['videoplayseconds']).astype('int8')
data_df['play_times'] = data_df['play'] / data_df['videoplayseconds']
play_cols = ['is_finish', 'play_times', 'play', 'stay']


# ctr ---------------------------------------------------------------------------------------
## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
FEA_COLUMN_LIST = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15
n_day = 5





for stat_cols in tqdm([
    ['userid'],
    ['feedid'],
    ['authorid'],
    # ['bgm_song_id'],
    # ['bgm_singer_id'],
    ['userid', 'authorid']
    # ['userid', 'bgm_song_id'],
    # ['userid', 'bgm_singer_id']
]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1

        tmp = data_df[((data_df['date_'] >= left) & (data_df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

        g = tmp.groupby(stat_cols)
        
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean').astype(np.float32)
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

        for x in play_cols[1:]:
            for stat in ['max', 'mean']:
                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

        for y in FEA_COLUMN_LIST[:4]:
            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum').astype(np.float32)
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean').astype(np.float32)
            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])

        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)

        del g, tmp

    data_df = data_df.merge(stat_df, on=stat_cols + ['date_'], how='left')

    del stat_df
    gc.collect()





## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

# 曝光 -----------------------------------------------------------
for f in tqdm(['userid', 'feedid', 'authorid']):

    data_df[f + '_count'] = data_df[f].map(data_df[f].value_counts())

# number of unique ------------------------------------------------------
for f1, f2 in tqdm([
    ['userid', 'feedid'],
    ['userid', 'authorid']
]):
    data_df['{}_in_{}_nunique'.format(f1, f2)] = data_df.groupby(f2)[f1].transform('nunique')
    data_df['{}_in_{}_nunique'.format(f2, f1)] = data_df.groupby(f1)[f2].transform('nunique')

# 偏好 ----------------------------------------------------------------
for f1, f2 in tqdm([
    ['userid', 'authorid']
]):
    data_df['{}_{}_count'.format(f1, f2)] = data_df.groupby([f1, f2])['date_'].transform('count')
    data_df['{}_in_{}_count_prop'.format(f1, f2)] = data_df['{}_{}_count'.format(f1, f2)] / (data_df[f2 + '_count'] + 1)
    data_df['{}_in_{}_count_prop'.format(f2, f1)] = data_df['{}_{}_count'.format(f1, f2)] / (data_df[f1 + '_count'] + 1)

# ----------------------------------------------------------------------------------
data_df['videoplayseconds_in_userid_mean'] = data_df.groupby('userid')['videoplayseconds'].transform('mean')
data_df['videoplayseconds_in_authorid_mean'] = data_df.groupby('authorid')['videoplayseconds'].transform('mean')
# 作者有多少视频数
data_df['feedid_in_authorid_nunique'] = data_df.groupby('authorid')['feedid'].transform('nunique')



# ## keyword w2v -------------------------------------------

keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']

keyword_data = pd.read_csv('./keyword_w2v_16.csv', index_col=0)
keyword_data.columns = [i+'w2v' for i in keyword_data.columns]


# # keyctr -------------------------------------------------------------------
# import pickle
# def load_pickle(input_file):
#     '''
#     读取pickle文件
#     :param pickle_path:
#     :param file_name:
#     :return:
#     '''
#     with open(str(input_file), 'rb') as f:
#         data = pickle.load(f)
#     return data

keyctr_data = pd.read_pickle('./keyword_ctr_final.pkl')

keyctr_data = keyctr_data.fillna(0)

data_df = data_df.merge(keyctr_data, how='left', on=['userid','feedid','date_'])

del keyctr_data

# keyword cate -----------------------------------------------------------------------
def process_keywords(colname):
    data = feed_info[colname].str.split(';')
    keyword_array = pd.DataFrame(np.zeros((feed_info.shape[0], 1)), columns=[colname]).astype(object) 
    for i in tqdm(range(data.shape[0])):
        x = feed_info.loc[i, colname]
        if x != np.nan and x != '':
            y = str(x).strip().split(";")
        else:
            y = []
        keyword_array.at[i,colname] = y
    res = pd.concat((feed_info['feedid'], keyword_array), axis=1)
    return res

manual_keyword_list_use  = process_keywords('manual_keyword_list')
manual_tag_list = process_keywords('manual_tag_list')
machine_keyword_list = process_keywords('machine_keyword_list')

# 也可以根据概率的分布，截取概率的threshold
def process_machine_tag_list(colname):
    # data = feed_info[colname].str.split(';')
    keyword_array = pd.DataFrame(np.zeros((feed_info.shape[0], 1)), columns=[colname]).astype(object) 
    for i in tqdm(range(feed_info.shape[0])):
        x = feed_info.loc[i, colname]
        if x != np.nan and x != '' and x !=' ' and x != ';':
            y = [i.split(' ') for i in str(x).strip().split(";")]
            if y[0][0] == 'nan':
                y=[]
            else:
                y = pd.DataFrame(y).astype({0: np.int, 1: np.float32}).sort_values(by=1, ascending=False).reset_index(drop=True)

                if y[1][0]<0.5 and y[1][0]>0.25:
                    y=[y[0][1]]
                elif y[1][0]>=0.5:
                    y = y[y[1]>0.5][0].tolist()
                else:
                    y=[]
        else:
            y = []
        keyword_array.at[i,colname] = y
    res = pd.concat((feed_info['feedid'], keyword_array), axis=1)
    return res

machine_tag_list= process_machine_tag_list('machine_tag_list')





#d2v ----------------------------------------------------------------------------------
# text doc
text_feature = ['description','ocr','asr']

text_data = pd.read_csv('./all_text_data_20v.csv', index_col=0)

text_data = text_data.drop('feedid', axis=1)
text_data.columns = [i+'_d2v' for i in text_data.columns]



# # ## feed embedding




# #------------------


# # ## merge with train and test data


# merge keyword

feed_info_use = pd.concat([feed_info_use, keyword_data, text_data], axis=1)
feed_info_use = pd.concat([feed_info_use, manual_keyword_list_use.iloc[:,1], manual_tag_list.iloc[:,1], machine_keyword_list.iloc[:,1], machine_tag_list.iloc[:,1]], axis=1)



# # merge embedding
feed_embed_processed = pd.read_csv('./feed_embeddings_use_rd_32.csv', index_col=0)
# feed_embed_processed.columns = ['feed_embed'+str(i) for i in range(feed_embed_processed.shape[1]-1)] + ['feedid']
# feed_embed_feature = ['feed_embed'+str(i) for i in range(feed_embed_processed.shape[1])]
data_df = pd.merge(data_df, feed_embed_processed, on=['feedid'],how='left',copy=False)



data_df = pd.merge(data_df, feed_info_use[['feedid']+keyword_data.columns.tolist() +text_data.columns.tolist()+ keyword_feature ], on=['feedid'],how='left',copy=False)



# ### use MultiLabelBinarizer

keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']

keyword_feature_binary = data_df[keyword_feature]



from sklearn.preprocessing import MultiLabelBinarizer
def MultiLabelBinarizer_process(data, target):
    mlb = MultiLabelBinarizer(sparse_output=True)
    encoded = mlb.fit_transform(data[target])
    return encoded

manual_keyword_sparse = MultiLabelBinarizer_process(keyword_feature_binary, 'manual_keyword_list')
machine_keyword_sparse = MultiLabelBinarizer_process(keyword_feature_binary, 'machine_keyword_list')
manual_tag_sparse = MultiLabelBinarizer_process(keyword_feature_binary, 'manual_tag_list')
machine_tag_sparse = MultiLabelBinarizer_process(keyword_feature_binary, 'machine_tag_list')




# description_sparse


# # ### filter out low frequency word

cut_num = 500000  # 1000 10000 70000
print(sum(pd.Series(manual_keyword_sparse.sum(axis=0).tolist()[0]) > 200000),
     sum(pd.Series(machine_keyword_sparse.sum(axis=0).tolist()[0]) > 320000),
     sum(pd.Series(manual_tag_sparse.sum(axis=0).tolist()[0]) > 500000),
     sum(pd.Series(machine_tag_sparse.sum(axis=0).tolist()[0]) > 400000)
     )

manual_keyword_sparse = manual_keyword_sparse[:,(pd.Series(manual_keyword_sparse.sum(axis=0).tolist()[0]) > 200000).values]
machine_keyword_sparse = machine_keyword_sparse[:,(pd.Series(machine_keyword_sparse.sum(axis=0).tolist()[0]) > 320000).values]
manual_tag_sparse = manual_tag_sparse[:,(pd.Series(manual_tag_sparse.sum(axis=0).tolist()[0]) > 500000).values]
machine_tag_sparse = machine_tag_sparse[:,(pd.Series(machine_tag_sparse.sum(axis=0).tolist()[0]) > 400000).values]




manual_keyword_sparse_df = pd.DataFrame(manual_keyword_sparse.toarray(), columns=["manual_keyword_sparse_"+str(i) for i in range(manual_keyword_sparse.shape[1])]).astype(np.float32)
machine_keyword_sparse_df = pd.DataFrame(machine_keyword_sparse.toarray(), columns=["machine_keyword_sparse_"+str(i) for i in range(machine_keyword_sparse.shape[1])]).astype(np.float32)
manual_tag_sparse_df = pd.DataFrame(manual_tag_sparse.toarray(), columns=["manual_tag_sparse_"+str(i) for i in range(manual_tag_sparse.shape[1])]).astype(np.float32)
machine_tag_sparse_df = pd.DataFrame(machine_tag_sparse.toarray(), columns=["machine_tag_sparse_"+str(i) for i in range(machine_tag_sparse.shape[1])]).astype(np.float32)



data_df = pd.concat((data_df, 
                    manual_keyword_sparse_df,
                    machine_keyword_sparse_df,
                    manual_tag_sparse_df,
                    machine_tag_sparse_df),axis=1)



del manual_keyword_sparse_df, machine_keyword_sparse_df, manual_tag_sparse_df, machine_tag_sparse_df


data_df = data_df.drop(keyword_feature, axis=1)


# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']

play_cols = ['is_finish', 'play_times', 'play', 'stay']
drop_columns = ['play', 'stay']
keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']
text_feature = ['description','ocr','asr']

data_df = data_df.drop(play_cols, axis=1)

train_use_all = data_df.iloc[0:user_action_df.shape[0],:]
test_use_all = data_df.iloc[user_action_df.shape[0]:,:]
test_use_all['date_'] = 15 # 预测15号



train_use_all.shape, test_use_all.shape


# train_use_all.to_csv('./train_userfeed_keyword.csv')
# test_use_all.to_csv('./test_userfeed_keyword.csv')


# train_use = train_use.drop('date_', axis=1)
# test_use = test_use.drop('date_', axis=1)


ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
training_features = [i for i in data_df.columns if i not in FEA_COLUMN_LIST+['date_']]

categorical_feature = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id'] + [
    "manual_keyword_sparse_"+str(i) for i in range(manual_keyword_sparse.shape[1])] + [
    "machine_keyword_sparse_"+str(i) for i in range(machine_keyword_sparse.shape[1])] + [
    "manual_tag_sparse_"+str(i) for i in range(manual_tag_sparse.shape[1])] + [
    "machine_tag_sparse_"+str(i) for i in range(machine_tag_sparse.shape[1])] 


# categorical_feature = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id']
categorical_feature = [i for i, e in enumerate(data_df.columns.tolist()) if e in categorical_feature]



# # train

trn_x = train_use_all[train_use_all['date_'] < 14].reset_index(drop=True)
val_x = train_use_all[train_use_all['date_'] == 14].reset_index(drop=True)

from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc
import time
import random
import pickle



def save_pickle(data, file_path):

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


# callback
##################### 线下验证 #####################
seed_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 2049, 2050]
for item in range(10):

    seed = seed_list[item]

    uauc_list = []
    r_list = []
    for y in ['read_comment','like', 'click_avatar', 'forward']:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=5000,
            num_leaves=63,
            subsample=0.8,
            #subsample_freq=5,
            colsample_bytree=0.8,
            random_state=seed,
            metric='None'
            
        )
        clf.fit(
            trn_x[training_features], trn_x[y],
            eval_set=[(val_x[training_features], val_x[y])],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=50
        )
        val_x[y + '_score'] = clf.predict_proba(val_x[training_features])[:, 1]
        val_uauc = uAUC(val_x[y].tolist(), val_x[y + '_score'].tolist(), val_x['userid'].tolist())
        uauc_list.append(val_uauc)
        print(val_uauc)
        r_list.append(clf.best_iteration_)
        print('runtime: {} mins\n'.format((time.time() - t)/60))
        
    weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
    print(uauc_list)
    print("weighted_uauc:",weighted_uauc)



    test_use_all[ACTION_LIST] = 0.0
    import random

    ##################### 全量训练 10 次 #####################

    r_dict = dict(zip(['read_comment','like', 'click_avatar', 'forward'], r_list))
    for y in ['read_comment','like', 'click_avatar', 'forward']:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=r_dict[y],
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed
        )

        clf.fit(
            train_use_all[training_features], train_use_all[y],
            eval_set=[(train_use_all[training_features], train_use_all[y])],
            early_stopping_rounds=r_dict[y],
            verbose=100
        )

        save_pickle(clf, '../../data/model/lgb_{}_{}.m'.format(y,item))

    #     test_use_all[y] = clf.predict_proba(test_use_all[training_features])[:, 1]
    #     print('runtime: {}\n'.format(time.time() - t))
    # test_use_all[['userid', 'feedid'] + ['read_comment','like', 'click_avatar', 'forward']].to_csv(
    #     'testb_fold_%d_sub63_%.6f_%.6f_%.6f_%.6f_%.6f.csv' % (item, weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3]),
    #     index=False
    # )


