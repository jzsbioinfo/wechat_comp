# -*- coding: utf-8 -*-

# create file

# userid_feedid_d2v_b.pkl
# userid_authorid_d2v_b.pkl
# keyword_w2v_16.csv
# all_text_data_20v.csv
# keyword_ctr_final.pkl
# feed_embeddings_use_rd_32.csv



from tqdm import tqdm

import gc
import numpy as np

from evaluation import uAUC
import pandas as pd

import os

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import utils
from gensim.models import Word2Vec


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

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader,Dataset,TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler




# 存储数据的根目录
ROOT_PATH = "../../data/wedata"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集 use test_b
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
#test_a = reduce_mem(pd.read_csv(TEST_FILE_a))
test = reduce_mem(pd.read_csv(TEST_FILE))
test['date_'] = 15


feed_use_feature = ['feedid','authorid','videoplayseconds','bgm_song_id', 'bgm_singer_id']

feed_info_use = feed_info[feed_use_feature] # feed_info

data_df = pd.concat([user_action_df, test], axis=0, ignore_index=True)
data_df = pd.merge(data_df, feed_info_use, on=['feedid'],how='left',copy=False)

data_df = data_df[['userid','feedid','authorid']]

# userid_feedid ----------------------------------------------

user_feed_list = data_df.groupby('userid').apply(lambda x: [str(i) for i in list(x['feedid'])]).reset_index()
colname = 'userid_feedid'
use_doc = user_feed_list[0].tolist()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(use_doc)]

model_dbow = Doc2Vec( dm=0, vector_size=16, negative=5, hs=0, min_count=5, sample = 0, workers=8,
                    window = 900)
model_dbow.build_vocab(documents)

for epoch in range(50):
    model_dbow.train(utils.shuffle(documents), total_examples=len(documents), epochs=1)
    model_dbow.alpha -= 0.0002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs
    regressors = [model.infer_vector(doc.words, steps=50) for doc in sents]
    return regressors
regressors = vec_for_learning(model_dbow, documents)

userid_feedid_d2v =  pd.DataFrame(np.array(regressors), columns=[colname+str(i) for i in  range(16)])
userid_feedid_d2v['userid'] = user_feed_list['userid'].values

userid_feedid_d2v.to_pickle('../../data/wedata/userid_feedid_d2v_b.pkl')

# userid_authorid ----------------------------------------------

user_author_list = data_df.groupby('userid').apply(lambda x: [str(i) for i in list(x['authorid'])]).reset_index()
colname = 'userid_authorid'
use_doc = user_author_list[0].tolist()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(use_doc)]

model_dbow = Doc2Vec( dm=0, vector_size=16, negative=5, hs=0, min_count=5, sample = 0, workers=8,
                    window = 900)
model_dbow.build_vocab(documents)

for epoch in range(50):
    model_dbow.train(utils.shuffle(documents), total_examples=len(documents), epochs=1)
    model_dbow.alpha -= 0.0002
    model_dbow.min_alpha = model_dbow.alpha
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs
    regressors = [model.infer_vector(doc.words, steps=50) for doc in sents]
    return regressors
regressors = vec_for_learning(model_dbow, documents)

userid_authorid_d2v =  pd.DataFrame(np.array(regressors), columns=[colname+str(i) for i in  range(16)])
userid_authorid_d2v['userid'] = user_author_list['userid'].values

userid_authorid_d2v.to_pickle('../../data/wedata/userid_authorid_d2v_b.pkl')


# keyword and tag w2v -----------------------------------------------------

keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']
keyword_data = feed_info[keyword_feature]

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
                y = pd.DataFrame(y).astype({0: str, 1: np.float32}).sort_values(by=1, ascending=False).reset_index(drop=True)

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

min_count_list = [2,5,2,5]
window_list = [10,10,10,10]

data_list = [manual_keyword_list_use,
        manual_tag_list,
        machine_keyword_list,
        machine_tag_list]

def get_w2v_dataframe(data, model, len_vec, colname):

    def word_2_vec(data, model, len_vec):
        all_word = list(model.wv.vocab.keys())
        res = np.zeros(len_vec)
        n=0
        for item in data:
            if item == 'nan':
                return res
            elif item not in all_word:
                continue 
            else:
                n+=1
                temp = model.wv[item]
                res += temp
        if n==0:
            return res
        else:
            return res/n

    def data_to_frame(data, len_vec, colname):

        keyword_array = pd.DataFrame(np.zeros((data.shape[0], len_vec)), columns=[colname+f'_{i}' for i in range(len_vec)]).astype(np.float32) 
        for i in tqdm(range(data.shape[0])):
            x = data.loc[i, 'w2v']
            keyword_array.iloc[i,:] = x
        return keyword_array   
    
    data['w2v'] = data[colname].apply(word_2_vec, args=(model, len_vec))
    keyword_array = data_to_frame(data, len_vec, colname)
    
    return keyword_array


index=0

data = data_list[index]
len_vec = 16
colname = keyword_feature[index]
min_count=min_count_list[index]
window=window_list[index]

use_sentences = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]][colname].tolist()

model = Word2Vec(# sentences=common_texts, 
                 size=len_vec,  #  Dimensionality of the word vectors.
                 window=window,  # Maximum distance between the current and predicted word within a sentence.
                 min_count=min_count,  # Ignores all words with total frequency lower than this.
                 workers=8,  # Use these many worker threads
                 sg = 1, # Training algorithm: 1 for skip-gram; otherwise CBOW.
                 seed=1,
                 compute_loss=True,
                 #sample=6e-5,   # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
                 alpha=0.025,  # The initial learning rate.
                 min_alpha=0.0001,  # Learning rate will linearly drop to min_alpha as training progresses.
                 negative=20  # If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
                )

from time import time
t = time()

model.build_vocab(sentences=use_sentences, progress_per=10000)
model.train(sentences=use_sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)

manual_keyword_list_use_w2v = get_w2v_dataframe(data, model, len_vec, colname)


index=1

data = data_list[index]
len_vec = 16
colname = keyword_feature[index]
min_count=min_count_list[index]
window=window_list[index]

use_sentences = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]][colname].tolist()

model = Word2Vec(# sentences=common_texts, 
                 size=len_vec,  #  Dimensionality of the word vectors.
                 window=window,  # Maximum distance between the current and predicted word within a sentence.
                 min_count=min_count,  # Ignores all words with total frequency lower than this.
                 workers=8,  # Use these many worker threads
                 sg = 1, # Training algorithm: 1 for skip-gram; otherwise CBOW.
                 seed=1,
                 compute_loss=True,
                 #sample=6e-5,   # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
                 alpha=0.025,  # The initial learning rate.
                 min_alpha=0.0001,  # Learning rate will linearly drop to min_alpha as training progresses.
                 negative=20  # If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
                )

from time import time
t = time()

model.build_vocab(sentences=use_sentences, progress_per=10000)
model.train(sentences=use_sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)

manual_tag_list_w2v = get_w2v_dataframe(data, model, len_vec, colname)


index=2

data = data_list[index]
len_vec = 16
colname = keyword_feature[index]
min_count=min_count_list[index]
window=window_list[index]

use_sentences = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]][colname].tolist()

model = Word2Vec(# sentences=common_texts, 
                 size=len_vec,  #  Dimensionality of the word vectors.
                 window=window,  # Maximum distance between the current and predicted word within a sentence.
                 min_count=min_count,  # Ignores all words with total frequency lower than this.
                 workers=8,  # Use these many worker threads
                 sg = 1, # Training algorithm: 1 for skip-gram; otherwise CBOW.
                 seed=1,
                 compute_loss=True,
                 #sample=6e-5,   # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
                 alpha=0.025,  # The initial learning rate.
                 min_alpha=0.0001,  # Learning rate will linearly drop to min_alpha as training progresses.
                 negative=20  # If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
                )

from time import time
t = time()

model.build_vocab(sentences=use_sentences, progress_per=10000)
model.train(sentences=use_sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)

machine_keyword_list_w2v = get_w2v_dataframe(data, model, len_vec, colname)


index=3

data = data_list[index]
len_vec = 16
colname = keyword_feature[index]
min_count=min_count_list[index]
window=window_list[index]

use_sentences = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]][colname].tolist()

model = Word2Vec(# sentences=common_texts, 
                 size=len_vec,  #  Dimensionality of the word vectors.
                 window=window,  # Maximum distance between the current and predicted word within a sentence.
                 min_count=min_count,  # Ignores all words with total frequency lower than this.
                 workers=8,  # Use these many worker threads
                 sg = 1, # Training algorithm: 1 for skip-gram; otherwise CBOW.
                 seed=1,
                 compute_loss=True,
                 #sample=6e-5,   # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
                 alpha=0.025,  # The initial learning rate.
                 min_alpha=0.0001,  # Learning rate will linearly drop to min_alpha as training progresses.
                 negative=20  # If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
                )

from time import time
t = time()

model.build_vocab(sentences=use_sentences, progress_per=10000)
model.train(sentences=use_sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)

machine_tag_list_w2v = get_w2v_dataframe(data, model, len_vec, colname)

all_w2v = pd.concat([manual_keyword_list_use_w2v, manual_tag_list_w2v, machine_keyword_list_w2v, machine_tag_list_w2v], ignore_index=False, axis=1)
all_w2v.to_csv('../../data/wedata/keyword_w2v_16.csv')


# text feature doc2v -------------------------------------------------------------------
char_feature = ['description','ocr','asr']
char_data = feed_info[char_feature]

def process_text_feature(colname):
    # data = feed_info[colname].str.split(' ')
    keyword_array = pd.DataFrame(np.zeros((feed_info.shape[0], 1)), columns=[colname]).astype(object) 
    for i in tqdm(range(feed_info.shape[0])):
        x = feed_info.loc[i, colname]
        if x != np.nan and x != '':
            y = str(x).strip().split(" ")
            #y = [i for i in y if i in use_word]
        else:
            y = []
        keyword_array.at[i,colname] = y
    res = pd.concat((feed_info['feedid'], keyword_array), axis=1)
    return res

description_list  = process_text_feature('description')
ocr_list = process_text_feature('ocr')
asr_list = process_text_feature('asr')


data = ocr_list
colname = 'ocr'
use_doc = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]][colname].tolist()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(use_doc)]

model_dbow = Doc2Vec( dm=0, vector_size=20, negative=5, hs=0, min_count=5, sample = 0, workers=8)
model_dbow.build_vocab(documents)

for epoch in range(50):
    model_dbow.train(utils.shuffle(documents), total_examples=len(documents), epochs=1)
    model_dbow.alpha -= 0.0002
    model_dbow.min_alpha = model_dbow.alpha
    
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs
    regressors = [model.infer_vector(doc.words, steps=20) for doc in sents]
    return regressors
regressors = vec_for_learning(model_dbow, documents)

ocr_data =  pd.DataFrame(np.array(regressors), columns=[colname+str(i) for i in  range(20)])
temp_data = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]]
ocr_data['feedid'] = temp_data['feedid']
ocr_data = ocr_data.merge(ocr_list, how='right',on='feedid').drop('ocr', axis=1).fillna(0.)



data = asr_list
colname = 'asr'
use_doc = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]][colname].tolist()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(use_doc)]

model_dbow = Doc2Vec( dm=0, vector_size=20, negative=5, hs=0, min_count=5, sample = 0, workers=8)
model_dbow.build_vocab(documents)

for epoch in range(50):
    model_dbow.train(utils.shuffle(documents), total_examples=len(documents), epochs=1)
    model_dbow.alpha -= 0.0002
    model_dbow.min_alpha = model_dbow.alpha
    
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs
    regressors = [model.infer_vector(doc.words, steps=20) for doc in sents]
    return regressors
regressors = vec_for_learning(model_dbow, documents)


asr_data =  pd.DataFrame(np.array(regressors), columns=[colname+str(i) for i in  range(20)])
temp_data = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]]
asr_data['feedid'] = temp_data['feedid']
asr_data = asr_data.merge(asr_list, how='right',on='feedid').drop('asr', axis=1).fillna(0.)


data = description_list
colname = 'description'
use_doc = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]][colname].tolist()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(use_doc)]

model_dbow = Doc2Vec( dm=0, vector_size=20, negative=5, hs=0, min_count=5, sample = 0, workers=8)
model_dbow.build_vocab(documents)

for epoch in range(50):
    model_dbow.train(utils.shuffle(documents), total_examples=len(documents), epochs=1)
    model_dbow.alpha -= 0.0002
    model_dbow.min_alpha = model_dbow.alpha
    
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs
    regressors = [model.infer_vector(doc.words, steps=20) for doc in sents]
    return regressors
regressors = vec_for_learning(model_dbow, documents)

description_data =  pd.DataFrame(np.array(regressors), columns=[colname+str(i) for i in  range(20)])
temp_data = data[[data[colname][i] != ['nan'] for i in range(data.shape[0])]]
description_data['feedid'] = temp_data['feedid']
description_data = description_data.merge(description_list, how='right',on='feedid').drop('description', axis=1).fillna(0.)


all_text_data = pd.concat([description_data, ocr_data.iloc[:,:-1], asr_data.iloc[:,:-1]], axis=1)

all_text_data.to_csv('../../data/wedata/all_text_data_20v.csv')


# keyword and tag ctr 转化率 ---------------------------------------------------------------

feed_use_feature = ['feedid','authorid','videoplayseconds','bgm_song_id', 'bgm_singer_id']
keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list']

feed_use_feature = ['feedid','authorid','videoplayseconds']
feed_info_use = feed_info[feed_use_feature+keyword_feature] # feed_info

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
                y = pd.DataFrame(y).astype({0: str, 1: np.float32}).sort_values(by=1, ascending=False).reset_index(drop=True)

                if y[1][0]<0.5 and y[1][0]>0.25:
                    y=[y[0][1]]
                elif y[1][0]>=0.5:
                    y = y[y[1]>0.5][0].tolist()
                else:
                    y=[]
        else:
            y = []
            
        y=';'.join(y)
        keyword_array.at[i,colname] = y
    res = pd.concat((feed_info['feedid'], keyword_array), axis=1)
    return res
machine_tag_list= process_machine_tag_list('machine_tag_list')




feed_info_use = pd.concat([feed_info_use, machine_tag_list.iloc[:,1]], axis=1)

data_df = pd.concat([user_action_df, test], axis=0, ignore_index=True)
data_df = pd.merge(data_df, feed_info_use, on=['feedid'],how='left',copy=False)

data_df.replace('','-1',inplace=True)

data_df = data_df[['userid','feedid','date_']+ACTION_LIST+keyword_feature]

data_df[keyword_feature] = data_df[keyword_feature].fillna('-1')




def myPivot(feat_feild,index,values, aggfunc):
    t = feat_feild.pivot_table(index=index, values=values, aggfunc=aggfunc)
    columns = ['_'.join(index)+ '_' + fun_name + '_' + v for fun_name,v in t.columns ]
    # print(columns)
    t.columns = columns
    t = t.reset_index()
    return t, columns


def getTagrate(x,c,user_tag_item):
    userid = x.userid
    if x[c]=='-1':
        return 0
    
    tags = x[c].split(';')
    score=0
    for tag in tags:
            tag = float(tag)
            if (userid,tag) in user_tag_item :
                score += user_tag_item[(userid,tag)]

    return score

max_day = 15
n_day = 5

for target_day in tqdm(range(2,max_day+1)):
    
    left, right = max(target_day - n_day, 1), target_day - 1

    feat_feild = data_df[((data_df['date_'] >= left) & (data_df['date_'] <= right))].reset_index(drop=True)
    feat_feild['date_'] = target_day
    
    label_f = data_df[data_df.date_==target_day]

    for c in keyword_feature:
        split_ = feat_feild[c].str.split(';', expand=True)  # 切分当前特征列
        split_.columns = [c + str(x) for x in split_.columns]
        split_ = split_.astype(np.float32)
        feat_feild_ = pd.concat([feat_feild, split_], axis=1)  #拼接
        t = feat_feild_[['userid', 'feedid'] +
                        split_.columns.tolist()].set_index(
                            ['userid', 'feedid'])
        t = t.stack(0, dropna=True).reset_index()
        t = t.merge(feat_feild[['userid', 'feedid', ]+ACTION_LIST],
                    how='left',
                    on=['userid', 'feedid'])[['userid', 'feedid', 0,] + ACTION_LIST]
        t.columns = ['userid', 'feedid', c + '_id', ] + ACTION_LIST

        user_transform_rate, cols = myPivot(t, index=['userid', c + '_id'], values=ACTION_LIST, aggfunc=['mean', 'sum'])
        
        user_transform_rate = user_transform_rate.set_index(['userid', c + '_id'])
        
        user_transform_rate = user_transform_rate[cols].to_dict()

        for col in cols:
            # print('====1====')
            user_tag_item = user_transform_rate[col]
            label_f[col] = label_f.apply(lambda x:getTagrate(x,c,user_tag_item), axis=1)
    if target_day==2:
        keyword_ctr = label_f
    else:
        keyword_ctr = pd.concat([keyword_ctr,label_f], axis=0)


del feat_feild, label_f, data_df

keyword_ctr[[i for i in keyword_ctr.columns if i not in ['userid','feedid','date_']+keyword_feature+ACTION_LIST]] = keyword_ctr[[i for i in keyword_ctr.columns if i not in ['userid','feedid','date_']+keyword_feature+ACTION_LIST]].astype(np.float32)

manual_keyword_list_columns = ['userid_manual_keyword_list_id_mean_click_avatar',
       'userid_manual_keyword_list_id_mean_forward',
       'userid_manual_keyword_list_id_mean_like',
       'userid_manual_keyword_list_id_mean_read_comment',
       'userid_manual_keyword_list_id_sum_click_avatar',
       'userid_manual_keyword_list_id_sum_forward',
       'userid_manual_keyword_list_id_sum_like',
       'userid_manual_keyword_list_id_sum_read_comment']
machine_keyword_list_columns = ['userid_machine_keyword_list_id_mean_click_avatar',
       'userid_machine_keyword_list_id_mean_forward',
       'userid_machine_keyword_list_id_mean_like',
       'userid_machine_keyword_list_id_mean_read_comment',
       'userid_machine_keyword_list_id_sum_click_avatar',
       'userid_machine_keyword_list_id_sum_forward',
       'userid_machine_keyword_list_id_sum_like',
       'userid_machine_keyword_list_id_sum_read_comment',]
manual_tag_list_columns =['userid_manual_tag_list_id_mean_click_avatar',
       'userid_manual_tag_list_id_mean_forward',
       'userid_manual_tag_list_id_mean_like',
       'userid_manual_tag_list_id_mean_read_comment',
       'userid_manual_tag_list_id_sum_click_avatar',
       'userid_manual_tag_list_id_sum_forward',
       'userid_manual_tag_list_id_sum_like',
       'userid_manual_tag_list_id_sum_read_comment']
machine_tag_list_columns = ['userid_machine_tag_list_id_mean_click_avatar',
       'userid_machine_tag_list_id_mean_forward',
       'userid_machine_tag_list_id_mean_like',
       'userid_machine_tag_list_id_mean_read_comment',
       'userid_machine_tag_list_id_sum_click_avatar',
       'userid_machine_tag_list_id_sum_forward',
       'userid_machine_tag_list_id_sum_like',
       'userid_machine_tag_list_id_sum_read_comment']

temp = keyword_ctr.loc[(keyword_ctr.manual_keyword_list=='-1') & (keyword_ctr.machine_keyword_list!='-1'), machine_keyword_list_columns]
keyword_ctr.loc[(keyword_ctr.manual_keyword_list=='-1') & (keyword_ctr.machine_keyword_list!='-1'), manual_keyword_list_columns] = temp.values
temp = keyword_ctr.loc[(keyword_ctr.machine_keyword_list=='-1') & (keyword_ctr.manual_keyword_list!='-1'), manual_keyword_list_columns]
keyword_ctr.loc[(keyword_ctr.machine_keyword_list=='-1') & (keyword_ctr.manual_keyword_list!='-1'), machine_keyword_list_columns] = temp.values
temp = keyword_ctr.loc[(keyword_ctr.manual_tag_list=='-1') & (keyword_ctr.machine_tag_list!='-1'), machine_tag_list_columns]
keyword_ctr.loc[(keyword_ctr.manual_tag_list=='-1') & (keyword_ctr.machine_tag_list!='-1'), manual_tag_list_columns] = temp.values
temp = keyword_ctr.loc[(keyword_ctr.machine_tag_list=='-1') & (keyword_ctr.manual_tag_list!='-1'), manual_tag_list_columns]
keyword_ctr.loc[(keyword_ctr.machine_tag_list=='-1') & (keyword_ctr.manual_tag_list!='-1'), machine_tag_list_columns] = temp.values

keyword_ctr.drop(keyword_feature+ACTION_LIST, axis=1, inplace=True)
keyword_ctr = keyword_ctr.drop_duplicates(["userid","feedid","date_"])
keyword_ctr.reset_index(drop=True).to_pickle('../../data/wedata/keyword_ctr_final.pkl')


# feed embed  auto-encoder ------------------------------------------------
def process_embed(data):
    feed_embed_array = np.zeros((data.shape[0], 512))
    for i in tqdm(range(data.shape[0])):
        x = data.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y
    res = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    res = pd.concat((data, res), axis=1)
    return res

feed_embeddings_use = process_embed(feed_embed)
feed_embeddings_use.drop('feed_embedding',axis=1,inplace=True)


data_use = np.array(feed_embeddings_use.iloc[:,1:].astype(np.float32))
data_use = MinMaxScaler().fit_transform(data_use)
 
 ###### The input data is converted into the dataset type accepted by the neural network, and the batch is set to 10
tensor_x=torch.from_numpy(data_use)
input_size = tensor_x.shape[1]


my_dataset=TensorDataset(tensor_x)
my_dataset_loader=DataLoader(my_dataset,batch_size=1000, shuffle=False)
#print(isinstance(my_dataset,Dataset))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
 
 ###### Define an autoencoder model
class autoencoder(nn.Module):
    def __init__(self, input_size):
        super(autoencoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32)
        )
        self.decoder=nn.Sequential(
            
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, input_size),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        encoder=self.encoder(x)
        decoder=self.decoder(encoder)
        return encoder,decoder

model=autoencoder(input_size).to(device)

criterion=nn.MSELoss()  # MAE



optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

class LRScheduler():
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


####### epoch is set to 300
from pytorchtools import EarlyStopping

####### Define optimization function
model=autoencoder(input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
lr_scheduler = LRScheduler(optimizer)
# model.apply(nn.init.xavier_uniform())
early_stopping = EarlyStopping(patience=10, verbose=True)
for epoch in range(1000):
    total_loss = 0
    for i, x in enumerate(my_dataset_loader):
        
        model.train()
        
        x = x[0].to(device)
        
        reduced, pred=model(x)
        loss=criterion(pred, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        
    lr_scheduler(total_loss)  
    #print(list(model.parameters())[7].data.cpu())
    if epoch % 10 == 0:
        print(f'epoch {epoch}, total_loss {total_loss}')

    early_stopping(total_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

###### Dimensionality reduction and visualization based on the trained model

for i, x in enumerate(my_dataset_loader):
    model.eval()
    x = x[0].to(device)
    reduced, pred = model(x)

    if i == 0:
        x_ = reduced.data.cpu().numpy()
        y_ = pred.data.cpu().numpy()
    else:
        x_ = np.concatenate((x_,reduced.data.cpu().numpy()), axis=0)
        y_ = np.concatenate((y_,pred.data.cpu().numpy()), axis=0)

feed_embeddings_use_rd = pd.DataFrame(x_)
feed_embeddings_use_rd.to_csv('../../data/wedata/feed_embeddings_use_rd_32.csv')