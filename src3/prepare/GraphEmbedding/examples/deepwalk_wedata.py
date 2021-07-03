
import numpy as np
import pandas as pd

from ge.classify import read_node_label, Classifier
from ge import DeepWalk,LINE,Struc2Vec,Node2Vec,SDNE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import pickle
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

user_action = pd.read_csv('../../../../data/wedata/user_action.csv')
max_user_id = max(user_action['userid'])
user_action['feedid'] = user_action['feedid']+max_user_id
user_action['userid'] =user_action['userid'].astype('str')
user_action['feedid'] =user_action['feedid'].astype('str')
userlist = set(user_action.userid.unique())
G=nx.from_pandas_edgelist(user_action, 'userid', 'feedid')


model = DeepWalk(G, walk_length=10, num_walks=80, workers=4)
model.train(embed_size = 32,window_size=5, iter=3)
embeddings = model.get_embeddings()

ids = set(embeddings.keys())&userlist
deepwalk_emb = {}
for id  in ids:    
    deepwalk_emb[id] = embeddings[id]
save_pickle(deepwalk_emb,'../../../train/data/wechat_algo_data1/deepwalk_emb.pkl')


train = pd.read_csv('../../../../data/wedata/user_action.csv')
tmp = train[['userid','feedid']].drop_duplicates('userid')
walk_emb = load_pickle('../../../train/data/wechat_algo_data1/deepwalk_emb.pkl')
emb = pd.DataFrame(walk_emb).T
emb['userid'] = emb.index
emb['userid'] =emb['userid'].astype('int')
emb.columns = ['graph_walk_{}_emb'.format(i) for i in range(32)]+['userid']
tmp = tmp.merge(emb,on = 'userid',how = 'left')
tmp = tmp.drop('feedid',axis = 1)
save_pickle(tmp,'../../../train/data/wechat_algo_data1/graph_walk_emb_32.pkl')
save_pickle(tmp,'../../../../data/wedata/graph_walk_emb_32.pkl')
