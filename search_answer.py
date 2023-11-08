import json
# import os
import pandas as pd
import faiss
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import heapq


#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")


with open('embedding_res1107.json', 'r') as f:
    data = json.load(f)

df_res=pd.read_csv('df_result1107.csv')
df_res['keywords']=df_res['keywords'].fillna("").apply(lambda x:x.split("、"))

faiss_embedding_list = df_res['embed_id'].to_list()
xb = []
for i in faiss_embedding_list:
    xb.append(data[i])

X = np.array(xb)
X = X.astype('float32')
dim = X.shape[1]
print(dim)
index_ip = faiss.IndexFlatIP(dim)
index_ip.add(X)


def keyword_match(query):
    """
    关键词匹配，返回匹配到的问题id和命中关键词个数
    :param query:
    :return: dict{序号:关键词命中数}
    """
    matchq={}
    for ind in df_res.index:
        keylist=df_res.loc[ind,'keywords']
        if keylist==[""]:
            continue
        for word in keylist:
            if word in query:
                if ind not in matchq.keys():
                    matchq[ind]=1
                else:
                    matchq[ind]+=1
    return matchq

# keyword_match("健管")


def get_answer_faiss(query):

    search_embedding = model.encode(query)

    D_all, I_all = index_ip.search(np.array([search_embedding]).astype('float32'), 3)
    print(D_all, I_all)

    prompt_list = []
    # print(data_all[data_all['embed_ids']==testid]['内容'])
    for i in I_all[0][:]:
        tmpid = faiss_embedding_list[i]
        prompt_list.append({"input":df_res[df_res['embed_id']==tmpid]['问题'].values[0], "label":df_res[df_res['embed_id']==tmpid]['答案'].values[0]})
    print(prompt_list[0]['input'])

    if D_all[0][0]>=0: # bert embedding 阈值90   openai 阈值0.8
        answer=prompt_list[0]['label']
        if "A1" in answer:
            ans_list = []
            for i in answer.split("A")[1:]:
                ans_list.append(i.split("：")[1])
            answer=random.choice(ans_list)
    else:
        answer="没有检索到该问题的答案TAT\n你可以问我：\n1.{0}\n2.{1}\n3.{2}\n...".format(prompt_list[0]['input'],prompt_list[1]['input'],prompt_list[2]['input'])
    return answer



def get_answer_cosine(query):
    """
    step1：相似度匹配
    step2：若相似度低于阈值，使用关键词召回
            若用关键词命中了多个问题，先按命中关键词个数排序，再按相似度匹配最相近的问题
    :param query:
    :return:
    """
    search_embedding = model.encode(query)
    cosine_scores = cos_sim(search_embedding, X).tolist()[0]
    I_all = list(map(cosine_scores.index, heapq.nlargest(3, cosine_scores)))
    D_all = heapq.nlargest(3, cosine_scores)
    print(D_all, I_all)
    prompt_list = []
    for i in I_all:
        tmpid = faiss_embedding_list[i]
        prompt_list.append({"input": df_res[df_res['embed_id'] == tmpid]['问题'].values[0],
                            "label": df_res[df_res['embed_id'] == tmpid]['解答'].values[0]})
    print(prompt_list[0]['input'])

    if D_all[0] >= 0.6:  #  openai 阈值0.8
        answer = prompt_list[0]['label']
        if "A1" in answer:
            ans_list = []
            for i in answer.split("A")[1:]:
                ans_list.append(i.split("：")[1])
            answer = random.choice(ans_list)
        return answer
    else:
        matchq=keyword_match(query)
        print(matchq)
        if len(matchq)==0:
            # 未命中关键词
            return "没有检索到该问题的答案TAT\n你可以问我：\n1.{0}\n2.{1}\n3.{2}\n...".format(prompt_list[0]['input'],
                                                                           prompt_list[1]['input'],
                                                                           prompt_list[2]['input'])

        k, v = np.array([a[0] for a in matchq.items()]), np.array([a[1] for a in matchq.items()])
        keys_of_max_val =k[v== v.max()] # 命中关键词数量最多的index 并列多个第一的全部输出
        if len(keys_of_max_val)==1:
            # 绝对命中数第一的，直接输出答案
            answer =df_res.loc[keys_of_max_val[0],'解答']
        else:
            # 有并列第一时，取相似度最高的一个输出答案
            maxid=I_all[0]
            maxscore=0
            for qid in keys_of_max_val:
                if cosine_scores[qid]>maxscore:
                    maxscore=cosine_scores[qid]
                    maxid=qid
            answer = df_res.loc[maxid, '解答']
        if "A1" in answer:
            ans_list = []
            for i in answer.split("A")[1:]:
                ans_list.append(i.split("：")[1])
            answer = random.choice(ans_list)
        return answer

