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


with open('embedding_res512.json', 'r') as f:
    data = json.load(f)

df_res=pd.read_csv('df_result512.csv')

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
    search_embedding = model.encode(query)
    cosine_scores = cos_sim(search_embedding, X).tolist()[0]
    I_all = list(map(cosine_scores.index, heapq.nlargest(3, cosine_scores)))
    D_all = heapq.nlargest(3, cosine_scores)
    print(D_all, I_all)
    prompt_list = []
    for i in I_all:
        tmpid = faiss_embedding_list[i]
        prompt_list.append({"input": df_res[df_res['embed_id'] == tmpid]['问题'].values[0],
                            "label": df_res[df_res['embed_id'] == tmpid]['答案'].values[0]})
    print(prompt_list[0]['input'])

    if D_all[0] >= 0.5:  # bert embedding 阈值90   openai 阈值0.8
        answer = prompt_list[0]['label']
        if "A1" in answer:
            ans_list = []
            for i in answer.split("A")[1:]:
                ans_list.append(i.split("：")[1])
            answer = random.choice(ans_list)
    else:
        answer = "没有检索到该问题的答案TAT\n你可以问我：\n1.{0}\n2.{1}\n3.{2}\n...".format(prompt_list[0]['input'],
                                                                           prompt_list[1]['input'],
                                                                           prompt_list[2]['input'])
    return answer










# query="你们这个产品好像比较贵"
# search_embedding = model.encode(query)
# cosine_scores = cos_sim(search_embedding, X).tolist()[0]
# I_all=list(map(cosine_scores.index, heapq.nlargest(3,cosine_scores)))
# D_all=heapq.nlargest(3,cosine_scores)
# prompt_list=[]
# for i in I_all:
#     tmpid = faiss_embedding_list[i]
#     prompt_list.append({"input": df_res[df_res['embed_id'] == tmpid]['问题'].values[0],
#                         "label": df_res[df_res['embed_id'] == tmpid]['答案'].values[0]})
# print(prompt_list[0]['input'])
#
# if D_all[0]>= 0:  # bert embedding 阈值90   openai 阈值0.8
#     answer = prompt_list[0]['label']
#     if "A1" in answer:
#         ans_list = []
#         for i in answer.split("A")[1:]:
#             ans_list.append(i.split("：")[1])
#         answer = random.choice(ans_list)
# else:
#     answer = "没有检索到该问题的答案TAT\n你可以问我：\n1.{0}\n2.{1}\n3.{2}\n...".format(prompt_list[0]['input'], prompt_list[1]['input'],
#                                                                        prompt_list[2]['input'])

