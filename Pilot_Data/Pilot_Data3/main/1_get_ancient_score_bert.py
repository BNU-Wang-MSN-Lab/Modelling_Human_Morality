import re
import json
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('../ancient_chinese_bert-pt31')
model = BertModel.from_pretrained('../ancient_chinese_bert-pt31').cuda()
word2id = {line.strip(): i for i, line in enumerate(open('../ancient_chinese_bert-pt31/vocab.txt'))}


def get_embedding_mean(x):
    input_ids = torch.tensor(tokenizer.encode(x[:254])).unsqueeze(0)
    outputs = model(input_ids.cuda(), output_hidden_states=True)
    
    last_hidden_states = outputs[0]
    embeddings = last_hidden_states[0].cpu().detach().numpy()
    embeddings = np.mean(embeddings[:], axis=0)
    return embeddings


def cosine_distance(a, b):
    r = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    return r


fn = '善恶行为+维度_古汉语.xlsx'

data = pd.read_excel(fn, sheet_name=['Evil_features', 'Good_features'])
s = {}
for k in data:
    s[k] = {}
    for i, j in data[k].iterrows():
        if (k[0] == 'E' and j['是否为40dim（全集）'] == 1) or (k[0] == 'G' and j['是否为40dim（全集）'] == 1):
            s[k][j['维度名称']] = get_embedding_mean(f"{j['古汉语描述']}")

desc = json.loads(open('data_ancient_description_gpt4120250414_1234_1014.json').read())

scores = {}
data = pd.read_excel(fn, sheet_name=['恶行描述', '善行描述'])
for k in data:
    for _, i in data[k].iterrows():
        print(i['行为'])
        
        if k[0] == '恶':
            kk = 'Evil_features'
        else:
            kk = 'Good_features'
        sss = s[kk]

        q = f'{desc[i["行为"]]}'
        
        e = get_embedding_mean(q)
        
        scores[i['行为']] = {}
        for ss in sorted(list(sss.keys())):
            
            r = cosine_distance(e, s[kk][ss])
            r = float(r)
            
            scores[i['行为']][ss] = r

open('data_ancient_scores_bnubert_1011.json', 'w').write(json.dumps(scores, ensure_ascii=False, indent=2))
