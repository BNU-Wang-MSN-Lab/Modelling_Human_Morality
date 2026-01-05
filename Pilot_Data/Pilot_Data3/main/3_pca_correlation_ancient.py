import re
import glob
import json
import numpy as np
import scipy as sp
import pandas as pd
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

np.random.seed(1234)

font_path = '../util/Arial Unicode.ttf'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['font.size'] = 12

# 需要排除行为
EXCLUDE_ACTIONS = {}

def get_dim(is_evil):
    df = pd.read_excel('现代中国人数据最终版/Morality_feature_beh_1011.xlsx', sheet_name=['Evil_features', 'Good_features'])
    s = {}
    for k in df:
        s[k] = {}
        for _, j in df[k].iterrows():
            # if (k[0] == 'E' and j['是否为40dim（全集）'] == 1) or (k[0] == 'G' and j['是否为40dim（全集）'] == 1):
            if (k[0] == 'E' and j['是否为30dim（子集）'] == 1) or (k[0] == 'G' and j['是否为30dim（子集）'] == 1):
                s[k][j['维度名称']] = {'维度定义': j['维度定义'], '负端例子': j['负端例子'], '正端例子': j['正端例子']}
    if is_evil:
        return s['Evil_features']
    else:
        return s['Good_features']

dims_all = {'恶': sorted(get_dim(True).keys()), '善': sorted(get_dim(False).keys())}


def get_data_model(fn):
    embeddings = {}
    scores = json.loads(open(fn).read())
    data = pd.read_excel('善行+恶行描述.xlsx', sheet_name=['恶行描述', '善行描述'])
    for k in data:
        for _, a in data[k].iterrows():
            e = []
            for d in dims_all[k[0]]:
                sss = scores[a['行为']][d]
                if type(sss) != str:
                    e.append(sss)
                    continue
                sss = re.sub(r'应当给予满分10分', '应当给予10分', sss)
                sss = re.sub(r'可以获得满分10分', '应当给予10分', sss)
                sss = re.sub(r'满分.?10分', '', sss)
                sss = re.sub(r'（10分）', '', sss)
                ii = []
                for i in sss.split('\n'):
                    if '整体评估' not in i and '综上所述' not in i and '综合来看' not in i and '结合以上' not in i and '评估结论' not in i \
                    and '综合上述因素' not in i and '评估分数' not in i and '综合考虑' not in i and '综合以上' not in i and '总结与分数' not in i \
                        and '综合评定' not in i:
                        if re.findall(r'^\d\.', i):
                            continue
                        if re.findall(r'^\-', i):
                            continue
                        if re.findall(r'^ +', i):
                            continue
                        if not re.findall(r'\d', i):
                            continue
                    if i:
                        ii.append(i)
                sss = '\n'.join(ii)
                ssss = re.findall(r'\*\*(\d+\.?\d*?)[分/]\*\*', sss)
                if not ssss:
                    ssss = re.findall(r'(\d+\.?\d*?)[分/]', sss)
                    if not ssss:
                        ssss = re.findall(r'\d+', sss)
                        if not ssss:
                            print(sss)
                            print(scores[a['行为']][d])
                e.append(min(float(i) for i in ssss[:1] + ssss[-1:]))
            e = np.array(e)
            if k not in embeddings:
                embeddings[k] = {}
            embeddings[k][a['行为']] = e
    return embeddings


embeddings_llm_ancient = get_data_model('data_ancient_scores_gpt4omini20240718_40dim_2345_1011.json')
embeddings_bert_ancient = get_data_model('data_ancient_scores_bnubert_1011.json')


def get_data_human(fn):
    dim_dict = {}
    d = pd.read_excel('现代中国人数据最终版/Morality_feature_beh_1011.xlsx', sheet_name=['Evil_features', 'Good_features'])
    for k in d:
        for _, i in d[k].iterrows():
            # 如果这里报错，是因为excel里面善行和恶行的“names of dimensions”有一个s的区别，需要先改一下excel。
            dim_dict[i['names of dimensions'].lower().strip()] = i['维度名称']
    dim_dict = {v: k for k, v in dim_dict.items()}
    embeddings = {}
    for fn in glob.glob(fn):
        if 'evil' in fn:
            k = '恶行描述'
        else:
            k = '善行描述'
        if k not in embeddings:
            embeddings[k] = {}
        df = pd.read_csv(fn)
        for _, i in df.iterrows():
            i = {a.lower(): i[a] for a in i.keys()}
            e = []
            for d in dims_all[k[0]]:
                e.append(i[dim_dict[d]])
            e = np.array(e)
            embeddings[k][i['unnamed: 0']] = e
    return embeddings


embeddings_human_modern = get_data_human('现代中国人数据最终版/*_chn_*dim_*beh_mean_scaled.csv')


def scale(y, axis=0):
    x = y.copy()
    x = (x - np.mean(x, axis=axis)) / np.std(x, axis=axis)
    x *= np.sqrt(x.shape[0] / (x.shape[0] - 1))
    return x


def varimax(Phi, gamma=1, q=100, tol=1e-8):
    sc = np.sqrt(np.sum(Phi**2, axis=1))
    Phi = Phi / sc[:, np.newaxis]
    p,k = Phi.shape
    R = eye(k)
    d=0
    for _ in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/(d_old + 1e-8) < tol: break
    z = dot(Phi, R)
    z = z * sc[:, np.newaxis]
    return z


def cosine(x0, x1):
    return x0 @ x1.T / np.linalg.norm(x0) / np.linalg.norm(x1)


def permutation_test_vectors(v0, v1, n_permutations=1000):
    v0 = np.array(v0)
    v1 = np.array(v1)
    observed_stat = np.abs(cosine(v0, v1))
    
    perm_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(v0)
        np.random.shuffle(v1)
        perm_stat = np.abs(cosine(v0, v1))
        perm_stats.append(perm_stat)
    
    p_value = np.mean([p >= observed_stat for p in perm_stats])
    
    return observed_stat, p_value, perm_stats


def pca_correlation(data0=None, data1=None, data_name0='', data_name1=''):
    for k in data0:
        es = []
        for i in data0[k]:
            if i in EXCLUDE_ACTIONS:
                continue
            es.append(data0[k][i])
        es = np.array(es)
        if 'human' not in data_name0:
            es0 = scale(es)
        else:
            es0 = es
        
        es = []
        for i in data1[k]:
            if i in EXCLUDE_ACTIONS:
                continue
            es.append(data1[k][i])
        es = np.array(es)
        if 'human' not in data_name1:
            es1 = scale(es)
        else:
            es1 = es
        
        pca0 = PCA(n_components=3)
        pca0.fit(es0)
        sdev = np.diag(np.sqrt(pca0.explained_variance_))
        components0 = pca0.components_.T @ sdev
        components0 = varimax(components0)
        components0 = components0.T
        
        pca1 = PCA(n_components=3)
        pca1.fit(es1)
        sdev = np.diag(np.sqrt(pca1.explained_variance_))
        components1 = pca1.components_.T @ sdev
        components1 = varimax(components1)
        components1 = components1.T
        
        dims = dims_all[k[0]]

        labels0 = [None, None, None]
        for i in range(len(dims)):
            if dims[i] in ['不当获利', '危害他人性', '放纵性', '关心他人', '付出性', '自控力']:
                labels0[np.argmax(np.abs(components0[:, i]))] = dims[i]
        labels1 = [None, None, None]
        for i in range(len(dims)):
            if dims[i] in ['不当获利', '危害他人性', '放纵性', '关心他人', '付出性', '自控力']:
                labels1[np.argmax(np.abs(components1[:, i]))] = dims[i]
                
        labels11 = [None, None, None]
        for i in range(len(labels0)):
            for j in range(len(labels1)):
                if labels0[i] == labels1[j]:
                    labels11[i] = j
        
        components1 = components1[labels11]
        components = np.concatenate([components0, components1], axis=0)
        
        correlation = np.zeros([6, 6])
        p_value = np.zeros([6, 6])
        for i in range(6):
            for j in range(6):
                if i < j:
                    r = 0
                else:
                    r = cosine(components[i], components[j])
                correlation[i][j] = np.abs(r)
                p_value[i][j] = permutation_test_vectors(components[i], components[j])[1]
        
        v = correlation
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.set_xticks(np.arange(v.shape[1]), labels=[data_name0 + '\n' + i for i in labels0] + [data_name1 + '\n' + i for i in labels0])
        ax.set_yticks(np.arange(v.shape[0]), labels=[data_name0 + '\n' + i for i in labels0] + [data_name1 + '\n' + i for i in labels0])

        im = ax.imshow(np.abs(v), cmap='Reds', interpolation='none')
        
        for i in range(6):
            for j in range(6):
                t = str(v[i][j])[:4]
                if i >= j:
                    if p_value[i][j] < 0.001:
                        t += '\n<0.001'
                    else:
                        t += '\n' + str(p_value[i][j])[:6]
                    ax.text(j, i, t, ha="center", va="center", color="black")

        ax.set_title(k[:4] + '+' + data_name0 + '+' + data_name1)

        fig.tight_layout()
        fig.colorbar(im)
        
        fig.savefig('pic/3_' + k[:4] + '_' + data_name0 + '_' + data_name1 + '.png', dpi=300)
        

pca_correlation(data0=embeddings_llm_ancient, data1=embeddings_human_modern, data_name0='llm_ancient', data_name1='human_modern')
pca_correlation(data0=embeddings_bert_ancient, data1=embeddings_human_modern, data_name0='bert_ancient', data_name1='human_modern')

plt.show()
