import re
import glob
import json
import numpy as np
import scipy as sp
import pandas as pd
import pingouin as pg

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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


def get_data(fn):
    embeddings = {}
    scores = json.loads(open(fn).read())
    data = pd.read_excel('善行+恶行描述.xlsx', sheet_name=['恶行描述', '善行描述'])
    for k in data:
        for _, a in data[k].iterrows():
            e = []
            for d in dims_all[k[0]]:
                sss = scores[a['行为']][d]
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


embeddings_llm_ancient1011 = get_data('data_ancient_scores_gpt4omini20240718_40dim_2345_1011.json')
embeddings_llm_ancient_prompt = get_data('data_ancient_scores_gpt4omini20240718_40dim_1234_1011_prompt.json')
embeddings_llm_ancient_0 = get_data('data_ancient_scores_gpt4omini20240718_40dim_1234_1011_1.json')
embeddings_llm_ancient_1 = get_data('data_ancient_scores_gpt4omini20240718_40dim_1234_1011_2.json')
embeddings_llm_ancient_2 = get_data('data_ancient_scores_gpt4omini20240718_40dim_1234_1011_3.json')
embeddings_llm_ancient_format = get_data('data_ancient_scores_gpt4omini20240718_40dim_1234_1011_format.json')


def heatmap(v, t, title):
    fig, ax = plt.subplots(figsize=(4, 7))

    ax.set_xticks(np.arange(v.shape[1]), labels=np.arange(1, v.shape[1] + 1))
    ax.set_yticks(np.arange(v.shape[0]), labels=t)

    im = ax.imshow(np.abs(v), cmap='Reds', interpolation='none')

    ax.set_title(title)
    ax.set_xlabel("维度")
    ax.set_ylabel("行为")
    fig.tight_layout()
    fig.colorbar(im)
    

def correlation():
    for k in embeddings_llm_ancient_prompt:
        es = []
        for i in embeddings_llm_ancient_prompt[k]:
            if i in EXCLUDE_ACTIONS:
                continue
            es.append(embeddings_llm_ancient_prompt[k][i])
        es = np.array(es)
        es_llm_ancient_prompt = es
        
        es = []
        for i in embeddings_llm_ancient1011[k]:
            if i in EXCLUDE_ACTIONS:
                continue
            es.append(embeddings_llm_ancient1011[k][i])
        es = np.array(es)
        es_llm_ancient1011 = es
        
        es = []
        for i in embeddings_llm_ancient_0[k]:
            if i in EXCLUDE_ACTIONS:
                continue
            es.append(embeddings_llm_ancient_0[k][i])
        es = np.array(es)
        es_llm_ancient_0 = es
        
        es = []
        for i in embeddings_llm_ancient_1[k]:
            if i in EXCLUDE_ACTIONS:
                continue
            es.append(embeddings_llm_ancient_1[k][i])
        es = np.array(es)
        es_llm_ancient_1 = es
        
        es = []
        for i in embeddings_llm_ancient_2[k]:
            if i in EXCLUDE_ACTIONS:
                continue
            es.append(embeddings_llm_ancient_2[k][i])
        es = np.array(es)
        es_llm_ancient_2 = es
        
        es = []
        for i in embeddings_llm_ancient_format[k]:
            if i in EXCLUDE_ACTIONS:
                continue
            es.append(embeddings_llm_ancient_format[k][i])
        es = np.array(es)
        es_llm_ancient_format = es
        
        ms = [es_llm_ancient1011, es_llm_ancient_prompt, es_llm_ancient_0, es_llm_ancient_1, es_llm_ancient_2, es_llm_ancient_format]

        # Spearman correlation
        correlation = np.zeros([len(ms), len(ms)])
        p_value = np.zeros([len(ms), len(ms)])
        for i in range(len(ms)):
            for j in range(len(ms)):
                if i < j:
                    continue
                r = sp.stats.spearmanr(ms[i].reshape(-1), ms[j].reshape(-1))
                correlation[i][j] = r.statistic
                p_value[i][j] = r.pvalue
        
        v = correlation
        fig, ax = plt.subplots(figsize=(15, 15))
        
        labels = ['es_llm_ancient1011', 'es_llm_ancient_prompt', 'es_llm_ancient_0', 'es_llm_ancient_1', 'es_llm_ancient_2', 'es_llm_ancient_format']
        ax.set_xticks(np.arange(v.shape[1]), labels=labels)
        ax.set_yticks(np.arange(v.shape[0]), labels=labels)

        im = ax.imshow(np.abs(v), cmap='Reds', interpolation='none')
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i < j:
                    continue
                t = str(v[i][j])[:4]
                if p_value[i][j] < 0.001:
                    t += '\n<0.001'
                else:
                    t += '\n' + str(p_value[i][j])[:6]
                ax.text(j, i, t, ha="center", va="center", color="black")

        ax.set_title(k[:4] + '_spearmanr')
        fig.tight_layout()
        fig.colorbar(im)
        fig.savefig('pic/3_spearmanr_' + k[:4] + '.png', dpi=300)

        
        # ICC (Intraclass Correlation Coefficient)
        
        d = {
            'subject': range(1, len(ms[0].reshape(-1)) + 1),
        }
        for i, _ in enumerate(ms):
            d['rater' + str(i)] = ms[i].reshape(-1)
        d = pd.DataFrame(d)
        d = pd.melt(d, id_vars=['subject'], value_vars=['rater' + str(i) for i, _ in enumerate(ms)], var_name='rater', value_name='score')
        icc = pg.intraclass_corr(data=d, targets='subject', raters='rater', ratings='score')
        print(k)
        print(icc)
        
correlation()

# plt.show()
