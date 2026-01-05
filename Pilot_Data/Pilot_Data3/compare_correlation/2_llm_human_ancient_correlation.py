import glob
import json
import numpy as np
import scipy as sp
import pandas as pd


# 需要排除行为
# Actions to be excluded
EXCLUDE_ACTIONS = {}


def get_dim(is_evil):
    df = pd.read_excel('morality_feature-beh.xlsx', sheet_name=['evil-feature', 'good-feature'])
    s = {}
    for k in df:
        s[k] = {}
        for _, j in df[k].iterrows():
            s[k][j['维度名称']] = {'维度定义': j['维度定义'], '负端例子': j['负端例子'], '正端例子': j['正端例子']}
    if is_evil:
        return s['evil-feature']
    else:
        return s['good-feature']


def human_llm_correlation(fn):
    print(fn)
    embeddings = {}
    scores = json.loads(open(fn).read())
    data = pd.read_excel('古籍语料LLM标注+张改描述.xlsx', sheet_name=['恶行描述', '善行描述'])
    for k in data:
        dims = sorted(get_dim(True if k[0] == '恶' else False).keys())
        for _, a in data[k].iterrows():
            e = []
            for d in dims:
                if d in scores[a['行为']]: 
                    sss = scores[a['行为']][d][1]
                else:
                    sss = 0
                e.append(sss)
            e = np.array(e)
            if k not in embeddings:
                embeddings[k] = {}
            embeddings[k][a['行为']] = e
    # 因为数据点是抽样的，所以没有对模型结果做scale。
    # Since the data points are sampled, no scaling is done on the model results.
    
    scores = {}
    actions = set()
    for fn in glob.glob('ancient_human_rating/问卷_*'):
        df = pd.read_excel(fn)
        n = fn.split('_')[-2]
        if n not in scores:
            scores[n] = {}
        for _, i in df.iterrows():
            a = i['行为']
            scores[n][a + '_' + i['维度']] = i['分数（0-10，满分为10分，0分是完全没有，10分是程度极高，）']
            actions.add(a + '_' + i['维度'])

    scores_sorted = {k: [] for k in scores}
    scores_sorted['mean'] = []
    scores_sorted['LLM'] = []
    for k in data:
        dims = sorted(get_dim(True if k[0] == '恶' else False).keys())
        for _, i in data[k].iterrows():   
            if i['行为'] in EXCLUDE_ACTIONS:
                continue
            for m, ss in enumerate(dims):
                if i['行为'] + '_' + ss in actions:
                    scores_sorted['LLM'].append(embeddings[k][i['行为']][m])
                    scores_sorted['mean'].append(0)
                    for kk in scores:
                        scores_sorted[kk].append(scores[kk][i['行为'] + '_' + ss])
                        scores_sorted['mean'][-1] += scores[kk][i['行为'] + '_' + ss]
                    scores_sorted['mean'][-1] /= 3
                    actions.remove(i['行为'] + '_' + ss)
    
    s = np.array([i for i in scores_sorted.values()])
    print(s.shape)
    sim = np.zeros([s.shape[0], s.shape[0]])
    pv = np.zeros([s.shape[0], s.shape[0]])
    for i in range(s.shape[0]):
        for j in range(s.shape[0]):
            c = sp.stats.spearmanr(s[i], s[j])
            sim[i][j] = c.statistic
            pv[i][j] = c.pvalue
    
    print('r(mean):', sim[-1][-2])
    print('mean(r):', np.mean(sim[-1][:-2]))
    print()


human_llm_correlation('data_ancient_scores_gpt4omini20240718_1234_0716_no_desc.json')
human_llm_correlation('data_ancient_scores_gpt4omini20240718_1234_0705.json')
human_llm_correlation('data_ancient_scores_gpt4o20241120_1234_0705.json')
human_llm_correlation('data_ancient_scores_claude3720250219_1234_0705.json')
human_llm_correlation('data_ancient_scores_claude3520241022_1234_0705.json')
human_llm_correlation('data_ancient_scores_deepseekv30324_1234_0705.json')
human_llm_correlation('data_ancient_scores_qwen3-235b-a22b_1234_0705.json')
