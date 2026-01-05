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


def pipeline(fn):
    embeddings = {}
    scores = json.loads(open(fn).read())
    data = pd.read_excel('善行+恶行描述.xlsx', sheet_name=['恶行描述', '善行描述'])
    for k in data:
        dims = sorted(get_dim(True if k[0] == '恶' else False).keys())
        for _, a in data[k].iterrows():
            e = []
            for d in dims:
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


    def human_llm_correlation():
        import opencc
        t2s = opencc.OpenCC('t2s')

        scores = {}
        actions = set()
        for fn in glob.glob('ancient_human_rating/问卷_*'):
            df = pd.read_excel(fn)
            n = fn.split('_')[-2]
            if n not in scores:
                scores[n] = {}
            for _, i in df.iterrows():
                a = t2s.convert(i['行为'])
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
        for i in range(s.shape[0]):
            for j in range(s.shape[0]):
                c = sp.stats.spearmanr(s[i], s[j])
                sim[i][j] = c.statistic
        
        print(scores_sorted.keys())
        print(sim)
        print('r(mean):', sim[-1][-2])
        print('mean(r):', np.mean(sim[-1][:-2]))


    human_llm_correlation()
    # exit()


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 11))
    fig.suptitle(fn)
    

    def heatmap(v, t, title, idx=0):
        ax = axes[idx]
        ax.set_xticks(np.arange(v.shape[1]), labels=np.arange(1, v.shape[1] + 1))
        ax.set_yticks(np.arange(v.shape[0]), labels=t)

        im = ax.imshow(np.abs(v), cmap='Reds', interpolation='none')

        ax.set_title(title)
        ax.set_xlabel("维度")
        ax.set_ylabel("行为")
        fig.tight_layout()
        fig.colorbar(im)
        

    def line_chart(x, y, x_label='', y_label='', title=''):
        fig, ax = plt.subplots(figsize=(4, 7))
        ax.plot(x, y)
        ax.set_xticks(x, labels=x)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()


    def get_sim(n_components=3):
        for k_idx, k in enumerate(embeddings):
            dims = sorted(get_dim(True if k[0] == '恶' else False).keys())
            es = []
            for i in embeddings[k]:
                if i in EXCLUDE_ACTIONS:
                    continue
                es.append(embeddings[k][i])
            es = np.array(es)
            es = scale(es)
            
            print(es.shape)

            pca = PCA(n_components=n_components)
            pca.fit(es)
            
            sdev = np.diag(np.sqrt(pca.explained_variance_))
            components = pca.components_.T @ sdev
            components = varimax(components)
            
            labels0 = [None, None, None]
            for i in range(len(dims)):
                if dims[i] in ['不当获利', '危害他人性', '放纵性', '关心他人', '付出性', '自控力']:
                    labels0[np.argmax(np.abs(components.T[:, i]))] = dims[i]
            print('Explained variance')
            print(labels0)
            print('raw:', pca.explained_variance_ratio_)
            print('normalize:', pca.explained_variance_ratio_ / np.sum(pca.explained_variance_ratio_))
            
            print('Explained variance (varimax)')
            print(labels0)
            explained_variance_ratio_varimax = np.sum(components * components, axis=0) / es.shape[1]
            print('raw:', explained_variance_ratio_varimax)
            print('normalize:', explained_variance_ratio_varimax / np.sum(explained_variance_ratio_varimax))
            
            ssss = sorted(zip(components, dims), key=lambda x: (-np.argmax(np.abs(x[0])), np.max(np.abs(x[0]))), reverse=True)
            heatmap(np.array([i[0] for i in ssss]), [i[1] for i in ssss], k[:2], idx=k_idx)
            
            print()
            
            # Scree plot
            # pca = PCA(n_components=25)
            # pca.fit(es)
            # line_chart(np.arange(1, 20), pca.explained_variance_, x_label='component', y_label='explained variance', title=k[:2])
        print()


    get_sim()
    fig.savefig('pic/2_' + fn + '.png', dpi=300)


pipeline('data_ancient_scores_gpt4omini20240718_40dim_2345_1011.json')
pipeline('data_ancient_scores_bnubert_1011.json')

plt.show()
