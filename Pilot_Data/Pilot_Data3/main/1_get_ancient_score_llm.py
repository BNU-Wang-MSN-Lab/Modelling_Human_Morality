import requests
import json
import re
import time
import random
import pandas as pd
from multiprocessing import Process, Manager


N_THREAD = 100


def get_gpt_response(d):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers={
        "Authorization": "Bearer sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    }
    data = {
        "model": "openai/gpt-4o-mini-2024-07-18",
        "seed": 1234,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": ""
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": d
                    },
                ]
            }
        ]
    }
    response = json.loads(requests.post(url, json=data, timeout=30, headers=headers).text)
    print(response)
    r = response['choices'][0]['message']['content']
    return r


def get_description(idx, rs):
    rr = []
    df = pd.read_excel('现代中国人数据最终版/Morality_feature_beh_1011.xlsx', sheet_name=['Evil_features', 'Good_features'])
    s = {}
    for k in df:
        s[k] = {}
        for i, j in df[k].iterrows():
            if (k[0] == 'E' and j['是否为40dim（全集）'] == 1) or (k[0] == 'G' and j['是否为40dim（全集）'] == 1) or j['是否为校标'] == 1:
            # if (k[0] == 'E' and j['是否为28dim（子集）'] == 1) or (k[0] == 'G' and j['是否为26dim（子集）'] == 1) or j['是否为校标'] == 1:
                s[k][j['维度名称']] = {'维度定义': j['维度定义'], '负端例子': j['负端例子'], '正端例子': j['正端例子']}
                
    old  = json.loads(open('data_ancient_scores_gpt4omini20240718_28dim_2345_0627.json').read())
    data = pd.read_excel('善行+恶行描述.xlsx', sheet_name=['恶行描述', '善行描述'])
    for k in data:
        n = 0
        for _, i in data[k].iterrows():
            n += 1
            if n % N_THREAD != idx:
                continue

            scores = {}
            if k[0] == '恶':
                sss = s['Evil_features']
            else:
                sss = s['Good_features']
            for ss in sorted(list(sss.keys())):
                
                q = '您作为一个熟悉中国古代知识的专家，需要对古代的某个行为从特定角度进行评估。\n\n'
                q += f'中国古代将{i["行为"]}认为是一种{k[:2]}。\n\n'
                q += re.sub(r"\n+", "\n", i["模型生成+张改"]) + '\n\n'
                q += f'根据描述，请判断{i["行为"]}作为{k[:2]}，在“{ss}”角度的程度分数。“{ss}”角度指的是{sss[ss]["维度定义"]}分数满分为10分，0分是完全无关或程度极低，10分是程度极高，需要在最后明确给出分数。\n'
                q += f'举例来说：分数低的表示{ss}{sss[ss]["负端例子"]}；分数高的表示{ss}{sss[ss]["正端例子"]}。\n'
                q += '请您从中国古代文化语境出发，考虑该行为的各个方面，包括这种行为的实施者、发生场景、发生过程、结果、影响及其他特征。需要强调的是，您需要判断的是您认为大多数古代中国人是怎样看待这种行为的，而不是您自己对这种行为的看法。'
                
                e = -1
                r = None
                
                if i['行为'] in old and ss in old[i['行为']]:
                    r = old[i['行为']][ss]
                    e = 100
                
                while r is None or e < 0:
                    time.sleep(random.random() * 5)
                    try:
                        r = get_gpt_response(q)
                        if r:
                            rrrr = re.sub(r'应当给予满分10分', '应当给予10分', r)
                            rrrr = re.sub(r'满分.?10分', '', rrrr)
                            rrrr = re.sub(r'（10分）', '', rrrr)
                            ii = []
                            for rrr in rrrr.split('\n'):
                                if '整体评估' not in rrr and '综上所述' not in rrr and '综合来看' not in rrr and '结合以上' not in rrr and '评估结论' not in rrr \
                                and '综合上述因素' not in rrr and '评估分数' not in rrr and '综合考虑' not in rrr and '综合以上' not in rrr and '总结与分数' not in rrr \
                                    and '综合评定' not in rrr:
                                    if re.findall(r'^\d\.', rrr):
                                        continue
                                    if re.findall(r'^\-', rrr):
                                        continue
                                    if re.findall(r'^ +', rrr):
                                        continue
                                    if not re.findall(r'\d', rrr):
                                        continue
                                if rrr:
                                    ii.append(rrr)
                            rrrr = '\n'.join(ii)
                            ssss = re.findall(r'\*\*(\d+\.?\d*?)[分/]\*\*', rrrr)
                            if not ssss:
                                ssss = re.findall(r'(\d+\.?\d*?)[分/]', rrrr)
                                if not ssss:
                                    ssss = re.findall(r'\d+', rrrr)
                                    if not ssss:
                                        print(sss)
                                        # print(scores[i['行为']][ss])
                            e = min(float(i) for i in ssss[:1] + ssss[-1:])
                    except:
                        pass
                    
                print('+' * 50)
                print(i)
                print('-' * 50)
                print(q)
                print('-' * 50)
                print([r])
                print(e)
                print('+' * 50)
                scores[ss] = r
                # break
            rr.append({'action': i['行为'], 'scores': scores})
            # break
    rs[idx] = rr


if __name__ == '__main__':
    rs = Manager().dict()
    ps = []
    for i in range(N_THREAD):
        p = Process(target=get_description, args=(i, rs))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
 
    data = {}
    for v in rs.values():
        for vv in v:
            data[vv['action']] = vv['scores']
    
    open('data_ancient_scores_gpt4omini20240718_40dim_2345_1011.json', 'w').write(json.dumps(data, ensure_ascii=False, indent=2))
