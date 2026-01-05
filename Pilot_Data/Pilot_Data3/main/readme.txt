This folder contains the main results of LLM and ancient Bert.
LLM is gpt-4o-mini-2024-07-18 and Bert is Ancient Chinese Bert.
这个是llm和bert的主要实验结果。
llm用的是gpt-4o-mini-2024-07-18，bert用的是古汉语bert。

+++++++++++++
1_get_ancient_score*.py
Get results from models.
获取模型结果。
+++++++++++++

+++++++++++++
2_pca_ancient.py
PCA reuslts.
查看降维结果。

#############
LLM explained variance
#############
Evil
恶行

Explained variance
['Harm to others', 'Unjust-gain', 'Self-indulgence']
['危害他人性', '不当获利', '放纵性']
raw: [0.39838342 0.08081377 0.05511686]
normalize: [0.74559787 0.1512477  0.10315443]

Explained variance (varimax)
['Harm to others', 'Unjust-gain', 'Self-indulgence']
['危害他人性', '不当获利', '放纵性']
raw: [0.32808169 0.14075179 0.0739281 ]
normalize: [0.60446742 0.25932526 0.13620732]
-------------
Good
善行

Explained variance
['Care for others', 'Self-control', 'Dedication']
['关心他人', '自控力', '付出性']
raw: [0.23739806 0.14712175 0.07488768]
normalize: [0.51674833 0.32024239 0.16300927]

Explained variance (varimax)
['Care for others', 'Self-control', 'Dedication']
['关心他人', '自控力', '付出性']
raw: [0.21683034 0.13607163 0.11836071]
normalize: [0.46010505 0.2887384  0.25115655]
+++++++++++++

+++++++++++++
3_pca_correlation_ancient.py
The correlations of PCA and human ratings.
查看降维结果和人的相关性，即Tucker's Score。
+++++++++++++
