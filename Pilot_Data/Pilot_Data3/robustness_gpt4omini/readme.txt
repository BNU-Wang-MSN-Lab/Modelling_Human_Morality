This folder contains the results of robustness checks of LLM.
  1011 main results
  1011_prompt prompt with rephrased language
  1011_1 1011_2 1011_3 temperature=0.6, random 3 times
  1011_format prompt with section delimiters
  
这个文件夹是robustness的结果。里面包括：
  1011 主要实验的结果
  1011_prompt prompt调整的结果
  1011_1 1011_2 1011_3 temperature=0.6的实验结果重复了3次
  1011_format prompt相比1011使用了更清晰的格式

----------------
1_get_ancient_score*.py
Get results of LLM.
获取LLM的结果。
----------------

----------------
2_pca_ancient.py
PCA results.
查看不同实验的PCA结果。
----------------

----------------
3_correlation_ancient.py
Calculate Spearman correlation and ICC of different responses of LLMs.
All Spearman correlations > 0.8, most of ICC > 0.9.
计算不同实验结果间的Spearman相关系数和ICC。结果显示不同实验之间的Spearman相关系数超过0.8，ICC大部分超过0.9（超过0.9是非常显著）。

Evil ICC
恶行描述
    Type              Description       ICC          F   df1    df2  pval         CI95%
0   ICC1   Single raters absolute  0.881756  45.742575  3839  19200   0.0  [0.88, 0.89]
1   ICC2     Single random raters  0.881904  48.850942  3839  19195   0.0  [0.87, 0.89]
2   ICC3      Single fixed raters  0.888581  48.850942  3839  19195   0.0  [0.88, 0.89]
3  ICC1k  Average raters absolute  0.978139  45.742575  3839  19200   0.0  [0.98, 0.98]
4  ICC2k    Average random raters  0.978169  48.850942  3839  19195   0.0  [0.98, 0.98] ✓
5  ICC3k     Average fixed raters  0.979530  48.850942  3839  19195   0.0  [0.98, 0.98]

Good ICC
善行描述
    Type              Description       ICC          F   df1    df2  pval         CI95%
0   ICC1   Single raters absolute  0.866452  39.927525  2369  11850   0.0  [0.86, 0.87]
1   ICC2     Single random raters  0.866583  41.769483  2369  11845   0.0  [0.86, 0.88]
2   ICC3      Single fixed raters  0.871711  41.769483  2369  11845   0.0  [0.86, 0.88]
3  ICC1k  Average raters absolute  0.974955  39.927525  2369  11850   0.0  [0.97, 0.98]
4  ICC2k    Average random raters  0.974982  41.769483  2369  11845   0.0  [0.97, 0.98] ✓
5  ICC3k     Average fixed raters  0.976059  41.769483  2369  11845   0.0  [0.97, 0.98]
----------------

----------------
4_pca_correlation_ancient.py
Correlation of models and human ratings.
从三个维度来计算不同实验之间的Tucker's score。
----------------
