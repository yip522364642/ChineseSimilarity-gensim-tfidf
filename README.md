# ChineseSimilarity-gensim-tfidf
"""<br />
基于gensim模块的中文句子相似度计算<br />
 <br />
思路如下：<br />
1.文本预处理：中文分词，去除停用词<br />
2.计算词频<br />
3.创建字典（单词与编号之间的映射） <br />
4.将待比较的文档转换为向量（词袋表示方法） <br />
5.建立语料库<br />
6.初始化模型<br />
7.创建索引<br />
8.相似度计算并返回相似度最大的文本<br />
"""<br />
<br />
可直接运行ChineseSimilartyCaculation.py<br />
stopwords.txt为中文停用词表<br />

