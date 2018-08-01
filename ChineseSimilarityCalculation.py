# -*- coding: utf-8 -*-
# @Time    : 2018/7/30 15:52
# @Author  : yip
# @Email   : 522364642@qq.com
# @File    : ChineseSimilarityCalculation.py

"""
基于gensim模块的中文句子相似度计算

思路如下：
1.文本预处理：中文分词，去除停用词
2.计算词频
3.创建字典（单词与编号之间的映射）
4.将待比较的文档转换为向量（词袋表示方法）
5.建立语料库
6.初始化模型
7.创建索引
8.相似度计算并返回相似度最大的文本
"""


from gensim import corpora, models, similarities
import logging
from collections import defaultdict
import jieba


# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 准备数据：现有8条文本数据，将8条文本数据放入到list中
documents = ["1)键盘是用于操作设备运行的一种指令和数据输入装置，也指经过系统安排操作一台机器或设备的一组功能键（如打字机、电脑键盘）",
             "2)鼠标称呼应该是“鼠标器”，英文名“Mouse”，鼠标的使用是为了使计算机的操作更加简便快捷，来代替键盘那繁琐的指令。",
             "3)中央处理器（CPU，Central Processing Unit）是一块超大规模的集成电路，是一台计算机的运算核心（Core）和控制核心（ Control Unit）。",
             "4)硬盘是电脑主要的存储媒介之一，由一个或者多个铝制或者玻璃制的碟片组成。碟片外覆盖有铁磁性材料。",
             "5)内存(Memory)也被称为内存储器，其作用是用于暂时存放CPU中的运算数据，以及与硬盘等外部存储器交换的数据。",
             "6)显示器（display）通常也被称为监视器。显示器是属于电脑的I/O设备，即输入输出设备。它是一种将一定的电子文件通过特定的传输设备显示到屏幕上再反射到人眼的显示工具。",
             "7)显卡（Video card，Graphics card）全称显示接口卡，又称显示适配器，是计算机最基本配置、最重要的配件之一。",
             "8)cache高速缓冲存储器一种特殊的存储器子系统，其中复制了频繁使用的数据以利于快速访问。"]
# 待比较的文档
new_doc = "内存又称主存，是CPU能直接寻址的存储空间，由半导体器件制成。"


# 1.文本预处理：中文分词，去除停用词
print('1.文本预处理：中文分词，去除停用词')
# 获取停用词
stopwords = set()
file = open("stopwords.txt", 'r', encoding='UTF-8')
for line in file:
    stopwords.add(line.strip())
file.close()

# 将分词、去停用词后的文本数据存储在list类型的texts中
texts = []
for line in documents:
    words = ' '.join(jieba.cut(line)).split(' ')    # 利用jieba工具进行中文分词
    text = []
    # 过滤停用词，只保留不属于停用词的词语
    for word in words:
        if word not in stopwords:
            text.append(word)
    texts.append(text)
for line in texts:
    print(line)

# 待比较的文档也进行预处理（同上）
words = ' '.join(jieba.cut(new_doc)).split(' ')
new_text = []
for word in words:
    if word not in stopwords:
        new_text.append(word)
print(new_text)

# 2.计算词频
print('2.计算词频')
frequency = defaultdict(int)  # 构建一个字典对象
# 遍历分词后的结果集，计算每个词出现的频率
for text in texts:
    for word in text:
        frequency[word] += 1
# 选择频率大于1的词(根据实际需求确定)
texts = [[word for word in text if frequency[word] > 1] for text in texts]
for line in texts:
    print(line)


# 3.创建字典（单词与编号之间的映射）
print('3.创建字典（单词与编号之间的映射）')
dictionary = corpora.Dictionary(texts)
print(dictionary)
# 打印字典，key为单词，value为单词的编号
print(dictionary.token2id)


# 4.将待比较的文档转换为向量（词袋表示方法）
print('4.将待比较的文档转换为向量（词袋表示方法）')
# 使用doc2bow方法对每个不同单词的词频进行了统计，并将单词转换为其编号，然后以稀疏向量的形式返回结果
new_vec = dictionary.doc2bow(new_text)
print(new_vec)


# 5.建立语料库
print('5.建立语料库')
# 将每一篇文档转换为向量
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)


# 6.初始化模型
print('6.初始化模型')
# 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数），表示方法为新的表示方法（Tfidf 实数权重）
tfidf = models.TfidfModel(corpus)
# 将整个语料库转为tfidf表示方法
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)


# 7.创建索引
print('7.创建索引')
# 使用上一步得到的带有tfidf值的语料库建立索引
index = similarities.MatrixSimilarity(corpus_tfidf)


# 8.相似度计算并返回相似度最大的文本
print('# 8.相似度计算并返回相似度最大的文本')
new_vec_tfidf = tfidf[new_vec]  # 将待比较文档转换为tfidf表示方法
print(new_vec_tfidf)
# 计算要比较的文档与语料库中每篇文档的相似度
sims = index[new_vec_tfidf]
print(sims)
sims_list = sims.tolist()
# print(sims_list.index(max(sims_list)))  # 返回最大值
print("最相似的文本为：", documents[sims_list.index(max(sims_list))])   # 返回相似度最大的文本


if __name__ == "__main__":
    pass

