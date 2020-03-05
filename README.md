**本文目的**

最近在研究LSH方法，主要发现用pyspark实现的较少，故结合黑马头条推荐系统实践的视频进行了本地实现。

本项目完整源码地址：
[https://github.com/angeliababy/text_LSH](https://github.com/angeliababy/text_LSH)

项目博客地址:
[https://blog.csdn.net/qq_29153321/article/details/104680282](https://blog.csdn.net/qq_29153321/article/details/104680282)
## 算法
本章主要介绍如何使用文章关键词获取文章相似性。主要用到了Word2Vec+Tfidf+LSH算法。
1.使用Word2Vec训练出文章的词向量。
2.Tfidf获取文章关键词及权重。
3.使用关键词权重乘以其词向量平均值作为训练集。
4.使用LSH求取两两文章相似性。

对于海量的数据，通过两两文章向量的欧式距离求取与当前文章最相似的文章，显然不太现实，故采取LSH进行相似性检索。

LSH即局部敏感哈希，主要用来解决海量数据的相似性检索。由spark的官方文档翻译为：LSH的一般思想是使用一系列函数将数据点哈希到桶中，使得彼此接近的数据点在相同的桶中具有高概率，而数据点是远离彼此很可能在不同的桶中。spark中LSH支持欧式距离与Jaccard距离。在此欧式距离使用较广泛。

## 实践
部分原始数据：

news_data:
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020030518151962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MTUzMzIx,size_16,color_FFFFFF,t_70)

**一、获取分词数据**

主要处理一个频道下的数据，便于进行文章相似性计算
```
# 中文分词
def segmentation(partition):
    import os
    import re
    import jieba
    import jieba.analyse
    import jieba.posseg as pseg
    import codecs

    # abspath = "words"

    # # 结巴加载用户词典
    # userDict_path = os.path.join(abspath, "ITKeywords.txt")
    # jieba.load_userdict(userDict_path)
    #
    # # 停用词文本
    # stopwords_path = os.path.join(abspath, "stopwords.txt")
    # def get_stopwords_list():
    #     """返回stopwords列表"""
    #     stopwords_list = [i.strip() for i in codecs.open(stopwords_path).readlines()]
    #     return stopwords_list
    # # 所有的停用词列表
    # stopwords_list = get_stopwords_list()

    # 分词
    def cut_sentence(sentence):
        """对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词"""
        seg_list = pseg.lcut(sentence)
        # seg_list = [i for i in seg_list if i.flag not in stopwords_list]
        filtered_words_list = []
        for seg in seg_list:
            if len(seg.word) <= 1:
                continue
            elif seg.flag == "eng":
                if len(seg.word) <= 2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            elif seg.flag in ["x", "eng"]:  # 是自定一个词语或者是英文单词
                filtered_words_list.append(seg.word)
        return filtered_words_list

    for row in partition:
        if row[1] == '4':
            sentence = re.sub("<.*?>", "", row[4])  # 替换掉标签数据
            words = cut_sentence(sentence)
            yield row[0], row[1], words


# 一、获取分词数据
# 数据：article_id,channel_id,channel_name,title,content,sentence
article_data = spark.sparkContext.textFile(r'news_data')
article_data = article_data.map(lambda line: line.split('\x01'))
print("原始数据", article_data.take(10))
words_df = article_data.mapPartitions(segmentation).toDF(['article_id', 'channel_id', 'words'])
print("分词数据", words_df.take(10))
```
数据格式：article_id,channel_id,channel_name,title,content,sentence
![在这里插入图片描述](https://img-blog.csdnimg.cn/202003052111286.png)
也可按实际情况正则去掉英文。

**二、word2vec训练分词数据**

```
# 二、word2vec训练分词数据
from pyspark.ml.feature import Word2Vec

w2v_model = Word2Vec(vectorSize=100, inputCol='words', outputCol='vector', minCount=3)
model = w2v_model.fit(words_df)
model.write().overwrite().save("models/word2vec_model/python.word2vec")

from pyspark.ml.feature import Word2VecModel

w2v_model = Word2VecModel.load("models/word2vec_model/python.word2vec")
vectors = w2v_model.getVectors()
vectors.show()
```
得到频道下所有词的词向量
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305211224686.png)

**三、关键词获取**

1.关键词机器权重和词向量
```
# tdidf
# 词频，即tf
from pyspark.ml.feature import CountVectorizer

# vocabSize是总词汇的大小，minDF是文本中出现的最少次数
cv = CountVectorizer(inputCol="words", outputCol="countFeatures", vocabSize=200 * 10000, minDF=1.0)
# 训练词频统计模型
cv_model = cv.fit(words_df)
cv_model.write().overwrite().save("models/CV.model")

from pyspark.ml.feature import CountVectorizerModel

cv_model = CountVectorizerModel.load("models/CV.model")
# 得出词频向量结果
cv_result = cv_model.transform(words_df)

# idf
from pyspark.ml.feature import IDF

idf = IDF(inputCol="countFeatures", outputCol="idfFeatures")
idf_model = idf.fit(cv_result)
idf_model.write().overwrite().save("models/IDF.model")

# tf-idf
from pyspark.ml.feature import IDFModel

idf_model = IDFModel.load("models/IDF.model")
tfidf_result = idf_model.transform(cv_result)

# 选取前20个作为关键词,此处仅为词索引
def sort_by_tfidf(partition):
    TOPK = 20
    for row in partition:
        # 找到索引与IDF值并进行排序
        _dict = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
        _dict = sorted(_dict, key=lambda x: x[1], reverse=True)
        result = _dict[:TOPK]
        for word_index, tfidf in result:
            yield row.article_id, row.channel_id, int(word_index), round(float(tfidf), 4)

keywords_by_tfidf = tfidf_result.rdd.mapPartitions(sort_by_tfidf).toDF(["article_id", "channel_id", "index", "weights"])

# 构建关键词与索引
keywords_list_with_idf = list(zip(cv_model.vocabulary, idf_model.idf.toArray()))

def append_index(data):
    for index in range(len(data)):
        data[index] = list(data[index])  # 将元组转为list
        data[index].append(index)  # 加入索引
        data[index][1] = float(data[index][1])

append_index(keywords_list_with_idf)
sc = spark.sparkContext
rdd = sc.parallelize(keywords_list_with_idf)  # 创建rdd
idf_keywords = rdd.toDF(["keywords", "idf", "index"])

# 求出文章关键词及权重tfidf
keywords_result = keywords_by_tfidf.join(idf_keywords, idf_keywords.index == keywords_by_tfidf.index).select(
    ["article_id", "channel_id", "keywords", "weights"])
print("关键词权重", keywords_result.take(10))

# 文章关键词与词向量join
keywords_vector = keywords_result.join(vectors, vectors.word == keywords_result.keywords, 'inner')
```
得到文章关键词的权重如下，并与上步join得到其词向量
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305211306492.png)

2.关键词权重乘以词向量
```
def compute_vector(row):
    return row.article_id, row.channel_id, row.keywords, row.weights * row.vector

article_keyword_vectors = keywords_vector.rdd.map(compute_vector).toDF(["article_id", "channel_id", "keywords", "weightingVector"])

# 利用 collect_set() 方法，将一篇文章内所有关键词的词向量合并为一个列表
article_keyword_vectors.registerTempTable('temptable')
article_keyword_vectors = spark.sql("select article_id, min(channel_id) channel_id, collect_set(weightingVector) vectors from temptable group by article_id")

```

3.计算权重向量平均值

```
def compute_avg_vectors(row):
    x = 0
    for i in row.vectors:
        x += i
    # 求平均值
    return row.article_id, row.channel_id, x / len(row.vectors)

article_vector = article_keyword_vectors.rdd.map(compute_avg_vectors).toDF(['article_id', 'channel_id', 'articlevector'])
print("文章最终vector",article_vector.take(10))
```
将文章关键词权重与词向量加权平均后得到训练数据（此处为什么不用全量的词，而用关键词可以思考下）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305211432919.png)

**四、LSH相似性**

```
# LSH
from pyspark.ml.feature import BucketedRandomProjectionLSH, MinHashLSH

train = article_vector.select(['article_id', 'articlevector'])

# 1.BucketedRandomProjectionLSH
brp = BucketedRandomProjectionLSH(inputCol='articlevector', outputCol='hashes', numHashTables=4.0, bucketLength=10.0)
model = brp.fit(train)

similar = model.approxSimilarityJoin(train, train, 2.0, distCol='EuclideanDistance')
similar.show()

# 2.MinHashLSH
brp = MinHashLSH(inputCol='articlevector', outputCol='hashes', numHashTables=4.0)
model = brp.fit(train)

# 获取所有相似对
similar = model.approxSimilarityJoin(train, train, 2.0, distCol='EuclideanDistance')
similar.show()
# 获取key指定个数的最近邻
# similar = model.approxNearestNeighbors(train, key, 2)
```

BucketedRandomProjectionLSH结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305211500467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MTUzMzIx,size_16,color_FFFFFF,t_70)

MinHashLSH结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200305211530413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MTUzMzIx,size_16,color_FFFFFF,t_70)
一般来讲第一种LSH在此处更适合。