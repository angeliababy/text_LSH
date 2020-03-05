from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("test1") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')


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


# 二、word2vec训练分词数据
from pyspark.ml.feature import Word2Vec

w2v_model = Word2Vec(vectorSize=100, inputCol='words', outputCol='vector', minCount=3)
model = w2v_model.fit(words_df)
model.write().overwrite().save("models/word2vec_model/python.word2vec")

from pyspark.ml.feature import Word2VecModel

w2v_model = Word2VecModel.load("models/word2vec_model/python.word2vec")
vectors = w2v_model.getVectors()
vectors.show()


# 三、关键词获取(tfidf)
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


# 2.关键词权重乘以词向量
def compute_vector(row):
    return row.article_id, row.channel_id, row.keywords, row.weights * row.vector

article_keyword_vectors = keywords_vector.rdd.map(compute_vector).toDF(["article_id", "channel_id", "keywords", "weightingVector"])

# 利用 collect_set() 方法，将一篇文章内所有关键词的词向量合并为一个列表
article_keyword_vectors.registerTempTable('temptable')
article_keyword_vectors = spark.sql("select article_id, min(channel_id) channel_id, collect_set(weightingVector) vectors from temptable group by article_id")

# 3.计算权重向量平均值
def compute_avg_vectors(row):
    x = 0
    for i in row.vectors:
        x += i
    # 求平均值
    return row.article_id, row.channel_id, x / len(row.vectors)

article_vector = article_keyword_vectors.rdd.map(compute_avg_vectors).toDF(['article_id', 'channel_id', 'articlevector'])
print("文章最终vector",article_vector.take(10))




