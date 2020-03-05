from data_pre import article_vector
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