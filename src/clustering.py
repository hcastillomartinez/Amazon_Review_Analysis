from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

session = SparkSession(sc)
session.conf.set("spark.sql.shuffle.partitions", 6)
sample_df = session.read.format("csv")\
    .option("header", "true")\
    .option("mode", "DROPMALFORMED")\
    .option("inferSchema", "true")\
    .option("sep", "\t")\
    .load("/FileStore/tables/final_sample/*.tsv")
    #update this path
print(sample_df.count())


# Where clustering is done
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, countDistinct
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd
max_clusters = 3
sil_list=[]
va = VectorAssembler(inputCols = ["helpful_votes","sentiment_rating", "subjectivity_rating","word_count"], outputCol = "features")
new_df = va.transform(sample_df)
for n in range(3,max_clusters+1):
  cluster_num = n
  print("Curr cluster # %d" % (cluster_num))
  kmeans = KMeans(k=cluster_num, seed=1)
  model = kmeans.fit(new_df.select('features'))
  clustered = model.transform(new_df)
  import numpy as np  
  pdf = clustered.toPandas()
  l = np.array_split(pdf,10)
  d = 0
  for df in l:
    df.to_csv('/FileStore/tables/clustered/clustered_samp'+d+'.tsv',sep='\t', quoting=csv.QUOTE_NONE)
    d +=1
  # Update path where it writes to
  df.to_csv('/FileStore/tables/clustered/clustered_samp.tsv',sep='\t', quoting=csv.QUOTE_NONE)
  