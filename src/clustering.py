from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

session = SparkSession(sc)
session.conf.set("spark.sql.shuffle.partitions", 5)
sample_df = session.read.format("csv")\
    .option("header", "true")\
    .option("mode", "DROPMALFORMED")\
    .option("inferSchema", "true")\
    .option("sep", "\t")\
    .load("/FileStore/tables/*.tsv")
print(sample_df.count())

# max_help = sample_df.groupby().max('helpful_votes').first().asDict()['max(helpful_votes)']
# max_sent = sample_df.groupby().max('sentiment_rating').first().asDict()['max(sentiment_rating)']
# max_sub = sample_df.groupby().max('subjectivity_rating').first().asDict()['max(subjectivity_rating)']
# print("Max help = %d Max sent = %d Max sub = %d" % (max_help,max_sent,max_sub))


# Where clustering is done
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, countDistinct
from pyspark.ml.evaluation import ClusteringEvaluator
max_clusters = 10
sil_list=[]
va = VectorAssembler(inputCols = ["helpful_votes","sentiment_rating", "subjectivity_rating","word_count"], outputCol = "features")
new_df = va.transform(sample_df)
for n in range(4,max_clusters+1):
  cluster_num = n
  print("Curr cluster # %d" % (cluster_num))
  kmeans = KMeans(k=cluster_num, seed=1)
  model = kmeans.fit(new_df.select('features'))
  clustered = model.transform(new_df)
  # Get Summary stats for each cluster
  for k in range(cluster_num):
    clustered.filter(clustered.prediction==k).select("helpful_votes", "sentiment_rating",      "subjectivity_rating","word_count","sentence_count","adjective_count","adverb_count", "noun_count","pronoun_count","verb_count","prediction")\
    .describe()\
    .show()

  evaluator = ClusteringEvaluator()
  # range is from -1 to 1. The higher the number means points are correctly assigned to the correct cluster
  # Negative means some points may have been assigned to wrong cluster.
  silhouette = evaluator.evaluate(clustered)
  sil_list.append(silhouette)
  print("Silhouette with squared euclidean distance = " + str(silhouette))
  # Shows the result.
  centers = model.clusterCenters()
  print("Cluster Centers: ")
  for center in centers:
      print(center)

print(sil_list)