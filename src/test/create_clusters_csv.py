import torch
from pyspark import SparkConf
from pyspark.sql import SparkSession
import json
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType


def inject_raid(x):
    # json_rp = json.loads(x[0])
    x[0]['raid'] = x[1]
    return x[0]


conf = SparkConf() \
    .setAppName("Dataset Processor") \
    .set("spark.driver.memory", "15g") \
    .set("spark.driver.maxResultSize", "0") \
    .setMaster("local[16]")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext

clusters_tensor = torch.load("../../dataset/AttentiveRAiDWalk128_dbscan_like_faiss_clusters.pt").type(torch.IntTensor)

# (cluster_id, rp_id)
clusters = sc.parallelize(clusters_tensor.tolist()).zipWithIndex().map(lambda x: (x[1], x[0]))
nodes = sc.textFile("../../dataset/raid_test_dump2/research_product").map(json.loads).zipWithIndex().map(lambda x: (x[1], x[0]))

# (rp_json, cluster_id)
join_res_rdd = nodes.join(clusters).map(lambda x: inject_raid(x[1])).filter(lambda x: x['raid'] != -1)

schema = StructType([
    StructField("description", StringType(), True),
    StructField("author", ArrayType(StringType()), True),
    StructField("title", StringType(), True),
    StructField("date", StringType(), True),
    StructField("type", StringType(), True),
    StructField("id", StringType(), False),
    StructField("subject", ArrayType(StringType()), True),
    StructField("raid", IntegerType(), False)
])

join_res_df = spark.createDataFrame(join_res_rdd, schema)

join_res_df.toPandas().to_csv("raid_inference_sample.csv")
