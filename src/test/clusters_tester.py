import torch
from pyspark import SparkConf
from pyspark.sql import SparkSession
import json

clusters = torch.load("../train/clustering/test_clusters.pt").type(torch.IntTensor)

raid_indexes = (clusters == 1486456).nonzero(as_tuple=True)[0].tolist()

conf = SparkConf() \
    .setAppName("Dataset Processor") \
    .set("spark.driver.memory", "15g") \
    .set("spark.driver.maxResultSize", "0") \
    .setMaster("local[16]")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext

nodes = sc.textFile("../../dataset/raid_test_dump2/research_product").map(json.loads).zipWithIndex()

nodes.filter(lambda x: x[1] in raid_indexes).foreach(print)

