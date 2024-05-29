import torch
from pyspark import SparkConf
from pyspark.sql import SparkSession
import json

conf = SparkConf() \
    .setAppName("Dataset Processor") \
    .set("spark.driver.memory", "15g") \
    .set("spark.driver.maxResultSize", "0") \
    .setMaster("local[16]")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext

recommendations = torch.load("../dataset/AttentiveRAiDWalk_recommendations.pt")

nodes = sc.textFile("../dataset/raid_test_dump2/research_product").map(json.loads).zipWithIndex()

near_indexes = recommendations[4].tolist()[0:10]

nodes.filter(lambda x: x[1] in near_indexes).foreach(print)
