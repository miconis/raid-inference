import json

from dgl import DGLGraph
from pyspark import SparkContext
import torch

def print_graph_stats(sc: SparkContext, graph_path):
    research_product = sc.textFile(graph_path + "research_product").map(json.loads)
    project = sc.textFile(graph_path + "project")
    relation = sc.textFile(graph_path + "relation").map(json.loads)

    print("Nodes")
    print("Project:", project.count())
    print("Publication:", research_product.filter(lambda x: x['type']=='publication').count())
    print("Software:", research_product.filter(lambda x: x['type']=='software').count())
    print("Dataset:", research_product.filter(lambda x: x['type']=='dataset').count())
    print("Edges")
    print("Cites:", relation.filter(lambda x: x['relClass']=='cites').count())
    print("Produces:", relation.filter(lambda x: x['relClass']=='produces').count())
    print("References:", relation.filter(lambda x: x['relClass']=='references').count())
    print("Supplements:", relation.filter(lambda x: x['relClass']=='supplements').count())


def id_to_long(relation, which, long_id):
    """
    Change the OpenAIRE ID with the respective Long ID.
    """
    relation[which] = long_id
    return relation


def tensor_intersection(t1, t2):
    indices = torch.zeros_like(t1, dtype=torch.bool)
    for elem in t2:
        indices = indices | (t1 == elem)
    return t1[indices]
