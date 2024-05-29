import json
import os
import random

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from pyspark import SparkConf
from pyspark.sql import SparkSession
from utils import *

random.seed(1234)
np.random.seed(1234)


class OpenAIRESubgraph(DGLDataset):
    """
    Custom subgraph.

    Parameters
    ----------
    raw_dir : directory that will store (or already stores) the downloaded data
    save_dir : directory to save preprocessed data
    force_reload : whether to reload dataset
    verbose : whether to print out progress information
    """

    def __init__(self,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        # context initialization
        self.conf = SparkConf() \
            .setAppName("Dataset Processor") \
            .set("spark.driver.memory", "15g") \
            .set("spark.driver.maxResultSize", "0") \
            .setMaster("local[16]")
        self.spark = SparkSession.builder.config(conf=self.conf).getOrCreate()
        self.sc = self.spark.sparkContext

        self.graph = None

        super().__init__(name="OpenAIRE Subgraph",
                         raw_dir=raw_dir,
                         save_dir=save_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def get_sc(self):
        return self.sc

    def download(self):
        """
        Download the raw data to local disk (create single files to be processed)
        """
        print("Data already downloaded!")
        pass

    def process(self):
        """
        Process raw data to create the graph
        """
        # Create node features
        print("Creating node features")
        research_product = self.sc.textFile(self.raw_dir + "research_product").map(json.loads).zipWithIndex()
        project = self.sc.textFile(self.raw_dir + "project").map(json.loads).zipWithIndex()
        organization = self.sc.textFile(self.raw_dir + "organization").map(json.loads).zipWithIndex()

        # collect Node's IDs and their respective Long ID to create edges
        nodes_id = research_product \
            .map(lambda x: (x[0]['id'], x[1])) \
            .union(project.map(lambda x: (x[0]['id'], x[1]))) \
            .union(organization.map(lambda x: (x[0]['id'], x[1])))

        # create node subtypes
        project_types = torch.zeros(project.count(), dtype=torch.long)  # projects are of the same type
        research_product_types = torch.LongTensor(
            research_product
            .map(lambda x: x[0]['type'])
            .map(lambda x: 1 if x == "software" else 2 if x == "dataset" else 0)  # 0: Publication, 1: Software, 2: Dataset
            .collect()
        )
        organization_types = torch.zeros(organization.count(), dtype=torch.long)  # organizations are of the same type

        # Create edges
        print("Creating edges")
        relations = self.sc.textFile(self.raw_dir + "relation") \
            .map(json.loads) \
            .map(lambda x: (x['source'], x)) \
            .join(nodes_id) \
            .map(lambda x: (x[1][0]['target'], id_to_long(x[1][0], "source", x[1][1]))) \
            .join(nodes_id) \
            .map(lambda x: id_to_long(x[1][0], "target", x[1][1]))

        # cites relations
        cites_rels = torch.LongTensor(
            relations.filter(lambda x: x['relClass'] == 'cites').map(lambda x: [x['source'], x['target']]).collect()
        )

        # produces relations
        produces_rels = torch.LongTensor(
            relations.filter(lambda x: x['relClass'] == 'produces').map(lambda x: [x['source'], x['target']]).collect()
        )

        # references relations
        references_rels = torch.LongTensor(
            relations.filter(lambda x: x['relClass'] == 'references').map(
                lambda x: [x['source'], x['target']]).collect()
        )

        # supplements relations
        supplements_rels = torch.LongTensor(
            relations.filter(lambda x: x['relClass'] == 'supplements').map(
                lambda x: [x['source'], x['target']]).collect()
        )

        # participants relations
        # participants_rels = torch.LongTensor(
        #     relations.filter(lambda x: x['relClass'] == 'participants').map(
        #         lambda x: [x['source'], x['target']]).collect()
        # )

        # versions relations
        versions_rels = torch.LongTensor(
            relations.filter(lambda x: x['relClass'] == 'versions').map(
                lambda x: [x['source'], x['target']]).collect()
        )

        # institutions relations
        institutions_rels = torch.LongTensor(
            relations.filter(lambda x: x['relClass'] == 'institutions').map(
                lambda x: [x['source'], x['target']]).collect()
        )

        # parts relations
        parts_rels = torch.LongTensor(
            relations.filter(lambda x: x['relClass'] == 'parts').map(
                lambda x: [x['source'], x['target']]).collect()
        )

        # Create graph
        print("Creating graph")
        self.graph = dgl.heterograph(
            data_dict={
                ("research_product", "Cites", "research_product"): (cites_rels[:, 0], cites_rels[:, 1]),
                ("research_product", "IsCitedBy", "research_product"): (cites_rels[:, 1], cites_rels[:, 0]),
                ("research_product", "References", "research_product"): (references_rels[:, 0], references_rels[:, 1]),
                ("research_product", "IsReferencedBy", "research_product"): (references_rels[:, 1], references_rels[:, 0]),
                ("research_product", "IsSupplementTo", "research_product"): (supplements_rels[:, 0], supplements_rels[:, 1]),
                ("research_product", "IsSupplementedBy", "research_product"): (supplements_rels[:, 1], supplements_rels[:, 0]),
                ("project", "Produces", "research_product"): (produces_rels[:, 0], produces_rels[:, 1]),
                ("research_product", "IsProducedBy", "project"): (produces_rels[:, 1], produces_rels[:, 0]),
                ("research_product", "IsVersionOf", "research_product"): (versions_rels[:, 0], versions_rels[:, 1]),
                ("research_product", "HasVersion", "research_product"): (versions_rels[:, 1], versions_rels[:, 0]),
                ("research_product", "IsPartOf", "research_product"): (parts_rels[:, 0], parts_rels[:, 1]),
                ("research_product", "HasPartOf", "research_product"): (parts_rels[:, 1], parts_rels[:, 0]),
                # ("project", "HasParticipant", "organization"): (participants_rels[:, 0], participants_rels[:, 1]),
                # ("organization", "IsParticipant", "project"): (participants_rels[:, 1], participants_rels[:, 0]),
                ("research_product", "HasAuthorInstitution", "organization"): (institutions_rels[:, 0], institutions_rels[:, 1]),
                ("organization", "IsAuthorInstitutionOf", "research_product"): (institutions_rels[:, 1], institutions_rels[:, 0])
            },
            num_nodes_dict={
                "research_product": research_product.count(),
                "project": project.count(),
                "organization": organization.count()
            }
        )

        self.graph.ndata["ntype"] = {
            "research_product": research_product_types,
            "project": project_types,
            "organization": organization_types
        }

    def get_graph(self):
        """
        Returns the Graph
        """
        return self.graph[0]

    def save(self):
        """
        Save processed data to directory (self.save_path)
        """
        print("Saving graph on disk")
        save_graphs(self.save_dir + "openaire_subgraph.dgl", [self.graph])

    def load(self):
        """
        Load processed data from directory (self.save_path)
        """
        print("Load graph from disk")
        self.graph = load_graphs(self.save_dir + "openaire_subgraph.dgl")[0]
        self.sc.stop()

    def has_cache(self):
        """
        Check whether there are processed data
        """
        return os.path.exists(self.save_dir + "openaire_subgraph.dgl")
