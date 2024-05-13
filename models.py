from dgl.nn.pytorch import DeepWalk
import dgl
import random

import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.nn import init
from dgl.sampling import random_walk, PinSAGESampler
from dgl.sampling import node2vec_random_walk
from dgl.random import choice
from dgl.nn.pytorch import DeepWalk

import utils


class RAiDAttentiveWalk(nn.Module):
    """
    It learns the research product node representations from scratch by maximizing the similarity of
    node pairs that are nearby (positive node pairs) and minimizing the similarity of other
    random node pairs (negative node pairs). Nearby nodes are determined using random walks
    computed following different types of edges.

    Parameters
    ----------
    skg : DGLGraph
        Graph for learning node embeddings
    emb_dim : int, optional
        Size of each embedding vector. Default: 128
    walk_length : int, optional
        Number of nodes in a random walk sequence. Default: 40
    window_size : int, optional
        In a random walk :attr:`w`, a node :attr:`w[j]` is considered close to a node
        :attr:`w[i]` if :attr:`i - window_size <= j <= i + window_size`. Default: 5
    neg_weight : float, optional
        Weight of the loss term for negative samples in the total loss. Default: 1.0
    negative_size : int, optional
        Number of negative samples to use for each positive sample. Default: 1
    fast_neg : bool, optional
        If True, it samples negative node pairs within a batch of random walks. Default: True
    sparse : bool, optional
        If True, gradients with respect to the learnable weights will be sparse.
        Default: True

    Attributes
    ----------
    node_embed : nn.Embedding
        Embedding table of the nodes
    """

    def __init__(
            self,
            skg,
            emb_dim=128,
            walk_length=40,
            window_size=5,
            neg_weight=1,
            negative_size=5,
            fast_neg=True,
            sparse=True,
    ):
        super().__init__()

        assert (
                walk_length >= window_size + 1
        ), f"Expect walk_length >= window_size + 1, got {walk_length} and {window_size + 1}"

        num_nodes = skg.num_nodes('research_product')

        # process heterogeneous graph
        self.g = skg
        self.coproduced_graph = self.graph_metapath_sampling('research_product', 'project')
        self.coinstitution_graph = self.graph_metapath_sampling('research_product', 'organization')
        self.part_graph = self.graph_sampling('research_product', 'HasPartOf')
        self.version_graph = self.graph_sampling('research_product', 'IsVersionOf')
        self.citation_graph = self.graph_sampling('research_product', 'Cites')
        self.references_graph = self.graph_sampling('research_product', 'References')
        self.supplements_graph = self.graph_sampling('research_product', 'IsSupplementedBy')

        self.emb_dim = emb_dim
        self.window_size = window_size
        self.walk_length = walk_length
        self.neg_weight = neg_weight
        self.negative_size = negative_size
        self.fast_neg = fast_neg

        # center node embedding
        self.node_embed = nn.Embedding(num_nodes, emb_dim, sparse=sparse)
        self.context_embed = nn.Embedding(num_nodes, emb_dim, sparse=sparse)
        self.reset_parameters()

        if not fast_neg:
            neg_prob = skg.out_degrees().pow(0.75)
            # categorical distribution for true negative sampling
            self.neg_prob = neg_prob / neg_prob.sum()

        # Create lists of indexes for positive samples.
        # Given i, positive index pairs are (i - window_size, i), ... ,
        # (i - 1, i), (i + 1, i), ..., (i + window_size, i)
        idx_list_src = []
        idx_list_dst = []

        for i in range(walk_length):
            for j in range(max(0, i - window_size), i):
                idx_list_src.append(j)
                idx_list_dst.append(i)
            for j in range(i + 1, min(walk_length, i + 1 + window_size)):
                idx_list_src.append(j)
                idx_list_dst.append(i)

        self.idx_list_src = torch.LongTensor(idx_list_src)
        self.idx_list_dst = torch.LongTensor(idx_list_dst)

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        init_range = 1.0 / self.emb_dim
        init.uniform_(self.node_embed.weight.data, -init_range, init_range)
        init.constant_(self.context_embed.weight.data, 0)

    def graph_sampling(self, ntype, etype):
        edges = self.g.edges(etype=etype)
        return dgl.add_reverse_edges(dgl.graph((edges[0], edges[1]), num_nodes=self.g.num_nodes(ntype)))

    def graph_metapath_sampling(self, ntype, other_type):
        """
        Create the homogeneous graph by sampling the edges

        Parameters
        ----------
        ntype : str
            The node type for which the graph would be constructed on
        other_type : str
            The other node type

        Returns
        ----------
        graph : DGLGraph
            The homogeneous graph
        """
        sampler = PinSAGESampler(self.g, ntype, other_type, 3, 0.5, 50, 20)
        seeds = torch.arange(self.g.num_nodes(ntype))
        frontier = sampler(seeds)
        src_id = frontier.all_edges(form='uv')[0]
        dst_id = frontier.all_edges(form='uv')[1]
        return dgl.add_reverse_edges(dgl.remove_self_loop(dgl.graph((src_id, dst_id), num_nodes=self.g.num_nodes(ntype))))

    def sample(self, indices):
        """
        Create random walks depending on the homogeneous graph

        Global walks (p=1, q=0.5) learn community structure
        Local walks (p=1, q=2) learn local roles
        """
        indices = torch.Tensor(indices)  # sample indices
        # coproduced graph
        sample_indices = utils.tensor_intersection((self.coproduced_graph.out_degrees() != 0).nonzero(as_tuple=True)[0], indices)
        coproduced_random_walk = node2vec_random_walk(self.coproduced_graph, sample_indices, p=1, q=2, walk_length=self.walk_length)
        # coinstitution graph
        sample_indices = utils.tensor_intersection((self.coinstitution_graph.out_degrees() != 0).nonzero(as_tuple=True)[0], indices)
        coinstitution_random_walk = node2vec_random_walk(self.coinstitution_graph, sample_indices, p=1, q=2, walk_length=self.walk_length)
        # part graph
        sample_indices = utils.tensor_intersection((self.part_graph.out_degrees() != 0).nonzero(as_tuple=True)[0], indices)
        part_random_walk = node2vec_random_walk(self.part_graph, sample_indices, p=1, q=0.25, walk_length=self.walk_length)
        # version graph
        sample_indices = utils.tensor_intersection((self.version_graph.out_degrees() != 0).nonzero(as_tuple=True)[0], indices)
        version_random_walk = node2vec_random_walk(self.version_graph, sample_indices, p=1, q=0.25, walk_length=self.walk_length)
        # citation graph
        sample_indices = utils.tensor_intersection((self.citation_graph.out_degrees() != 0).nonzero(as_tuple=True)[0], indices)
        citation_random_walk = node2vec_random_walk(self.citation_graph, sample_indices, p=1, q=1, walk_length=self.walk_length)
        # references graph
        sample_indices = utils.tensor_intersection((self.references_graph.out_degrees() != 0).nonzero(as_tuple=True)[0], indices)
        references_random_walk = node2vec_random_walk(self.references_graph, sample_indices, p=1, q=1, walk_length=self.walk_length)
        # supplements graph
        sample_indices = utils.tensor_intersection((self.supplements_graph.out_degrees() != 0).nonzero(as_tuple=True)[0], indices)
        supplements_random_walk = node2vec_random_walk(self.supplements_graph, sample_indices, p=1, q=0.25, walk_length=self.walk_length)

        random_walks = torch.cat((
            coproduced_random_walk,
            coinstitution_random_walk,
            part_random_walk,
            version_random_walk,
            citation_random_walk,
            references_random_walk,
            supplements_random_walk), 0)
        return random_walks

    def forward(self, batch_walk):
        """Compute the loss for the batch of random walks

        Parameters
        ----------
        batch_walk : torch.Tensor
            Random walks in the form of node ID sequences. The Tensor
            is of shape :attr:`(batch_size, walk_length)`.

        Returns
        -------
        torch.Tensor
            Loss value
        """
        batch_size = len(batch_walk)
        device = batch_walk.device

        batch_node_embed = self.node_embed(batch_walk).view(-1, self.emb_dim)
        batch_context_embed = self.context_embed(batch_walk).view(
            -1, self.emb_dim
        )

        batch_idx_list_offset = torch.arange(batch_size) * self.walk_length
        batch_idx_list_offset = batch_idx_list_offset.unsqueeze(1)
        idx_list_src = batch_idx_list_offset + self.idx_list_src.unsqueeze(0)
        idx_list_dst = batch_idx_list_offset + self.idx_list_dst.unsqueeze(0)
        idx_list_src = idx_list_src.view(-1).to(device)
        idx_list_dst = idx_list_dst.view(-1).to(device)

        pos_src_emb = batch_node_embed[idx_list_src]
        pos_dst_emb = batch_context_embed[idx_list_dst]

        neg_idx_list_src = idx_list_dst.unsqueeze(1) + torch.zeros(
            self.negative_size
        ).unsqueeze(0).to(device)
        neg_idx_list_src = neg_idx_list_src.view(-1)
        neg_src_emb = batch_node_embed[neg_idx_list_src.long()]

        if self.fast_neg:
            neg_idx_list_dst = list(range(batch_size * self.walk_length)) * (
                    self.negative_size * self.window_size * 2
            )
            random.shuffle(neg_idx_list_dst)
            neg_idx_list_dst = neg_idx_list_dst[: len(neg_idx_list_src)]
            neg_idx_list_dst = torch.LongTensor(neg_idx_list_dst).to(device)
            neg_dst_emb = batch_context_embed[neg_idx_list_dst]
        else:
            neg_dst = choice(
                self.g.num_nodes(), size=len(neg_src_emb), prob=self.neg_prob
            )
            neg_dst_emb = self.context_embed(neg_dst.to(device))

        pos_score = torch.sum(torch.mul(pos_src_emb, pos_dst_emb), dim=1)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        pos_score = torch.mean(-F.logsigmoid(pos_score))

        neg_score = torch.sum(torch.mul(neg_src_emb, neg_dst_emb), dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        neg_score = (
                torch.mean(-F.logsigmoid(-neg_score))
                * self.negative_size
                * self.neg_weight
        )

        return torch.mean(pos_score + neg_score)
