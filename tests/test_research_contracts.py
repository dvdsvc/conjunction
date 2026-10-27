"""Small numerical contracts for the retained graph/ranking implementation."""
from __future__ import annotations
import unittest
import numpy as np
import scipy.sparse as sp
import torch
from research.temporal.fixes import precompute_v15_fix as pre
from research.temporal.fixes import train_temporal_v15_fix as train

class FeatureContracts(unittest.TestCase):
    def test_pair_feature_width_and_jaccard_slots(self):
        a=sp.csr_matrix(np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]],dtype=np.float32))
        ppr=np.eye(4,dtype=np.float32)
        base=pre.compute_all_pair_features(np.array([0]),np.array([1]),ppr,a,pre.fast_clustering_coefficients(a))
        gravity=pre.compute_gravity_features(np.array([0]),np.array([1]),a,np.asarray(a.sum(axis=1)).ravel())
        self.assertEqual(np.column_stack([base,gravity]).shape,(1,21))
        self.assertAlmostEqual(float(base[0,4]),1/3,places=6)
        self.assertEqual(base[0,4],base[0,14])
        self.assertTrue(np.isfinite(np.column_stack([base,gravity])).all())

    def test_relevance_boundaries(self):
        observed=pre.dist_to_relevance(np.array([0,1,1.01,2,5,10,11],dtype=np.float32))
        np.testing.assert_array_equal(observed,[4,4,3,3,2,1,0])

    def test_endpoint_retention_under_degree_truncation(self):
        neighborhoods=[np.arange(6),np.arange(6),np.arange(6),np.arange(6),np.arange(6),np.arange(6)]
        nodes,u,v=pre.build_merged_subgraph_fast(0,1,neighborhoods,np.array([0,0,90,100,110,120]),max_nodes=4)
        self.assertEqual(int(nodes[u]),0)
        self.assertEqual(int(nodes[v]),1)
        self.assertEqual(len(nodes),4)

class RankingContracts(unittest.TestCase):
    def test_weighted_ranking_loss_is_joint_permutation_invariant(self):
        scores=torch.tensor([[.2,1.3,-.4,.7]])
        labels=torch.tensor([[1.,4.,0.,2.]])
        permutation=torch.tensor([2,0,3,1])
        self.assertTrue(torch.allclose(train.lambdarank_loss(scores,labels),train.lambdarank_loss(scores[:,permutation],labels[:,permutation]),atol=1e-7))

    def test_equal_labels_have_zero_finite_gradient(self):
        scores=torch.tensor([[.2,1.3,-.4]],requires_grad=True)
        loss=train.lambdarank_loss(scores,torch.ones_like(scores))
        loss.backward()
        self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(float(scores.grad.abs().sum()),0.)

    def test_ndcg_distinguishes_known_rankings(self):
        labels=np.array([4.,2.,0.]);ideal=np.array([3.,2.,1.]);bad=ideal[::-1].copy()
        self.assertAlmostEqual(float(train.compute_ndcg_at_k(ideal,labels,3)),1.)
        self.assertLess(float(train.compute_ndcg_at_k(bad,labels,3)),1.)

if __name__=='__main__':unittest.main()
