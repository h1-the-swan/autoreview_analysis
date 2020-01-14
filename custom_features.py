# -*- coding: utf-8 -*-

DESCRIPTION = """Custom features to use for autoreview models"""

import sys, os, time
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

class AbsoluteDistanceToSeedTransformer(BaseEstimator, TransformerMixin):
    """
    For a continuous variable, calculate the absolute deviation from the seed papers' mean.
    """
    def __init__(self, colname, seed_papers=None):
        self.seed_papers = seed_papers
        self.colname = colname
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, df):
        seed_mean = self.seed_papers[self.colname].mean()
        abs_dist = df[self.colname].apply(lambda x: abs(x - seed_mean))
        return abs_dist.to_numpy().reshape(-1, 1)

class EmbeddingSimilarityTransformer(BaseEstimator, TransformerMixin):
    """
    cosine similarity between the embedding vector and the average vector of the seed papers
    """
    def __init__(self, seed_papers=None, embeddings=None, id_colname='ID'):
        """
        :seed_papers: dataframe of seed papers
        :embeddings: dictionary mapping paper ID to embedding vector. should include all of the seed and test papers
        """
        self.seed_papers = seed_papers
        self.embeddings = embeddings
        self.id_colname = id_colname

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        seed_embeddings = [self.embeddings[id_] for id_ in self.seed_papers[self.id_colname].values if id_ in self.embeddings]
        seed_embeddings = np.array(seed_embeddings)
        test_embeddings = [self.embeddings[id_] for id_ in df[self.id_colname].values if id_ in self.embeddings]
        test_embeddings = np.array(test_embeddings)
        avg_seed_embeddings = seed_embeddings.mean(axis=0).reshape(1, -1)
        csims = cosine_similarity(test_embeddings, avg_seed_embeddings)
        return csims

