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

