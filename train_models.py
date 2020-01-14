# -*- coding: utf-8 -*-

DESCRIPTION = """Given a directory for a single review article, train models for all of the seed/candidate splits in subdirectories (provided that seed/candidate sets have already been generated)."""

import sys, os, time, json
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

# from config import Config
from autoreview.config import Config
from autoreview import Autoreview
from autoreview.util import load_data_from_pickles
from autoreview.util import ItemSelector, DataFrameColumnTransformer, ClusterTransformer, AverageTfidfCosSimTransformer

from slugify import slugify
import pandas as pd

from custom_features import AbsoluteDistanceToSeedTransformer, EmbeddingSimilarityTransformer

def get_timestamp():
    return "{:%Y%m%d%H%M%S%f}".format(datetime.now())


def collect_paper_info(dirpath, id_field='UID', year_field='pub_year'):
    """Get paper info from a 'paper_info.json' file

    :dirpath: directory containing a 'paper_info.json' file
    :returns: dict containing some of the paper info. Returns None if it can't find the info

    """
    paperinfo_fpath = dirpath.joinpath('paper_info.json')
    if not paperinfo_fpath.exists():
        return None
    paperinfo = json.loads(paperinfo_fpath.read_text())
    return {
        'paper_id': paperinfo.get(id_field, None),
        'year': paperinfo.get(year_field, None)
    }

class TransformerSelection:

    """
    Specify different arrangements of features/transformers to use as inputs for the autoreview models
    """

    def __init__(self, switch_num=1, seed_papers=None, embeddings=None):
        self.switch_num = switch_num
        self.seed_papers = seed_papers
        self.embeddings = embeddings
        # potential features/transformers to use
        self.transformers = {
            'avg_distance_to_train': 
                ('avg_distance_to_train', ClusterTransformer(self.seed_papers)),
            'ef': 
                ('ef', DataFrameColumnTransformer('EF')),
            'efDist': 
                ('efDist', AbsoluteDistanceToSeedTransformer('year', seed_papers=self.seed_papers)),
            'year': 
                ('year', DataFrameColumnTransformer('year')),
            'yearDist': 
                ('yearDist', AbsoluteDistanceToSeedTransformer('year', seed_papers=self.seed_papers)),
            'avg_title_tfidf_cosine_similarity': 
                ('avg_title_tfidf_cosine_similarity', AverageTfidfCosSimTransformer(seed_papers=self.seed_papers, colname='title')),
            'title_embeddings':
                ('title_embeddings', EmbeddingSimilarityTransformer(seed_papers=self.seed_papers, embeddings=self.embeddings))
        }
        self._switch(self.switch_num)

    def _switch(self, switch_num=1):
        """
        dispatch method
        """
        switch = {
            1: self.network_and_title,
            2: self.network,
            3: self.title,
            4: self.network_title_year,
            5: self.clustering_only,
            6: self.network_efDist,
            7: self.network_efDist_title_yearDist,
            8: self.network_efDist_title,
            9: self.embeddings_only,
        }
        return switch[switch_num]()

    def network_and_title(self):
        """network and title features
        """
        self.name = "network_and_title_features"
        self.transformer_list = [
            self.transformers['avg_distance_to_train'],
            self.transformers['ef'],
            self.transformers['avg_title_tfidf_cosine_similarity']
        ]

    def network(self):
        """network features only
        """
        self.name = "network_features_only"
        self.transformer_list = [
            self.transformers['avg_distance_to_train'],
            self.transformers['ef'],
        ]

    def title(self):
        """title features only"""
        self.name = "title_features_only"
        self.transformer_list = [
            self.transformers['avg_title_tfidf_cosine_similarity']
        ]

    def network_title_year(self):
        """network, title, and year features"""
        self.name = "network_title_year_features"
        self.transformer_list = [
            self.transformers['avg_distance_to_train'],
            self.transformers['ef'],
            self.transformers['avg_title_tfidf_cosine_similarity'],
            self.transformers['year']
        ]

    def clustering_only(self):
        """clustering features only
        """
        self.name = "clustering_features_only"
        self.transformer_list = [
            self.transformers['avg_distance_to_train'],
        ]

    def network_efDist(self):
        """network features only, using EF distance from mean of seed papers
        """
        self.name = "network_features_only_efDist"
        self.transformer_list = [
            self.transformers['avg_distance_to_train'],
            self.transformers['efDist'],
        ]

    def network_efDist_title_yearDist(self):
        """features: network, EF distance from mean of seed papers, title, and year distance from mean of seed papers"""
        self.name = "network_efDist_title_yearDist"
        self.transformer_list = [
            self.transformers['avg_distance_to_train'],
            self.transformers['efDist'],
            self.transformers['avg_title_tfidf_cosine_similarity'],
            self.transformers['yearDist']
        ]

    def network_efDist_title(self):
        """features: network, EF distance from mean of seed papers, title"""
        self.name = "network_efDist_title"
        self.transformer_list = [
            self.transformers['avg_distance_to_train'],
            self.transformers['efDist'],
            self.transformers['avg_title_tfidf_cosine_similarity'],
        ]

    def embeddings_only(self):
        """features: title embeddings cosine similarity with average seed papers"""
        self.name = "embeddings_only"
        self.transformer_list = [
            self.transformers['title_embeddings'],
        ]

def get_embeddings(dirpath, all_papers, glob_pattern='embeddings*.pickle', id_colname='ID'):
    """get a dictionary mapping ID to embedding vector

    :dirpath: path to directory containing embeddings as pickled pandas Series
    :all_papers: dataframe of all papers
    :returns: dictionary

    """
    dirpath = Path(dirpath)
    g = dirpath.glob(glob_pattern)
    df = pd.DataFrame()
    logger.debug("getting embeddings for {} papers from directory: {}".format(len(all_papers), dirpath))
    for fpath in g:
        _embeddings = pd.read_pickle(fpath)
        _embeddings.name = 'embedding'
        subset = all_papers.merge(_embeddings, how='inner', left_on=id_colname, right_index=True)
        df = pd.concat([df, subset])
    embeddings_dict = df.set_index('ID')['embedding'].to_dict()
    return embeddings_dict

def run_train(paper_id, year, outdir, seed, transformer_scheme, embeddings=None):
    """Train models

    :paper_id: paper ID
    :year: publication year of the paper
    :outdir: directory with the seed/candidate/target papers
    :seed: random seed to use
    :transformer_scheme: integer mapping which features/transformers to use (see the TransformerSelection object definition)

    """
    a = Autoreview(outdir, random_seed=seed, use_spark=False)
    candidate_papers, seed_papers, target_papers = load_data_from_pickles(a.outdir)
    transformer_conf = TransformerSelection(transformer_scheme, seed_papers=seed_papers)
    if 'embedding' in transformer_conf.name:
        if embeddings is None:
            raise ValueError("'embeddings' must be specified for transformer scheme {} ({})".format(transformer_scheme, transformer_conf.name))
        all_papers = pd.concat([candidate_papers, seed_papers, target_papers]).drop_duplicates(subset=['ID'])
        embeddings_dict = get_embeddings(embeddings, all_papers)
        # reinitialize transformer_conf with embeddings_dict specified
        transformer_conf = TransformerSelection(transformer_scheme, seed_papers=seed_papers, embeddings=embeddings_dict)
    model_outdir = outdir.joinpath(transformer_conf.name)
    logger.debug("output directory is {}".format(model_outdir))
    if model_outdir.is_dir() and model_outdir.joinpath('._COMPLETE').exists():
        logger.debug("experiments for {} have already been completed. Skipping".format(model_outdir))
        return
    model_outdir.mkdir(exist_ok=True)
    log_file_handler = logging.FileHandler(model_outdir.joinpath('train_log_{}.log'.format(get_timestamp())))
    logger.debug("logging info to file: {}".format(log_file_handler.baseFilename))
    # logger.addHandler(log_file_handler)
    logging.getLogger('').addHandler(log_file_handler)
    logger.debug("number of seed papers: {}".format(len(seed_papers)))
    logger.debug("number of target papers: {}".format(len(target_papers)))
    logger.debug("number of candidate papers (haystack): {}".format(len(candidate_papers)))
    a.train_models(seed_papers=seed_papers, 
                    target_papers=target_papers, 
                    candidate_papers=candidate_papers, 
                    subdir=model_outdir.name,
                    transformer_list=transformer_conf.transformer_list,
                    year_lowpass=year)
    model_outdir.joinpath('._COMPLETE').touch()
    # logger.removeHandler(log_file_handler)
    logging.getLogger('').removeHandler(log_file_handler)

def main(args):
    dirpath = Path(args.dirname).resolve()
    if dirpath.is_dir():
        logger.debug("Output dirpath {} exists. Using this directory.".format(dirpath))
    else:
        raise ValueError("Output dirpath {} does not exist".format(dirpath))
    paper_info = collect_paper_info(dirpath)
    paper_id = paper_info['paper_id']
    year = paper_info['year']
    # year = get_year(paper_id, args.years, outdir)
    # logger.debug("year is {}".format(year))
    subdirs = [x for x in dirpath.glob('seed*') if x.is_dir()]
    subdirs.sort()
    for subdir in subdirs:
        seed = int(subdir.name[4:])
        logger.debug("\n\n\ntraining models for subdir {}".format(subdir))
        run_train(paper_id, year, subdir, seed, args.transformer_scheme)

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("dirname", help="input/output directory (should already exist, and contain seed/candidate paper splits in subfolders. should also contain a 'paper_info.json' file)")
    parser.add_argument("transformer_scheme", type=int, nargs='?', default=1, help="integer mapping which features/transformers to use (see the TransformerSelection object definition)")
    parser.add_argument("--embeddings", help="path to directory containing embeddings as pickled pandas Series")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    else:
        logger.setLevel(logging.INFO)
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
