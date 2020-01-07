# -*- coding: utf-8 -*-

DESCRIPTION = """Given an input file with one row per review article WoS ID, select a single row, and train models (provided that seed/candidate sets have already been generated)."""

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

def get_timestamp():
    return "{:%Y%m%d%H%M%S%f}".format(datetime.now())


def get_wos_id(fpath, row_num, sep='\t', header=True, col_num=0):
    """Get the WoS ID to use

    :fpath: Path object for the file containing the WoS IDs
    :row_num: row number to use
    :col_num: column number (0-indexed) for the column that contains WoS IDs (default: 0)
    :returns: WoS ID (string)

    """
    with fpath.open() as f:
        curr_row = 0
        for i, line in enumerate(f):
            if i == 0 and header is True:
                continue
            if curr_row == row_num:
                line = line.strip().split(sep)
                return line[col_num]
            curr_row += 1
    return None

# def run_collection(paper_id, args):
#     """Run the collection for one paper
#
#     :paper_id: paper ID
#     :args: command line arguments
#
#     """
#     use_spark = not args.no_spark
#     config = Config(spark_mem='100g')
#     if use_spark is True:
#         config._spark = config.load_spark_session(mem=config.spark_mem,
#                                                     additional_conf=[('spark.worker.cleanup.enabled', 'true')])
#     try:
#         pc = PaperCollector(config,
#                     basedir = args.basedir,
#                     paper_id = paper_id,
#                     citations=args.citations,
#                     papers=args.papers,
#                     sample_size=args.sample_size,
#                     id_colname=args.id_colname,
#                     cited_colname=args.cited_colname,
#                     use_spark=use_spark)
#         pc.main(args)
#     finally:
#         pc._config.teardown()
#

def get_year(paper_id=None, years_fname=None, dirpath=None):
    """Get year of the paper from a TSV file. If no TSV file is provided, look for a paper_info.json file with a 'pub_year' field instead.

    Must supply as arguments EITHER paper_id and years_fname, OR dirpath. If all three are supplied, the TSV will be used

    :paper_id: WoS paper id
    :years_fname: filename for TSV file containing the year
    :dirpath: directory Path (containing a 'paper_info.json' file)
    :returns: year (int), or None if the year is unavailable

    """
    if years_fname is not None:
        return _get_year_from_csv(paper_id, years_fname)
    if dirpath is not None:
        return _get_year_from_paperinfo_json(dirpath)
    return None

def _get_year_from_csv(paper_id, years_fname, id_colname='UID', years_colname='pub_date', sep='\t'):
    """Get the publication year

    :paper_id: WoS paper id
    :years_fname: filename for TSV file containing the years
    :returns: year (int)

    """
    df = pd.read_csv(years_fname, sep=sep)
    date = df[df[id_colname]==paper_id][years_colname].iloc[0]
    date = pd.to_datetime(str(date))
    return date.year

def _get_year_from_paperinfo_json(dirpath, field='pub_year'):
    """Get the publication year from a 'paper_info.json' file

    :dirpath: directory containing a 'paper_info.json' file
    :returns: year (int), or None if the year is unavailable

    """
    paperinfo_fpath = dirpath.joinpath('paper_info.json')
    if not paperinfo_fpath.exists():
        return None
    paperinfo = json.loads(paperinfo_fpath.read_text())
    if field in paperinfo:
        return paperinfo[field]
    else:
        return None

class TransformerSelection:

    """
    Specify different arrangements of features/transformers to use as inputs for the autoreview models
    """

    def __init__(self, switch_num=1, seed_papers=None):
        self.switch_num = switch_num
        self.seed_papers = seed_papers
        # potential features/transformers to use
        self.transformers = {
            'avg_distance_to_train': 
                ('avg_distance_to_train', ClusterTransformer(self.seed_papers)),
            'ef': 
                ('ef', DataFrameColumnTransformer('EF')),
            'year': 
                ('year', DataFrameColumnTransformer('year')),
            'avg_title_tfidf_cosine_similarity': 
                ('avg_title_tfidf_cosine_similarity', AverageTfidfCosSimTransformer(seed_papers=self.seed_papers, colname='title')),
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

def run_train(paper_id, year, outdir, seed, transformer_scheme):
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
    model_outdir = outdir.joinpath(transformer_conf.name)
    logger.debug("output directory is {}".format(model_outdir))
    if model_outdir.is_dir() and model_outdir.joinpath('._COMPLETE').exists():
        logger.debug("experiments for {} have already been completed. Skipping".format(model_outdir))
        return
    model_outdir.mkdir(exist_ok=True)
    log_file_handler = logging.FileHandler(model_outdir.joinpath('train_log_{}.log'.format(get_timestamp())))
    logger.debug("logging info to file: {}".format(log_file_handler.baseFilename))
    logger.addHandler(log_file_handler)
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
    logger.removeHandler(log_file_handler)

def main(args):
    fpath = Path(args.id_list)
    header = not args.no_header
    paper_id = get_wos_id(fpath, args.rownum, header=header)
    if paper_id is None:
        raise RuntimeError("Could not get the paper ID")
    logger.debug("paper_id is {}".format(paper_id))
    basedir = Path(args.basedir).resolve()
    if basedir.is_dir():
        logger.debug("Output basedir {} exists. Using this directory.".format(basedir))
    else:
        raise ValueError("Output basedir {} does not exist".format(basedir))
    paper_id_slug = slugify(paper_id, lowercase=False)
    outdir = basedir.joinpath(paper_id_slug)
    if not outdir.is_dir():
        raise RuntimeError("Could not find directory {}".format(outdir))
    year = get_year(paper_id, args.years, outdir)
    logger.debug("year is {}".format(year))
    subdirs = [x for x in outdir.glob('seed*') if x.is_dir()]
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
    parser.add_argument("id_list", help="tab-separated input file (with header) where the first column is the WoS ID to use")
    parser.add_argument("rownum", type=int, help="row number in the input file (`id_list`) to use (0 indexed)")
    parser.add_argument("basedir", help="output base directory (should already exist, and contain seed/candidate papers in a subfolder)")
    parser.add_argument("transformer_scheme", type=int, default=1, help="integer mapping which features/transformers to use (see the TransformerSelection object definition)")
    parser.add_argument("--years", help="path to TSV file containing the publication year for the papers.")
    parser.add_argument("--no-header", action='store_true', help="specify that there is no header in the input `id_list` file. if this option is not specified, it assumed that there is a header.")
    # parser.add_argument("--citations", help="citations data (to be read by spark)")
    # parser.add_argument("--papers", help="papers/cluster data (to be read by spark)")
    # parser.add_argument("--sample-size", type=int, default=200, help="number of articles to sample from the set to use to train the model (integer, default: 200)")
    # parser.add_argument("--id-colname", default='UID', help="column name for paper id (default: \"UID\")")
    # parser.add_argument("--cited-colname", default='cited_UID', help="column name for cited paper id (default: \"cited_UID\")")
    # parser.add_argument("--no-spark", action='store_true', help="don't use spark to collect candidate papers")
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
