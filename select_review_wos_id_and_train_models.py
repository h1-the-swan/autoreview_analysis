# -*- coding: utf-8 -*-

DESCRIPTION = """Given an input file with one row per review article WoS ID, select a single row, and train models (provided that seed/candidate sets have already been generated)."""

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

# from config import Config
from autoreview.config import Config
from autoreview import Autoreview
from autoreview.util import load_data_from_pickles

from slugify import slugify
import pandas as pd

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
def get_year(paper_id, years_fname, id_colname='UID', years_colname='pub_date', sep='\t'):
    """Get the publication year

    :paper_id: WoS paper id
    :years_fname: filename for TSV file containing the years
    :returns: year (int)

    """
    df = pd.read_csv(years_fname, sep=sep)
    return df[df[id_colname]==paper_id][years_colname].iloc[0]

def run_train(paper_id, year, outdir, seed):
    """Train models

    :paper_id: paper ID
    :args: command line arguments

    """
    a = Autoreview(outdir, seed)
    candidate_papers, seed_papers, target_papers = load_data_from_pickles(a.outdir)
    a.train_models(seed_papers=seed_papers, target_papers=target_papers, candidate_papers=candidate_papers, year_lowpass=year)



def main(args):
    fpath = Path(args.id_list)
    paper_id = get_wos_id(fpath, args.rownum)
    if paper_id is None:
        raise RuntimeError("Could not get the paper ID")
    logger.debug("paper_id is {}".format(paper_id))
    year = get_year(paper_id, args.years)
    logger.debug("year is {}".format(year))
    basedir = Path(args.basedir).resolve()
    if basedir.is_dir():
        logger.debug("Output basedir {} exists. Using this directory.".format(basedir))
    else:
        raise ValueError("Output basedir {} does not exist".format(basedir))
    paper_id_slug = slugify(paper_id, lowercase=False)
    outdir = basedir.joinpath(paper_id_slug)
    if not outdir.is_dir():
        raise RuntimeError("Could not find directory {}".format(outdir))
    subdirs = [x for x in outdir.glob('seed*') if x.is_dir()]
    subdirs.sort()
    for subdir in subdirs:
        seed = int(subdir.name[4:])
        logger.debug("\n\n\ntraining models for subdir {}".format(subdir))
        run_train(paper_id, year, subdir, seed)

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    if os.environ.get('SLURM_JOB_ID'):
        logger.info('SLURM_JOB_ID: {}'.format(os.environ['SLURM_JOB_ID']))
        logger.info('Running on node: {}'os.environ.get('SLURM_JOB_NODELIST'))
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("id_list", help="tab-separated input file (with header) where the first column is the WoS ID to use")
    parser.add_argument("rownum", type=int, help="row number in the input file (`id_list`) to use (0 indexed)")
    parser.add_argument("basedir", help="output base directory (should already exist, and contain seed/candidate papers in a subfolder)")
    parser.add_argument("--years", required=True, help="path to TSV file containing the publication year for the papers")
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
