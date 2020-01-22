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

from train_models import run_train

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

def main(args):
    fpath = Path(args.id_list)
    header = not args.no_header
    save_best = not args.no_save_best
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
        run_train(paper_id, year, subdir, seed, args.transformer_scheme, args.embeddings, save_best=save_best, force_rerun=args.force_rerun)

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
    parser.add_argument("--no-save-best", action='store_true', help="do not save the best model to pickle file")
    parser.add_argument("--embeddings", help="path to directory containing embeddings as pickled pandas Series")
    parser.add_argument("--force-rerun", action='store_true', help="force rerun of training for models that have already completed")
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
