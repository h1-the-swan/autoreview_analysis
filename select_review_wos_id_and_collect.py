# -*- coding: utf-8 -*-

DESCRIPTION = """Given an input file with one row per review article WoS ID, select a single row, and collect seed/candidate papers based on the references to that paper."""

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

from config import Config
from get_references_and_collect import PaperCollector

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
            if curr_row == rownum:
                line = line.strip().split(sep)
                return line[colnum]
            curr_row += 1
    return None

def run_collection(paper_id, args):
    """Run the collection for one paper

    :paper_id: paper ID
    :args: command line arguments

    """
    config = Config()
    try:
        pc = PaperCollector(config,
                    basedir = args.basedir,
                    paper_id = paper_id,
                    citations=args.citations,
                    papers=args.papers,
                    sample_size=args.sample_size,
                    id_colname=args.id_colname,
                    cited_colname=args.cited_colname)
        pc.main(args)
    finally:
        pc._config.teardown()



def main(args):
    fpath = Path(args.id_list)
    paper_id = get_wos_id(fpath, args.rownum)
    if paper_id is not None:
        run_collection(paper_id)
    else:
        raise RuntimeError("Could not get the paper ID")

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("id_list", help="tab-separated input file (with header) where the first column is the WoS ID to use")
    parser.add_argument("rownum", help="row number in the input file (`id_list`) to use (0 indexed)")
    parser.add_argument("basedir", help="output base directory (will be created if it doesn't exist)")
    parser.add_argument("--citations", help="citations data (to be read by spark)")
    parser.add_argument("--papers", help="papers/cluster data (to be read by spark)")
    parser.add_argument("--sample-size", type=int, default=200, help="number of articles to sample from the set to use to train the model (integer, default: 200)")
    parser.add_argument("--id-colname", default='UID', help="column name for paper id (default: \"UID\")")
    parser.add_argument("--cited-colname", default='cited_UID', help="column name for cited paper id (default: \"cited_UID\")")
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
