# -*- coding: utf-8 -*-

DESCRIPTION = """Get a sample of WoS review articles with min and max cutoffs for number of references"""

import sys, os, time
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

import pandas as pd

def cutoff_num_references(df, min_cutoff, max_cutoff, refs_colname='num_citations'):
    """limit dataframe based on a minimum and maximum cutoff for number of references 

    :df: pandas dataframe
    :refs_colname: column name for the number of references
    :returns: pandas dataframe

    """
    return df[(df[refs_colname]>=min_cutoff)&(df[refs_colname]<max_cutoff)]

def main(args):
    logger.debug("reading input file: {}".format(args.review_ids))
    df = pd.read_csv(args.review_ids, sep='\t')
    logger.debug("input file has {} rows".format(len(df)))
    logger.debug ("limiting to articles with >= {} and < {} references".format(args.min_cutoff, args.max_cutoff))
    df = cutoff_num_references(df, args.min_cutoff, args.max_cutoff, refs_colname=args.refs_colname)
    logger.debug("there are {} such articles".format(len(df)))
    logger.debug("taking a sample of {} (using random seed {})".format(args.sample_size, args.random_seed))
    df = df.sample(n=args.sample_size, random_state=args.random_seed)
    logger.debug("saving to {}".format(args.output))
    df['UID'].to_csv(args.output, index=False, header=False)

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("review_ids", help="path to TSV file containing WoS IDs for review articles, and number of references for each one")
    parser.add_argument("output", help="path to output file (newline separated list of IDs, no header)")
    parser.add_argument("--min-cutoff", type=int, default=200, help="minimum cutoff for number of references")
    parser.add_argument("--max-cutoff", type=int, default=250, help="maximum cutoff for number of references")
    parser.add_argument("--sample-size", type=int, default=100, help="sample size (default: 100 papers)")
    parser.add_argument("--random-seed", type=int, default=999, help="random seed used for sampling (default: 999)")
    parser.add_argument("--refs-colname", default='num_citations', help="column name in the `review_ids` input file for the number of references (default: \"num_citations\")")
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
