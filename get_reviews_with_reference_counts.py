# -*- coding: utf-8 -*-

DESCRIPTION = """Get WoS IDs for review articles, along with counts of their references, limiting to references for which we have clustering information."""

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

import pandas as pd
import numpy as np

from autoreview.config import Config
from autoreview.util import load_spark_dataframe

def main(args):
    config = Config()
    spark = config.spark

    id_colname = args.id_colname

    try:
        logger.debug("reading data from {}".format(args.reviews))
        sdf_reviews = load_spark_dataframe(args.reviews, spark)
        logger.debug("reviews dataframe has {} rows".format(sdf_reviews.count()))
        sdf_reviews = sdf_reviews.select([id_colname])

        logger.debug("loading spark dataframe from {}".format(args.papers))
        sdf_papers = load_spark_dataframe(args.papers, spark)
        logger.debug("papers dataframe has {} rows".format(sdf_papers.count()))
        logger.debug("dropping rows with no cluster information")
        sdf_papers = sdf_papers.dropna(subset=['cl'])
        sdf_papers.persist()
        logger.debug("after dropping rows, papers dataframe has {} rows".format(sdf_papers.count()))

        logger.debug("merging review dataframe and papers dataframe")
        sdf_reviews = sdf_reviews.join(sdf_papers, on=id_colname, how='inner')
        sdf_reviews.persist()
        logger.debug("merged dataframe has {} rows".format(sdf_reviews.count()))

        logger.debug("loading spark dataframe from {}".format(args.citations))
        sdf_citations = load_spark_dataframe(args.citations, spark)
        logger.debug("citations dataframe has {} rows".format(sdf_citations.count()))

        logger.debug("joining citations dataframe with papers dataframe IDs on both columns, to filter out dropped rows")
        sdf_papers_ids = sdf_papers.select([args.id_colname])
        sdf_citations = sdf_citations.join(sdf_papers_ids, on=args.id_colname, how='inner')
        sdf_citations = sdf_citations.join(sdf_papers_ids.withColumnRenamed(args.id_colname, args.cited_colname), on=args.cited_colname, how='inner')
        sdf_citations.persist()
        logger.debug("citations dataframe now has {} rows".format(sdf_citations.count()))

        columns_to_keep = [id_colname, 'doi', 'pub_date', 'title', 'title_source']
        sdf_reviews = sdf_reviews.select(columns_to_keep)
        logger.debug("merging reviews with citations and counting references")
        sdf_reviews = sdf_reviews.join(sdf_citations, on=id_colname, how='inner')
        sdf_reviews = sdf_reviews.groupby(id_colname).count()
        sdf_reviews = sdf_reviews.withColumnRenamed('count', 'num_references')

        logger.debug("converting to pandas dataframe")
        df_reviews = sdf_reviews.toPandas()
        logger.debug("dataframe has {} rows".format(len(df_reviews)))
    finally:
        config.teardown()

    logger.debug("sorting dataframe")
    df_reviews = df_reviews.sort_values('num_references', ascending=False)
    logger.debug("saving to {}".format(args.output))
    df_reviews.to_csv(args.output, sep='\t', index=False)


if __name__ == "__main__":
    total_start = timer()
    # logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("-o", "--output", help="output file to be written (TSV)")
    parser.add_argument("--reviews", help="path to a TSV file with review papers")
    parser.add_argument("--papers", help="path to papers/cluster data (to be read by spark)")
    parser.add_argument("--citations", help="citations data (to be read by spark)")
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
