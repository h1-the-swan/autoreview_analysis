# -*- coding: utf-8 -*-

DESCRIPTION = """Go through subdirectories and collect paper info for the original review articles (title, pub date, etc.). Save this as a json file in the subdirectory"""

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

from slugify import slugify
from autoreview.config import Config
from autoreview.util import load_pandas_dataframe, load_spark_dataframe
from get_references_and_collect import PaperCollector

class BaseDirectoryNotFound(Exception):
    pass

class DirectoryNotFound(Exception):
    pass

class PaperInfoSaver:

    def __init__(self, basedir, paper_info):
        """

        :basedir: Path to base directory
        :paper_info: pandas Series (one row of a DataFame) for a single paper

        """
        self.basedir = Path(basedir).resolve()
        self.paper_info = paper_info
        self.paper_id = self.paper_info['UID']
        paper_id_slug = slugify(self.paper_id, lowercase=False)
        self.outdir = self.basedir.joinpath(paper_id_slug)
        if self.outdir.is_dir():
            logger.debug("Found directory {}. Using this directory.".format(self.outdir))
        else:
            raise DirectoryNotFound("Expected directory {} not found".format(self.outdir))

        self._save_paper_info()

    def _save_paper_info(self):
        """Save info (title, publication year, etc.) about the review paper to a JSON file

        """
        outfpath = self.outdir.joinpath('paper_info.json')
        logger.debug('saving paper info for this paper ({}) to {}'.format(self.paper_id, outfpath))
        self.paper_info.to_json(outfpath)
        

def get_all_wos_ids(fpath, sep='\t', header=True, col_num=0):
    """Get the WoS ID to use

    :fpath: Path object for the file containing the WoS IDs
    :row_num: row number to use
    :col_num: column number (0-indexed) for the column that contains WoS IDs (default: 0)
    :returns: WoS ID (string)

    """
    with fpath.open() as f:
        curr_row = 0
        paper_ids = []
        for i, line in enumerate(f):
            if i == 0 and header is True:
                continue
            line = line.strip().split(sep)
            paper_ids.append(line[col_num])
            curr_row += 1
    return paper_ids

def main(args):
    basedir = Path(args.basedir).resolve()
    if not basedir.is_dir():
        raise BaseDirectoryNotFound("base directory {} not found".format(basedir))
    fpath_ids = Path(args.id_list).resolve()
    if not fpath_ids.exists():
        raise RuntimeError("file {} does not exist".format(fpath_ids))
    header = not args.no_header
    paper_ids = get_all_wos_ids(fpath_ids, header=header)
    config = Config(spark_mem='100g')
    config._spark = config.load_spark_session(mem=config.spark_mem,
                                                additional_conf=[('spark.worker.cleanup.enabled', 'true')])
    spark = config._spark
    sdf_paper_ids = spark.createDataFrame(pd.DataFrame(paper_ids), schema='UID string')
    sdf_papers = load_spark_dataframe(args.papers, spark)
    df_paperinfo = sdf_papers.join(sdf_paper_ids, on='UID', how='inner').toPandas()
    if args.save is not None:
        logger.debug("saving paper info ({} rows) TSV to {}".format(len(df_paperinfo), args.save))
        df_paperinfo.to_csv(args.save, sep='\t')

    config.teardown()

    for _, row in df_paperinfo.iterrows():
        try:
            PaperInfoSaver(basedir, row)
        except DirectoryNotFound as e:
            logger.info("{} -- Skipping paper {}".format(e, row['UID']))

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("id_list", help="tab-separated input file (default: with header) where the first column is the WoS ID to use")
    parser.add_argument("basedir", help="output base directory. should have subdirectories that match the IDs in `id_list`")
    parser.add_argument("--no-header", action='store_true', help="specify that there is no header in the input `id_list` file. if this option is not specified, it assumed that there is a header.")
    parser.add_argument("--papers", help="papers/cluster data (to be read by spark)")
    parser.add_argument("--save", help="save all paper info to a file (tab-separated)")
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
