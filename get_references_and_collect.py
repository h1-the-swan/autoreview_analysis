# -*- coding: utf-8 -*-

DESCRIPTION = """Given a paper ID, get all of the outgoing references, then collect seed, target, and candidate sets for different random seeds based on those references."""

import sys, os, time, shutil
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

from slugify import slugify
from autoreview import Autoreview
from autoreview.util import prepare_directory, load_spark_dataframe, load_pandas_dataframe

from autoreview.config import Config

import pandas as pd

class PaperCollector(object):

    def __init__(self, config, basedir, paper_id, citations, papers, sample_size, id_colname, cited_colname, use_spark=True):
        """

        :config: Config object

        """
        self._config = config
        self.spark = config.spark
        self.use_spark = use_spark
        # TODO:
        # if self.use_spark is True:
        #     self.spark = config.spark
        self.basedir = basedir
        self.paper_id = paper_id
        self.citations = citations
        self.papers = papers
        self.sample_size = sample_size
        self.id_colname = id_colname
        self.cited_colname = cited_colname

        self.basedir = Path(self.basedir).resolve()
        if self.basedir.is_dir():
            logger.debug("Output basedir {} exists. Using this directory.".format(self.basedir))
        else:
            logger.debug("Output basedir {} does not exist. Creating it...".format(self.basedir))
            self.basedir.mkdir(parents=True)
        paper_id_slug = slugify(paper_id, lowercase=False)
        self.outdir = self.basedir.joinpath(paper_id_slug)
        if self.outdir.is_dir():
            logger.debug("Output directory {} exists. Using this directory.".format(self.outdir))
        else:
            logger.debug("Output directory {} does not exist. Creating it...".format(self.outdir))
            self.outdir.mkdir()

        self.df_citations = None
        self.df_papers = None

    def get_reference_ids(self, paper_id, use_spark=True, df_citations=None):
        """get references

        :returns: list of paper IDs

        """
        if use_spark is True:
            return self._get_reference_ids_spark(paper_id)
        if df_citations is None or not isinstance(df_citations, pd.DataFrame):
            raise ValueError('`df_citations` must be supplied to this method, and it must be a pandas dataframe')
        reference_ids = df_citations[df_citations[self.id_colname]==paper_id]
        reference_ids = reference_ids[self.cited_colname].tolist()
        return reference_ids

    def _get_reference_ids_spark(self, paper_id):
        """Use spark to get references

        :returns: list of paper IDs

        """
        sdf_citations = load_spark_dataframe(self.citations, self.spark)
        logger.debug("collecting references of paper ID: {}".format(paper_id))
        reference_ids = sdf_citations[sdf_citations[self.id_colname]==paper_id]
        reference_ids = reference_ids.toPandas()
        reference_ids = reference_ids[self.cited_colname].tolist()
        return reference_ids

    def main(self, args):
        logfile = self.outdir.joinpath('collect.log').open(mode='a', buffering=1)
        logfile.write("{} - starting collection\n".format(datetime.now()))
        logfile.write(" ".join(sys.argv))
        logfile.write("\n")
        try:
            # load citation data (if not using spark)
            if self.use_spark is False:
                logger.debug('loading citation data...')
                start = timer()
                self.df_citations = load_pandas_dataframe(self.citations)
                logger.debug('done loading citation data. took {}'.format(format_timespan(timer()-start)))
            else:
                self.df_citations = None
            reference_ids = self.get_reference_ids(self.paper_id, self.use_spark, self.df_citations)
            outfile = self.outdir.joinpath('reference_ids.csv')
            logger.debug("writing {} reference ids to {}".format(len(reference_ids), outfile))
            logfile.write("writing {} reference ids to {}\n".format(len(reference_ids), outfile))
            outfile.write_text("\n".join(reference_ids))

            # load paper data (if not using spark)
            if self.use_spark is False:
                logger.debug('loading paper data...')
                start = timer()
                self.df_papers = load_pandas_dataframe(self.papers)
                logger.debug('done loading papers data. took {}'.format(format_timespan(timer()-start)))
            else:
                self.df_papers = None

            # collect paper sets for each random seed (1-5):
            for random_seed in range(1, 6):
                this_outdir = self.outdir.joinpath('seed{:03d}'.format(random_seed))
                if this_outdir.is_dir():
                    if this_outdir.joinpath('._COMPLETE').exists():
                        logfile.write("directory {} exists and collection has already completed. skipping.\n".format(this_outdir))
                        logger.debug("skipping directory {}".format(this_outdir))
                        continue
                    else:
                        logfile.write("directory {} exists but collection did not complete. deleting this directory and continuing with collection.\n".format(this_outdir))
                        shutil.rmtree(this_outdir)
                this_outdir.mkdir()
                this_start = timer()
                logger.debug("collecting papers to go in {}".format(this_outdir))
                logfile.write("collecting papers to go in {}\n".format(this_outdir))
                a = Autoreview(id_list=reference_ids,
                                citations=self.citations,
                                papers=self.papers,
                                outdir=this_outdir,
                                sample_size=self.sample_size,
                                random_seed=random_seed,
                                id_colname=self.id_colname,
                                cited_colname=self.cited_colname,
                                config=self._config,
                                use_spark=self.use_spark,
                                citations_data_preloaded=self.df_citations,
                                paper_data_preloaded=self.df_papers)
                a.get_papers_2_degrees_out(use_spark=self.use_spark)
                this_outdir.joinpath('._COMPLETE').touch()
                logfile.write("collecting papers for {} took {}\n".format(this_outdir, timer()-this_start))

            logfile.write("\n{} - COLLECTION COMPLETED\n".format(datetime.now()))
        finally:
            logfile.close()


def main(args):
    config = Config()
    try:
        pc = PaperCollector(config,
                    basedir = args.basedir,
                    paper_id = args.paper_id,
                    citations=args.citations,
                    papers=args.papers,
                    sample_size=args.sample_size,
                    id_colname=args.id_colname,
                    cited_colname=args.cited_colname)
        pc.main(args)
    finally:
        pc._config.teardown()


if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("basedir", help="output base directory (will be created if it doesn't exist)")
    parser.add_argument("paper_id", help="paper ID to start with")
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
