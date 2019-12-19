# -*- coding: utf-8 -*-

DESCRIPTION = """Go through the old analysis results to collect WoS IDs for all of the review articles analyzed previously."""

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

def get_wos_id(path_to_json):
    """Get the WoS ID and MAG ID from a paperinfo json file

    :returns: tuple (WoS ID, MAG ID)

    """
    data = json.loads(path_to_json.read_text())
    return (data.get('wos_id'), data.get('mag_id'))

def main(args):
    basedir = Path(args.basedir)
    logger.debug("searching in {}".format(basedir))
    subdirs = basedir.glob('review_*')
    wos_ids = []
    for subdir in subdirs:
        if subdir.is_dir():
            paperinfo_path = list(subdir.glob('*paperinfo*.json'))
            if paperinfo_path:
                assert len(paperinfo_path) == 1
                paperinfo_path = paperinfo_path[0]
                wos_ids.append(get_wos_id(paperinfo_path))
    logger.debug("{} records found".format(len(wos_ids)))
    out = Path(args.output)
    logger.debug("saving to {}".format(out))
    sep = '\t'
    with out.open(mode='w') as outf:
        # write header
        outf.write(f"wos_id{sep}mag_id\n")
        # write data
        for wos_id, mag_id in wos_ids:
            outf.write(f"{wos_id}{sep}{mag_id}\n")

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("basedir", help="base directory to search through")
    parser.add_argument("output", help="path to output (TSV) file")
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
