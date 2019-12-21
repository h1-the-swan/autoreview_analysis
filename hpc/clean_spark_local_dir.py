# -*- coding: utf-8 -*-

DESCRIPTION = """Delete temporary files that have not been used within a given time"""

import sys, os, time
from datetime import datetime
from pathlib import Path
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

def should_remove(path, seconds):
    """returns True if the file should be removed

    :path: Path to file
    :seconds: number of seconds threshold
    :returns: boolean

    """
    now = datetime.now().timestamp()
    st = path.stat()
    ctime = st.st_ctime
    if now - ctime > args.seconds:
        return True
    else:
        return False

def test_dry_run(to_remove):
    """As a dry run, don't erase any files, but instead write to a file the expected files to be deleted

    :to_remove: list of Path objects

    """
    out = Path('./clean_local_dir_TEST.txt')
    with out.open('w') as outf:
        for p in to_remove:
            outf.write("{}\n".format(p))

def main(args):
    basedir = Path(args.basedir).resolve()
    if not basedir.is_dir():
        raise ValueError("{} is not a valid directory".format(basedir))
    to_remove = []
    for p in basedir.rglob("*"):
        if p.is_file():
            if should_remove(p, args.seconds):
                to_remove.append(p)
    logger.debug("identified {} files to be deleted".format(len(to_remove)))
    test_dry_run(to_remove)

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("basedir", help="directory with temporary files to search through")
    parser.add_argument("seconds", type=int, default=3600, help="delete files that have not been modified within this number of seconds (using Unix `ctime`). default: 3600 (1 hour)")
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
