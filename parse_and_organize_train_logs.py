# -*- coding: utf-8 -*-

DESCRIPTION = """
Due to a bug, not all of the output logs ended up in the right place.

This script parses the master logs and saves a dictionary (as a pickle file) of 
directory -> log fragment, ready to parse with autoreview_analysis.parse_train_log()
"""

import sys, os, time, re, pickle
from pathlib import Path
from collections import defaultdict
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

pattern_dirname = re.compile(r"output directory is (\S+)\s")

def get_dirpath_from_log_fragment(log_fragment, basedir, pattern=pattern_dirname):
    m = pattern_dirname.search(log_fragment)
    if not m:
        return None
    output_dirpath = m.group(1)
    # align with basedir
    output_dirpath = output_dirpath[output_dirpath.find(str(basedir)):]
    output_dirpath = Path(output_dirpath)
    return output_dirpath

def process_one_log(fpath, split_phrase, basedir):
    """process one train log

    :fpath: Path to train log file
    :returns: TODO

    """
    log_txt = fpath.read_text()
    log_split = log_txt.split(split_phrase)
    if len(log_split) <= 1:
        # no data in this log, or not properly formatted
        return (None, None)
    for log_fragment in log_split[1:]:
        if 'skipping' in log_fragment.lower():
            continue
        if 'CANCELLED' in log_fragment:
            continue
        log_fragment = split_phrase + log_fragment
        output_dirpath = get_dirpath_from_log_fragment(log_fragment, basedir)
        if output_dirpath:
            yield (output_dirpath, log_fragment)

def main(args):
    basedir = Path(args.basedir)
    if not basedir.is_dir():
        raise FileNotFoundError("base directory doesn't exist: {}".format(basedir))
    outfpath = Path(args.output)
    logger.debug("using base directory {}".format(basedir))
    # glob pattern to find all of the log files within the base directory (recursively)
    glob_pattern = "*train_models*.out"
    # phrase to split the log text on, to separate the sections of the log files
    split_phrase = "training models for subdir"

    train_fpaths = basedir.rglob(glob_pattern)
    data = {}
    duplicates_found = []
    mismatches = defaultdict(list)
    for fp in train_fpaths:
        for output_dirpath, log_fragment in process_one_log(fp, split_phrase, basedir):
            if output_dirpath:
                if output_dirpath in data:
                    # raise ValueError("error while processing {}: duplicate entry found, for output_dirpath: {}".format(fp, output_dirpath))
                    if data[output_dirpath] == log_fragment:
                        dup_type = 'true_duplicate'
                    else:
                        dup_type = 'mismatch'
                        if output_dirpath not in mismatches:
                            mismatches[output_dirpath].append(data[output_dirpath])
                        mismatches[output_dirpath].append((fp, log_fragment))
                    duplicates_found.append((output_dirpath, dup_type))
                    pass # ignore and overwrite for now
                data[output_dirpath] = log_fragment
    logger.debug("done collecting data")

    logger.debug("saving {} log fragments to output dictionary: {}".format(len(data), outfpath))
    outfpath.write_bytes(pickle.dumps(data))

    logger.debug("number of duplicates found (these were overwritten): {}".format(len(duplicates_found)))
    logger.debug("{} of these were mismatches".format(len([x for x in duplicates_found if x[1] == 'mismatch'])))
    outfpath_dups = outfpath.with_suffix('.MISMATCHES.pickle')
    logger.debug('saving these mismatches to {}'.format(outfpath_dups))
    outfpath_dups.write_bytes(pickle.dumps(mismatches))

if __name__ == "__main__":
    total_start = timer()
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("basedir", help="base directory in which to search (recursively) for train logs")
    parser.add_argument("output", help="output file (pickle)")
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
