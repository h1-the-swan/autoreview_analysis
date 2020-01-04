# -*- coding: utf-8 -*-

DESCRIPTION = """Classes to help with analysis of Autoreview experiments"""

import sys, os, time, re
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

pattern_wos_id_slug = re.compile(r"/([^/]+)/reference_ids\.csv")
# pattern_correctly_predicted = re.compile(r"False\s+?(\d+)\nTrue\s+?(\d+)") # group 1 is incorrect, group 2 is correct
pattern_predicted_false = re.compile(r"False\s+?(\d+)\n")
pattern_predicted_true = re.compile(r"True\s+?(\d+)\n")
pattern_log_line_split = re.compile(r"\d\d.*? : ")
pattern_paper_set_sizes = re.compile(r"after year.*?size of haystack: (\d+).*?(\d+) target papers\. (\d+) of these", 
                                     flags=re.DOTALL) # group 1 is num_candidates, group 2 is num_target, group 3 is num_target_in_candidates

def get_logfname(basedir, n, logtype='collect'):
    return basedir.joinpath("select_wos_id_and_{}_{}.out".format(logtype, n))

class NoDataError(Exception):
    pass

class LogFormatError(Exception):
    pass

class AutoreviewAnalysis:
    def __init__(self, basedir, row_num, parent=None):
        self.basedir = Path(basedir)
        self.row_num = row_num
        self.parent = parent  # AutoreviewAnalysisCollection object
        self.collect_log_fpath = get_logfname(self.basedir, self.row_num, 'collect')
        if not self.collect_log_fpath.exists():
            raise NoDataError("{} does not exist".format(self.collect_log_fpath))
        self.collect_log = self.collect_log_fpath.read_text()
        self.wos_id_slug = self.get_wos_id_slug()
        self.dirpath = self.basedir.joinpath(self.wos_id_slug)
        if not self.dirpath.exists():
            raise NoDataError("directory {} does not exist".format(self.dirpath))
        self.reference_ids = self.get_reference_ids()
        subdirs = [x for x in self.dirpath.glob('seed*') if x.is_dir()]
        subdirs.sort()
        self.subdirs = []
        for subdir in subdirs:
            subdir = AutoreviewAnalysisSubdir(subdir, parent=self)
            subdir.parent = self
            self.subdirs.append(subdir)
        self.all_complete = all(subdir.is_complete for subdir in self.subdirs)
        
        self.train_log_fpath = get_logfname(self.basedir, self.row_num, 'train_models')
        if self.train_log_fpath.exists():
            self.train_log = self.train_log_fpath.read_text()
            train_log_split = self.train_log.split('training models for subdir')
            for log_fragment in train_log_split:
                seed = re.search(r"seed(\d\d\d)", log_fragment)
                if seed is None:
                    continue
                seed = int(seed.group(1))
                subdir_idx = seed - 1
                self.subdirs[subdir_idx].train_log_fragment = log_fragment
                try:
                    self.subdirs[subdir_idx].parse_train_log()
                except LogFormatError:
                    pass
        else:
            self.train_log_fpath = None
            self.train_log = None
        
    def get_wos_id_slug(self):
        return pattern_wos_id_slug.search(self.collect_log).group(1)
    
    def get_reference_ids(self):
        fpath = self.dirpath.joinpath('reference_ids.csv')
        if not fpath.exists():
            raise NoDataError("file {} does not exist".format(fpath))
        reference_ids = []
        with fpath.open() as f:
            for line in f:
                reference_ids.append(line.strip())
        return reference_ids

    def top_models(self):
        """get top models for all subdirs as a list
        :returns: list of dictionaries representing top models

        """
        return [x.top_model() for x in self.subdirs if x.models]
    
        
class AutoreviewAnalysisSubdir:
    def __init__(self, subdir, parent=None):
        self.subdir = subdir
        self.parent = parent  # AutoreviewAnalysis object
        self.seed = int(subdir.name[4:])
        self.is_complete = self.check_complete()
        self.candidate_papers = None
        self.seed_papers = None
        self.target_papers = None
        self.models = []
        
    def check_complete(self):
        return self.subdir.joinpath('._COMPLETE').exists()
    
    def load_paper_sets(self, force=False):
        if self.candidate_papers is None or force is True:
            self.candidate_papers, self.seed_papers, self.target_papers = load_data_from_pickles(self.subdir)
        pass

    def _actual_seed_size(self):
        fpath = self.subdir.joinpath('seed_papers.pickle')
        seed_papers = pd.read_pickle(fpath)
        return len(seed_papers)
    
    def parse_train_log(self):
        log_split = self.train_log_fragment.split('========Pipeline')
        sizes = pattern_paper_set_sizes.findall(log_split[0])
        if not sizes:
            raise LogFormatError("set sizes pattern not found in log fragment for subdir {}".format(self.subdir))
        sizes = [int(x) for x in sizes[0]]
        self.num_candidates, self.num_target, self.num_target_in_candidates = sizes
        self.models = [] # list of dicts
        for model_txt in log_split[1:]:
            clf = pattern_log_line_split.split(model_txt)[2]
            clf_type = clf[:clf.find("(")]
            
            m = pattern_predicted_false.search(model_txt)
            score_false = int(m.group(1)) if m else 0
            
            m = pattern_predicted_true.search(model_txt)
            score_true = int(m.group(1)) if m else 0
            score = score_true / self.num_target
            this_model = {
                'clf': clf,
                'clf_type': clf_type,
                'num_correctly_predicted': score_true,
                'score_correctly_predicted': score
            }
            self.models.append(this_model)
            
    def top_model(self):
        if self.models:
            return sorted(self.models, key=lambda x: x['score_correctly_predicted'])[-1]
        return None

class AutoreviewAnalysisCollection:

    """collection of review papers (each with seed/candidate splits, which in turn each have trained models)"""

    def __init__(self, name):
        self.name = name
        # TODO

        

class AutoreviewAnalysisMultipleSeeds:
    def __init__(self, basedir):
        """

        :basedir: base directory containing multiple directories with seed/candidate paper sets

        """
        pattern_idx = re.compile(r"_(\d+)\.out")
        self.basedir = basedir
        self.dirpaths = [x for x in self.basedir.iterdir() if x.is_dir()]
        self.dirpaths.sort()
        # initialize empty dictionaries
        self.reviews = {x.name: [] for x in self.dirpaths}
        self.reviews_with_train = {x.name: [] for x in self.dirpaths}
        self.no_data_error = {x.name: [] for x in self.dirpaths}
        self.log_format_error = {x.name: [] for x in self.dirpaths}

        for dirpath in self.dirpaths:
            for logfpath in dirpath.glob("select*and_collect*.out"):
                m = pattern_idx.search(logfpath.name)
                if m:
                    idx = m.group(1)
                else:
                    continue

                try:
                    this_review = AutoreviewAnalysis(dirpath, idx)
                    self.reviews[dirpath.name].append(this_review)
                    if this_review.train_log is not None:
                        self.reviews_with_train[dirpath.name].append(this_review)
                except NoDataError:
                    self.no_data_error[dirpath.name].append({
                        'dirpath': dirpath,
                        'logfpath': logfpath
                    })
                    continue
                except LogFormatError:
                    self.log_format_error[dirpath.name].append({
                        'dirpath': dirpath,
                        'logfpath': logfpath
                    })
                    continue
        
