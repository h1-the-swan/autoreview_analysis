# -*- coding: utf-8 -*-

DESCRIPTION = """Classes to help with analysis of Autoreview experiments"""

import sys, os, time, re, json
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
pattern_seed_size = re.compile(r"number of seed papers: (\d+)\s")
pattern_clf = re.compile(r"(\S+\(.*?\))\s.*?Fitting pipeline", flags=re.DOTALL)  # group 1 is the string representation of the classifier
pattern_feature_names = re.compile(r"feature names: (.*)$", flags=re.MULTILINE)
pattern_prec_recall_f1_at_n = re.compile(r"n==(\d+): prec==(.*?), recall==(.*?), f1==(.*?)$", flags=re.MULTILINE)
pattern_average_precision = re.compile(r"average_precision==(.*?)$", flags=re.MULTILINE)

pattern_timestamp = re.compile(r"(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)")

pattern_transformer_num = re.compile(r"features(\d+?)\D")

pattern_sample_set = re.compile(r"train_models\S*\.py (\S*sample\S*) (\S*) ")  # group 1 is the path to the sample IDs file; group 2 is the index within this file

from autoreview.util import load_data_from_pickles

def get_logfname(basedir, n, logtype='collect'):
    return basedir.joinpath("select_wos_id_and_{}_{}.out".format(logtype, n))

class NoDataError(Exception):
    pass

class LogFormatError(Exception):
    pass

def get_num_seed(path):
    """Right now, one of the only ways to get the number of seeds is to search for the seed papers and count the number of rows.
    TODO: find a better way

    :path: Path object. Will search within this path for `seed_papers.pickle`, then go back one directory and search again, etc.
    :returns: number of seed papers (int)

    """
    path = path.resolve()
    if not path.is_dir():
        path = path.parent
    while True:
        seed_fpath = path.joinpath('seed_papers.pickle')
        if seed_fpath.exists():
            seed_papers = pd.read_pickle(seed_fpath)
            return len(seed_papers)

        # if not found, climb up one directory and try again
        path = path.joinpath('..').resolve()
        if len(path.parts) == 1:
            # if we've gone all the way up to the root directory, stop. We have failed.
            return None

def parse_train_log(train_log, yield_models=False):
    # if yield_models is true, yield the models one by one instead of returning a list
    # if the input is a Path object, read it. otherwise, assume it is the actual text of the log
    if isinstance(train_log, Path):
        log_fpath = train_log
        train_log = train_log.read_text()
    else:
        log_fpath = None
    if yield_models is True:
        return _parse_train_log_generator(train_log, log_fpath)
    else:
        return list(_parse_train_log_generator(train_log, log_fpath))

def _parse_train_log_generator(train_log, log_fpath):
    # train_log: text of the log
    log_split = train_log.split('========Pipeline:\n')
    sizes = pattern_paper_set_sizes.findall(log_split[0])
    if not sizes:
        raise LogFormatError("set sizes pattern not found in this log")
    sizes = [int(x) for x in sizes[0]]
    num_candidates, num_target, num_target_in_candidates = sizes

    match_seed = pattern_seed_size.search(log_split[0])
    if match_seed:
        num_seed = match_seed.group(1)
    else:
        # fallback method
        num_seed = get_num_seed(log_fpath)

    models = [] # list of AutoreviewAnalysisModel instances
    for model_idx, model_txt in enumerate(log_split[1:]):
        # get the string representation of the classifier
        m = pattern_clf.search(model_txt)
        clf = m.group(1) if m else None
        if clf is None:
            continue
        clf_type = clf[:clf.find("(")]

        m = pattern_feature_names.search(model_txt)
        feature_names = m.group(1) if m else None
        
        m = pattern_predicted_false.search(model_txt)
        score_false = int(m.group(1)) if m else 0
        
        m = pattern_predicted_true.search(model_txt)
        score_true = int(m.group(1)) if m else 0
        score = score_true / num_target

        prec_recall_f1_at_n = pattern_prec_recall_f1_at_n.findall(model_txt)
        if prec_recall_f1_at_n:
            scores_at_rank_n = {}
            for n, prec, recall, f1 in prec_recall_f1_at_n:
                n = int(n)
                prec = float(prec)
                recall = float(recall)
                f1 = float(f1)
                scores_at_rank_n[n] = {
                    'prec': prec,
                    'recall': recall,
                    'f1': f1
                }
        else:
            scores_at_rank_n = None

        m = pattern_average_precision.search(model_txt)
        average_precision = float(m.group(1)) if m else None

        this_model = AutoreviewAnalysisModel(
            log_fpath=log_fpath,
            model_idx=model_idx,
            clf=clf,
            clf_type=clf_type,
            feature_names=feature_names,
            num_correctly_predicted=score_true,
            score_correctly_predicted=score,
            num_seed=num_seed,
            num_candidates=num_candidates,
            num_target=num_target,
            num_target_in_candidates=num_target_in_candidates,
            scores_at_rank_n=scores_at_rank_n,
            average_precision=average_precision
        )
        yield this_model

pattern_dirname = re.compile(r"output directory is (\S+)\s")

def get_dirpath_from_log_fragment(log_fragment, basedir, pattern=pattern_dirname):
    m = pattern.search(log_fragment)
    if not m:
        return None
    output_dirpath = m.group(1)
    # align with basedir
    output_dirpath = output_dirpath[output_dirpath.find(str(basedir)):]
    output_dirpath = Path(output_dirpath)
    return output_dirpath

def get_log_start_time(log_txt, pattern=pattern_timestamp, fmt='%Y-%m-%d %H:%M:%S'):
    """get the first timestamp from the log text

    :returns: POSIX timestamp (float)

    """
    m = pattern.findall(log_txt)
    if m:
        ts = datetime.strptime(m[0], fmt)
        return ts.timestamp()
    return None

def process_one_log(log, split_phrase='training models for subdir', basedir=Path('.'), ignore_rerun=True):
    """process one train log

    :log: Path to train log file, or the text of a log
    :yields: log fragments for subdirs

    """
    if isinstance(log, Path):
        log = log.read_text()
    log_split = log.split(split_phrase)
    if len(log_split) <= 1:
        # no data in this log, or not properly formatted
        return (None, None)
    for log_fragment in log_split[1:]:
        # ignore in certain cases:
        _logupper = log_fragment.upper()
        if 'SKIPPING' in _logupper:
            continue
        if 'CANCELLED' in _logupper:
            continue
        if ignore_rerun is True and 'RERUNNING' in _logupper:
            continue
        log_fragment = split_phrase + log_fragment
        output_dirpath = get_dirpath_from_log_fragment(log_fragment, basedir)
        if output_dirpath:
            yield (output_dirpath, log_fragment)

def filter_out_bad_logs(transformer_num, log_timestamp, buggy_transformer_nums=[6,7,8,10,11], ignore_before=1579920800):
    """Due to a bug, models trained with certain features before a certain datetime should be ignored.

    :transformer_num: transformer number for the training run
    :log_timestamp: timestamp of log given by get_log_start_time()
    :buggy_transformer_nums: list of transformer numbers that could be problematic
    :ignore_before: filter out logs with those features if the timestamp is earlier than this
    :returns: True if the log passes the filter, False if it is buggy and should be ignored

    """
    if transformer_num in buggy_transformer_nums:
        if log_timestamp > ignore_before:
            return True
        else:
            return False
    return True

class AutoreviewAnalysisModel:

    """Represents stats for a single trained model (e.g. LogisticRegression or RandomForestClassifier)"""

    def __init__(self, 
                 log_fpath=None, 
                 dirpath=None, 
                 model_idx=None,
                 origin_train_log_fpath=None,
                 origin_train_log_start_time=None,
                 clf=None, 
                 clf_type=None, 
                 feature_names=None, 
                 num_correctly_predicted=None, 
                 score_correctly_predicted=None, 
                 num_seed=None, 
                 num_target=None, 
                 num_candidates=None, 
                 num_target_in_candidates=None, 
                 paper_info=None,
                 scores_at_rank_n=None,  # dictionary of n to dictionary -> {prec, recall, f1}
                 average_precision=None,
                 sample_set=None,
                 sample_set_idx=None):
        self.log_fpath = log_fpath
        self.dirpath = dirpath
        if self.dirpath is None and self.log_fpath is not None:
            self.dirpath = self.log_fpath.parent
        self.model_idx = model_idx  # index of model within the train log
        self.origin_train_log_fpath = origin_train_log_fpath  # if the data comes from an aggregate train log, this will keep track of which log
        self.origin_train_log_start_time = origin_train_log_start_time
        self.train_log = None
        self.clf = clf  # string representing the model and model parameters
        self.clf_type = clf_type
        self.feature_names = feature_names
        self.num_correctly_predicted = num_correctly_predicted
        self.score_correctly_predicted = score_correctly_predicted
        self.num_seed = num_seed
        self.num_target = num_target
        self.num_candidates_before_year_lowpass = None
        self.num_candidates = num_candidates  # after year filter
        self.num_target_in_candidates = num_target_in_candidates  # number of target papers that appear in the candidate set
        self.scores_at_rank_n = scores_at_rank_n
        self.average_precision = average_precision
        self.sample_set = sample_set
        self.sample_set_idx = sample_set_idx

        self.paper_info = paper_info
        if not self.paper_info and self.dirpath is not None:
            self.paper_info = self.get_paper_info(self.dirpath)

    def get_paper_info(self, path):
        """find the paper_info.json file, and read it

        :path: Path object. Will search within this path for `paper_info.json`, then go back one directory and search again, etc.
        :returns: dictionary parsed from JSON file

        """
        path = path.resolve()
        if not path.is_dir():
            path = path.parent
        while True:
            paperinfo_fpath = path.joinpath('paper_info.json')
            if paperinfo_fpath.exists():
                return json.loads(paperinfo_fpath.read_text())

            # if not found, climb up one directory and try again
            path = path.joinpath('..').resolve()
            if len(path.parts) == 1:
                # if we've gone all the way up to the root directory, stop. We have failed.
                return None

class AutoreviewAnalysisDataset:

    """Get data for all experiments from train logs."""

    def __init__(self, basedir='data/hpc', glob_pattern="*train_models*features*.out", split_phrase="training models for subdir", ignore_transformer_nums=[1,2]):
        self.basedir = Path(basedir)
        self.glob_pattern = glob_pattern
        self.split_phrase = split_phrase
        self.ignore_transformer_nums = ignore_transformer_nums

        # keep track of errors and ignored logs
        self.ignored_fpaths = []
        self.attr_error_fpaths = []
        self.no_timestamp = []

        self.models = []  # initialize

        self.collect_data()

        self.df = self.get_dataframe(self.models)

        self.top_models = self.get_top_models(self.df)

    def collect_data(self):
        """collect the data set

        """
        train_fpaths = self.basedir.rglob(self.glob_pattern)
        for fp in train_fpaths:
            m = pattern_transformer_num.search(str(fp))
            if not m:
                raise RuntimeError("unexpected error: transformer number not found in the filename for {}".format(fp))
            transformer_num = int(m.group(1))
            log_txt = fp.read_text()
            log_timestamp = get_log_start_time(log_txt)
            if not log_timestamp:
                self.no_timestamp.append(fp)
                continue

            m = pattern_sample_set.search(log_txt)
            if m:
                sample_set_fpath = Path(m.group(1))
                sample_set_idx = int(m.group(2))
            else:
                sample_set_fpath = None
                sample_set_idx = None

            if filter_out_bad_logs(transformer_num, log_timestamp) and transformer_num not in self.ignore_transformer_nums:
                for output_dirpath, log_fragment in process_one_log(log_txt, self.split_phrase, self.basedir, ignore_rerun=False):
                    if output_dirpath:
                        try:
                            for model in parse_train_log(log_fragment, yield_models=True):
                                model.dirpath = output_dirpath
                                model.paper_info = model.get_paper_info(model.dirpath)
                                model.origin_train_log_fpath = fp
                                model.origin_train_log_start_time = log_timestamp
                                model.transformer_num = transformer_num
                                model.sample_set = sample_set_fpath
                                model.sample_set_idx = sample_set_idx
                                self.models.append(model)
                        except AttributeError:
                            self.attr_error_fpaths.append(fp)
            else:
                self.ignored_fpaths.append(fp)

    def get_dataframe(self, models):
        """Get the data as a pandas dataframe and dedup

        :models: list of AutoreviewAnalysisModel objects
        :returns: pandas dataframe

        """
        data = []
        for i, model in enumerate(models):
            paper_info = model.paper_info
            if not paper_info:
                paper_info = {}
            data.append({
                'clf_type': model.clf_type,
                'feature_names': model.feature_names,
                'score_correctly_predicted': model.score_correctly_predicted,
                'num_seed': model.num_seed,
                'num_candidates': model.num_candidates,
                'num_target': model.num_target,
                'num_target_in_candidates': model.num_target_in_candidates,
                'log_fpath': model.log_fpath,
                'dirpath': model.dirpath,
                'UID': paper_info.get('UID'),
                'pub_year': paper_info.get('pub_year'),
                'models_idx': i,
                'scores_at_rank_n': model.scores_at_rank_n,
                'average_precision': model.average_precision,
                'origin_train_log_fpath': model.origin_train_log_fpath,
                'origin_train_log_start_time': model.origin_train_log_start_time,
                'transformer_num': model.transformer_num,
                'sample_set': model.sample_set,
                'sample_set_idx': model.sample_set_idx,
            })
        df_models = pd.DataFrame(data)

        df_models['num_seed'] = df_models.num_seed.astype(int)
        df_models['num_refs'] = df_models.num_seed + df_models.num_target
        df_models['ratio_target_in_candidates'] = df_models.num_target_in_candidates / df_models.num_target

        # drop duplicates, keeping the most recent
        dedup_cols = ['clf_type', 'score_correctly_predicted', 'dirpath']
        df_models_dedup = df_models.sort_values('origin_train_log_start_time', ascending=False).drop_duplicates(dedup_cols, keep='first')

        return df_models_dedup

    def get_top_models(self, df):
        """Get the top models, one for each review article/seed size/feature set

        :df: dataframe for individual models
        :returns: pandas dataframe with just the top models

        """
        top_models = df.sort_values('score_correctly_predicted', ascending=False)
        top_models = top_models.drop_duplicates(subset=['dirpath'], keep='first')
        return top_models


        

        

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
        m = pattern_wos_id_slug.search(self.collect_log)
        if m:
            return m.group(1)
        else:
            raise LogFormatError("could not find WoS ID slug in log {}".format(self.collect_log_fpath))
    
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
        
