# -*- coding: utf-8 -*-

DESCRIPTION = """Reload pickled trained models, match with model stats, rerun predictions, and output to files"""

import sys, os, time, pickle
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

from ast import literal_eval
import joblib
from autoreview_analysis import parse_train_log, LogFormatError
from autoreview.util import predict_ranks_from_data, prepare_data_for_model

def load_model_info_all_models(collected_logs_fpath):
    """get a list of all AutoreviewAnalysisModel objects from the dirpaths and log_txt in collected_logs_fpath

    :collected_logs_fpath: path to pickle file containing a dictionary of dirpath -> log_txt for logs from training autoreview models (output of parse_and_organize_train_logs.py)
    :returns: list of AutoreviewAnalysisModel objects

    """
    collected_logs = pickle.loads(collected_logs_fpath.read_bytes())
    logger.debug('there are {} items in {}'.format(len(collected_logs), collected_logs_fpath))
    log_format_errs = []
    models = []
    for dirpath, log_txt in collected_logs.items():
        try:
            for model in parse_train_log(log_txt, yield_models=True):
                model.dirpath = dirpath
                model.paper_info = model.get_paper_info(model.dirpath)
                models.append(model)
        except LogFormatError:
            log_format_errs.append(dirpath)
    return models

def get_sklearn_model_fpaths(autoreview_analysis_models, glob_pattern='best_model.pickle'):
    """for every AutoreviewAnalysisModel object, get the path to the pickle containing the last best model saved

    :autoreview_analysis_models: list of all AutoreviewAnalysisModel objects
    :returns: set of unique Paths to pickled sklearn models

    """
    model_fpaths = []
    for model in autoreview_analysis_models:
        model_fpath = model.dirpath.rglob(glob_pattern)
        model_fpath = list(model_fpath)
        if model_fpath:
            model_fpath.sort()
            model_fpath = model_fpath[-1]
            model_fpaths.append(model_fpath)
    return set(model_fpaths)

def parse_clf(clf_str):
    clf_type = clf_str[:clf_str.find('(')]
    p = clf_str[clf_str.find('(')+1:]
    p = p[:p.rfind(')')]
    p = p.split(',')
    p = [x.strip().split('=') for x in p]
    arg_dict = {}
    for k, v in p:
        if k != 'random_state':
            arg_dict[k] = literal_eval(v)
    return clf_type, arg_dict

def match_model(trained_model, autoreview_analysis_model):
    """
    Match a trained sklearn model (loaded from a pickle file using joblib.load) 
    to an AutoreviewAnalysisModel object.
    
    Return True if they are a match, False otherwise.
    """
    trained_model_clf_type, trained_model_arg_dict = parse_clf(str(trained_model._final_estimator))
    clf_type, arg_dict = parse_clf(autoreview_analysis_model.clf)
    if clf_type == trained_model_clf_type:
        if trained_model_clf_type == "RandomForestClassifier":
            conds = [
                arg_dict.get('n_estimators') == trained_model_arg_dict.get('n_estimators'),
                arg_dict.get('criterion') == trained_model_arg_dict.get('criterion'),
            ]
        elif trained_model_clf_type == 'LogisticRegression':
            conds = [
                arg_dict.get('class_weight') == trained_model_arg_dict.get('class_weight'),
                arg_dict.get('penalty') == trained_model_arg_dict.get('penalty'),
            ]
        else:
            # if it's a different type (probably AdaBoost), just assume true
            conds = [True]
        if all(conds):
            return True
    return False

def get_model_match_for_sklearn_model(trained_model_fpath, trained_model, autoreview_analysis_models):
    """Given one sklearn model and the list of all AutoreviewAnalysisModel objects, 
    find the AutoreviewAnalysisModel the matches the sklearn model

    :trained_model_fpath: path to pickled sklearn model
    :trained_model: sklearn model loaded from trained_model_fpath using joblib.load()
    :autoreview_analysis_models: list of AutoreviewAnalysisModel objects
    :returns: matching AutoreviewAnalysisModel object

    """
    for autoreview_analysis_model in autoreview_analysis_models:
        # match_model() is expensive, so first check if the directories match:
        if str(autoreview_analysis_model.dirpath) in str(trained_model_fpath):
            if match_model(trained_model, autoreview_analysis_model):
                return autoreview_analysis_model
    return None

def output(basedir, i, df_preds, autoreview_analysis_model_match, trained_model_fpath):
    outdir = basedir.joinpath("{:09d}".format(i))
    logger.debug('putting output files in {}'.format(outdir))
    outdir.mkdir(exist_ok=False)

    outfpath = outdir.joinpath('ranked_pred_y.csv.gz')
    df_preds.target.to_csv(outfpath, index=False, header=False)

    outfpath = outdir.joinpath('autoreview_analysis_model.pickle')
    outfpath.write_bytes(pickle.dumps(autoreview_analysis_model_match))

    outfpath = outdir.joinpath('sklearn_model_pickle_fpath.txt')
    outfpath.write_text(str(trained_model_fpath))

    


def main(args):
    outdir_base = Path(args.outdir)
    if not outdir_base.is_dir():
        raise FileNotFoundError('could not locate output base directory {}'.format(outdir_base))
    collected_logs_fpath = Path(args.collected_logs)
    autoreview_analysis_models = load_model_info_all_models(collected_logs_fpath)
    logger.debug('loaded {} AutoreviewAnalysisModel objects'.format(len(autoreview_analysis_models)))
    sklearn_model_fpaths = get_sklearn_model_fpaths(autoreview_analysis_models)
    sklearn_model_fpaths = list(sklearn_model_fpaths)
    sklearn_model_fpaths.sort()
    logger.debug('found {} sklearn_model_fpaths'.format(len(sklearn_model_fpaths)))
    for i, trained_model_fpath in enumerate(sklearn_model_fpaths):
        logger.debug('processing trained_model_fpath {} (i=={})'.format(trained_model_fpath, i))
        trained_model = joblib.load(trained_model_fpath)
        autoreview_analysis_model_match = get_model_match_for_sklearn_model(trained_model_fpath, trained_model, autoreview_analysis_models)
        if not autoreview_analysis_model_match:
            logger.warn('could not match model {}'.format(trained_model_fpath))
            logger.debug('skipping {}'.format(trained_model_fpath))
            continue

        year = autoreview_analysis_model_match.paper_info.get('pub_year')
        if not year:
            logger.warn('could not find year for model: {}'.format(trained_model_fpath))
            logger.debug('skipping {}'.format(trained_model_fpath))
            continue
        logger.debug('year is {}'.format(year))
        candidate_papers, seed_papers, target_papers = prepare_data_for_model(autoreview_analysis_model_match.dirpath.parent, year=year, id_colname='ID')
        # logger.debug('loaded {} candidate_papers, {} seed_papers, {} target_papers'.format(len(candidate_papers), len(seed_papers), len(target_papers)))
        df_preds = predict_ranks_from_data(trained_model, candidate_papers)
        # logger.debug('predicted {} ranks'.format(len(df_preds)))
        output(basedir, i, df_preds, autoreview_analysis_model_match, trained_model_fpath)


    # PSEUDOCODE:
    #     output(df_preds, autoreview_analysis_model_match, trained_model_fpath)


if __name__ == "__main__":
    total_start = timer()
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("collected_logs", help="path to pickle file containing a dictionary of dirpath -> log_txt for logs from training autoreview models (output of parse_and_organize_train_logs.py)")
    parser.add_argument("outdir", help="base directory (should exist) for output (numbered subdirectories will be put here)")
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
