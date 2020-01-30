# -*- coding: utf-8 -*-

DESCRIPTION = """WoS journal subject metadata, to analyze by discipline"""

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

# manual mapping of subjects to broad subjects
subject_map = {
    'Chemistry': 'Natural Sciences',
    'Physiology': 'Medicine',
    'Physics': 'Natural Sciences',
    'Biochemistry & Molecular Biology': 'Biology',
    'Pharmacology & Pharmacy': 'Medicine',
    'Neurosciences & Neurology': 'Medicine',
    'Materials Science': 'Engineering',
    'Microbiology': 'Biology',
    'Endocrinology & Metabolism': 'Medicine',
    'General & Internal Medicine': 'Medicine',
    'Cardiovascular System & Cardiology': 'Medicine',
    'Environmental Sciences & Ecology': 'Environmental Sciences',
    'Engineering': 'Engineering',
    'Thermodynamics': 'Engineering',
    'Polymer Science': 'Engineering',
    'Oncology': 'Medicine',
    'Plant Sciences': 'Environmental Sciences',
    'Behavioral Sciences': 'Psychology and Social Sciences',
    'Life Sciences & Biomedicine - Other Topics': 'Biology',
    'Developmental Biology': 'Biology',
    'Psychology': 'Psychology and Social Sciences',
    'Geology': 'Earth Sciences',
    'Biotechnology & Applied Microbiology': 'Biology',
    'Toxicology': 'Medicine',
    'Anthropology': 'Psychology and Social Sciences',
    'Health Care Sciences & Services': 'Medicine',
    'Allergy': 'Medicine',
    'Cell Biology': 'Biology',
    'Astronomy & Astrophysics': 'Natural Sciences',
    'Ophthalmology': 'Medicine',
    'Genetics & Heredity': 'Medicine',
    'Psychiatry': 'Medicine',
    'Hematology': 'Medicine',
    'Zoology': 'Biology',
    'Fisheries': 'Environmental Sciences',
    'Entomology': 'Biology',
    'Pediatrics': 'Medicine',
    'Infectious Diseases': 'Medicine',
    'Optics': 'Natural Sciences',
    'Nutrition & Dietetics': 'Medicine',
    'Gastroenterology & Hepatology': 'Medicine',
    'Rheumatology': 'Medicine',
    'Oceanography': 'Earth Sciences',
    'Research & Experimental Medicine': 'Medicine',
    'Surgery': 'Medicine',
    'Meteorology & Atmospheric Sciences': 'Earth Sciences',
    'Anesthesiology': 'Medicine',
    'Immunology': 'Medicine',
    'Urology & Nephrology': 'Medicine',
    # 'Agriculture': '',
    'Geochemistry & Geophysics': 'Earth Sciences',
    # 'Science & Technology - Other Topics': '',
    'Medical Laboratory Technology': 'Medicine',
    'Obstetrics & Gynecology': 'Medicine',
    'Pathology': 'Medicine',
    'Virology': 'Medicine',
    'Nuclear Science & Technology': 'Natural Sciences',
    'Dentistry, Oral Surgery & Medicine': 'Medicine',
    'Spectroscopy': 'Natural Sciences',
    # 'Food Science & Technology': '',
    'Energy & Fuels': 'Natural Sciences',
    'Substance Abuse': 'Medicine',
    'Biophysics': 'Biology',
    'Social Sciences - Other Topics': 'Psychology and Social Sciences',
    'Computer Science': 'Engineering',
    'Reproductive Biology': 'Biology',
    'Electrochemistry': 'Natural Sciences',
    'Geriatrics & Gerontology': 'Medicine',
    'Dermatology': 'Medicine',
    'Radiology, Nuclear Medicine & Medical Imaging': 'Medicine',
    'Marine & Freshwater Biology': 'Biology',
    'Public, Environmental & Occupational Health': 'Psychology and Social Sciences',
    'Emergency Medicine': 'Medicine',
    'Respiratory System': 'Medicine',
    # 'Instruments & Instrumentation': '',
    'Anatomy & Morphology': 'Biology',
}

def get_subj_ext_first(subj_ext):
    if subj_ext is not None:
        split = subj_ext.split('; ')
        return split[0]
    return None

def get_subj_mapped_from_model(model, subject_map):
    """Get the mapped (broad) subject from a model

    :model: AutoreviewAnalysisModel object
    :subject_map: dictionary mapping subjects to broader categories
    :returns: mapped subject

    """
    subj_ext = model.paper_info.get('subject_extended')
    subj_first = get_subj_ext_first(subj_ext)
    return subject_map.get(subj_first, 'Other')
