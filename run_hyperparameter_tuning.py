# -*- coding: utf-8 -*-
"""
@author: F. B. Pérez Maurera
"""

import argparse
import logging
import os
import psutil
import time
from functools import partial
from typing import List, Type

from scipy.sparse import isspmatrix_csr, csr_matrix

from Base.BaseRecommender import BaseRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout
from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from Recommender_import_list import *
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.config import configure_logger
from Utils.dataset import ContentWiseImpressions, read_dataset


def search_hyperparameter_to_recommenders(urm_train_split: csr_matrix,
                                          urm_validation_split: csr_matrix,
                                          urm_test_split: csr_matrix,
                                          urm_impressions: csr_matrix,
                                          recommender: Type[BaseRecommender]):
    URM_train = urm_train_split.copy()
    URM_validation = urm_validation_split.copy()
    URM_test = urm_test_split.copy()
    URM_impressions = urm_impressions.copy()

    if any(not isspmatrix_csr(split) for split in [URM_train, URM_validation, URM_test, URM_impressions]):
        raise ValueError("The matrices are not all CSR matrices.")

    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    if recommender_class.RECOMMENDER_NAME == Random.RECOMMENDER_NAME:
        evaluator_validation = EvaluatorHoldout(URM_validation,
                                                cutoff_list=[10],
                                                parallel=False)

        evaluator_test = EvaluatorHoldout(URM_test,
                                          cutoff_list=[5, 10, 20],
                                          parallel=False)
    else:
        evaluator_validation = EvaluatorHoldout(URM_validation,
                                                cutoff_list=[10],
                                                parallel=True,
                                                num_workers=NUM_WORKERS)

        evaluator_test = EvaluatorHoldout(URM_test,
                                          cutoff_list=[5, 10, 20],
                                          parallel=True,
                                          num_workers=NUM_WORKERS)

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       URM_train_last_test=URM_train + URM_validation,
                                                       metric_to_optimize=METRIC_TO_OPTIMIZE,
                                                       evaluator_validation_earlystopping=evaluator_validation,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluator_test=evaluator_test,
                                                       output_folder_path=EXPERIMENTS_FOLDER_PATH,
                                                       parallelizeKNN=False,
                                                       allow_weighting=True,
                                                       resume_from_saved=True,
                                                       n_cases=NUM_CASES,
                                                       n_random_starts=NUM_RANDOM_STARTS,
                                                       URM_impressions=URM_impressions)

    try:
        runParameterSearch_Collaborative_partial(recommender)
    except Exception as e:
        logging.exception(f"On recommender {recommender} Exception {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--tune_recommenders',
                        default=False,
                        help="Run hyperparameter search for recommenders",
                        action='store_true')
    input_flags = parser.parse_args()
    tune_recommenders = input_flags.tune_recommenders

    configure_logger(logs_dir=os.path.join(".", "logs"),
                     root_filename=f"{time.time()}_{os.path.basename(__file__)}")

    logger = logging.getLogger("contentwise-impressions")

    DATASET_VARIANT = ContentWiseImpressions.Variant.CW10M
    dataset: ContentWiseImpressions = read_dataset(DATASET_VARIANT)

    EXPERIMENTS_FOLDER_PATH = os.path.join(".", "result_experiments", dataset.name, dataset.variant, "")

    COLLABORATIVE_ALGORITHM_LIST: List[Type[BaseRecommender]] = [
        Random,
        TopPop,
        ItemKNNCFRecommender,
        RP3betaRecommender,
        PureSVDRecommender,
        MatrixFactorization_BPR_Impression_Cython,
    ]

    METRIC_TO_OPTIMIZE = "MAP"
    NUM_CASES = 50
    NUM_RANDOM_STARTS = 15
    NUM_WORKERS = int(psutil.cpu_count())

    logger.debug(f"\nDataset variant: {DATASET_VARIANT.value}"
                 f"\nAlgorithm names: {'_'.join([algorithm_class.__name__ for algorithm_class in COLLABORATIVE_ALGORITHM_LIST])}"
                 f"\nMetric to optimize: {METRIC_TO_OPTIMIZE}"
                 f"\nNum cases: {NUM_CASES}"
                 f"\nNum random starts: {NUM_RANDOM_STARTS}"
                 f"\nExperiments folder path: {EXPERIMENTS_FOLDER_PATH}")

    os.makedirs(EXPERIMENTS_FOLDER_PATH, exist_ok=True)
    logger.info(f"Created folder: {EXPERIMENTS_FOLDER_PATH}")

    if tune_recommenders:
        for recommender_class in COLLABORATIVE_ALGORITHM_LIST:
            logger.debug(f"Recommender class: {recommender_class.__name__}")

            search_hyperparameter_to_recommenders(urm_train_split=dataset.URM["train"],
                                                  urm_validation_split=dataset.URM["validation"],
                                                  urm_test_split=dataset.URM["test"],
                                                  urm_impressions=dataset.URM["impressions"],
                                                  recommender=recommender_class,)
