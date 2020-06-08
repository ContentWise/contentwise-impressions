# -*- coding: utf-8 -*-
"""
@author: F. B. Pérez Maurera
"""

import argparse
import logging
import os

import numpy as np
from scipy.sparse import csr_matrix

from Utils.ResultFolderLoader import ResultFolderLoader
from Utils.config import configure_logger
from Utils.dataset import ContentWiseImpressions, read_dataset


def print_results(urm_test_split: csr_matrix):

    urm_test = urm_test_split.copy()

    n_test_users = np.sum(np.ediff1d(urm_test.indptr) >= 1)

    result_loader = ResultFolderLoader(EXPERIMENTS_FOLDER_PATH,
                                       base_algorithm_list=None,
                                       other_algorithm_list=None,
                                       KNN_similarity_list=KNN_SIMILARITY_LIST,
                                       ICM_names_list=None,
                                       UCM_names_list=None)

    article_metrics_latex_results_filename = os.path.join(RESULTS_EXPORT_FOLDER_PATH,
                                                          "article_metrics_latex_results.txt")
    result_loader.generate_latex_results(article_metrics_latex_results_filename,
                                         metrics_list=["RECALL", "MAP"],
                                         cutoffs_list=METRICS_CUTOFF_TO_REPORT_LIST,
                                         table_title=None,
                                         highlight_best=True)

    beyond_accuracy_metrics_latex_results_filename = os.path.join(RESULTS_EXPORT_FOLDER_PATH,
                                                                  "beyond_accuracy_metrics_latex_results.txt")
    result_loader.generate_latex_results(beyond_accuracy_metrics_latex_results_filename,
                                         metrics_list=["DIVERSITY_MEAN_INTER_LIST",
                                                       "DIVERSITY_HERFINDAHL",
                                                       "COVERAGE_ITEM",
                                                       "DIVERSITY_GINI",
                                                       "SHANNON_ENTROPY"],
                                         cutoffs_list=OTHERS_CUTOFF_TO_REPORT_LIST,
                                         table_title=None,
                                         highlight_best=True)

    all_metrics_latex_results_filename = os.path.join(RESULTS_EXPORT_FOLDER_PATH,
                                                      "all_metrics_latex_results.txt")
    result_loader.generate_latex_results(all_metrics_latex_results_filename,
                                         metrics_list=["PRECISION",
                                                       "RECALL",
                                                       "MAP",
                                                       "MRR",
                                                       "NDCG",
                                                       "F1",
                                                       "HIT_RATE",
                                                       "ARHR",
                                                       "NOVELTY",
                                                       "DIVERSITY_MEAN_INTER_LIST",
                                                       "DIVERSITY_HERFINDAHL",
                                                       "COVERAGE_ITEM",
                                                       "DIVERSITY_GINI",
                                                       "SHANNON_ENTROPY"],
                                         cutoffs_list=OTHERS_CUTOFF_TO_REPORT_LIST,
                                         table_title=None,
                                         highlight_best=True)

    time_latex_results_filename = os.path.join(RESULTS_EXPORT_FOLDER_PATH,
                                               "time_latex_results.txt")
    result_loader.generate_latex_time_statistics(time_latex_results_filename,
                                                 n_evaluation_users=n_test_users,
                                                 table_title=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--show_results',
                        default=False,
                        help="Save results",
                        action='store_true')
    input_flags = parser.parse_args()

    show_results = input_flags.show_results

    configure_logger(logs_dir=os.path.join(".", "logs"),
                     root_filename=f"{os.path.basename(__file__)}")

    logger = logging.getLogger("contentwise-impressions")

    DATASET_VARIANT = ContentWiseImpressions.Variant.CW10M
    dataset: ContentWiseImpressions = read_dataset(DATASET_VARIANT)

    KNN_SIMILARITY_LIST = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]
    METRICS_CUTOFF_TO_REPORT_LIST = [5, 10, 20]
    OTHERS_CUTOFF_TO_REPORT_LIST = [20]

    # Experiments path MUST end with the trailing / (or \ on Windows) in order to load the data.
    # We put the "" at the end to ensure that os.path.join adds the trailing slash that we want.
    EXPERIMENTS_FOLDER_PATH = os.path.join(".", "result_experiments", dataset.name, dataset.variant, "")
    if not os.path.exists(EXPERIMENTS_FOLDER_PATH):
        raise ValueError(
            f"Results folder does not exist. Cannot read results from a non-existing folder. Tried: {EXPERIMENTS_FOLDER_PATH}")

    RESULTS_EXPORT_FOLDER_PATH = os.path.join(EXPERIMENTS_FOLDER_PATH, "results_export")
    os.makedirs(RESULTS_EXPORT_FOLDER_PATH, exist_ok=True)

    logger.info(f"Created folder to export results: {RESULTS_EXPORT_FOLDER_PATH}")

    logger.debug(f"\nShow results: {show_results}"
                 f"\nDataset variant: {DATASET_VARIANT.value}"
                 f"\nMetrics cutoff to report list: {METRICS_CUTOFF_TO_REPORT_LIST}"
                 f"\nOthers cutoff to report list: {OTHERS_CUTOFF_TO_REPORT_LIST}"
                 f"\nExperiments folder path: {EXPERIMENTS_FOLDER_PATH}"
                 f"\nResults export folder path: {RESULTS_EXPORT_FOLDER_PATH}")

    if show_results:
        try:
            print_results(urm_test_split=dataset.URM["test"])
        except:
            logger.exception("An error occurred. See the logs.")
