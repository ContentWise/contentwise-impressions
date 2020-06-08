#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/06/18

@author: Maurizio Ferrari Dacrema
@author: F. B. Pérez Maurera
"""

import copy
import logging
import sys
import time
import psutil
from enum import Enum

import dask
import numpy as np
import scipy.sparse as sps
from typing import Dict

from Base.BaseRecommender import BaseRecommender
from Base.Evaluation.metrics import roc_auc, precision, precision_recall_min_denominator, recall, MAP, MRR, ndcg, arhr, \
    Novelty, Coverage_Item, _Metrics_Object, Coverage_User, Gini_Diversity, Shannon_Entropy, Diversity_MeanInterList, \
    Diversity_Herfindahl, AveragePopularity
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

logger = logging.getLogger("contentwise-impressions")


class EvaluatorMetrics(Enum):

    ROC_AUC = "ROC_AUC"
    PRECISION = "PRECISION"
    PRECISION_RECALL_MIN_DEN = "PRECISION_RECALL_MIN_DEN"
    RECALL = "RECALL"
    MAP = "MAP"
    MRR = "MRR"
    NDCG = "NDCG"
    F1 = "F1"
    HIT_RATE = "HIT_RATE"
    ARHR = "ARHR"
    # RMSE = "RMSE"
    NOVELTY = "NOVELTY"
    AVERAGE_POPULARITY = "AVERAGE_POPULARITY"
    DIVERSITY_SIMILARITY = "DIVERSITY_SIMILARITY"
    DIVERSITY_MEAN_INTER_LIST = "DIVERSITY_MEAN_INTER_LIST"
    DIVERSITY_HERFINDAHL = "DIVERSITY_HERFINDAHL"
    COVERAGE_ITEM = "COVERAGE_ITEM"
    COVERAGE_USER = "COVERAGE_USER"
    DIVERSITY_GINI = "DIVERSITY_GINI"
    SHANNON_ENTROPY = "SHANNON_ENTROPY"


def _create_empty_metrics_dict(cutoff_list, n_items, n_users, URM_train, URM_test, ignore_items, ignore_users, diversity_similarity_object):
    empty_dict = {}

    for cutoff in cutoff_list:
        cutoff_dict = {}

        for metric in EvaluatorMetrics:
            if metric == EvaluatorMetrics.COVERAGE_ITEM:
                cutoff_dict[metric.value] = Coverage_Item(n_items, ignore_items)

            elif metric == EvaluatorMetrics.DIVERSITY_GINI:
                cutoff_dict[metric.value] = Gini_Diversity(n_items, ignore_items)

            elif metric == EvaluatorMetrics.SHANNON_ENTROPY:
                cutoff_dict[metric.value] = Shannon_Entropy(n_items, ignore_items)

            elif metric == EvaluatorMetrics.COVERAGE_USER:
                cutoff_dict[metric.value] = Coverage_User(n_users, ignore_users)

            elif metric == EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST:
                cutoff_dict[metric.value] = Diversity_MeanInterList(n_items, cutoff)

            elif metric == EvaluatorMetrics.DIVERSITY_HERFINDAHL:
                cutoff_dict[metric.value] = Diversity_Herfindahl(n_items, ignore_items)

            elif metric == EvaluatorMetrics.NOVELTY:
                cutoff_dict[metric.value] = Novelty(URM_train)

            elif metric == EvaluatorMetrics.AVERAGE_POPULARITY:
                cutoff_dict[metric.value] = AveragePopularity(URM_train)

            elif metric == EvaluatorMetrics.MAP:
                cutoff_dict[metric.value] = MAP()

            elif metric == EvaluatorMetrics.MRR:
                cutoff_dict[metric.value] = MRR()

            elif metric == EvaluatorMetrics.DIVERSITY_SIMILARITY:
                if diversity_similarity_object is not None:
                    cutoff_dict[metric.value] = copy.deepcopy(diversity_similarity_object)
            else:
                cutoff_dict[metric.value] = 0.0

        empty_dict[cutoff] = cutoff_dict

    return empty_dict


def get_result_string(results_run, n_decimals=7):
    output_str = ""

    for cutoff in results_run.keys():

        results_run_current_cutoff = results_run[cutoff]

        output_str += "CUTOFF: {} - ".format(cutoff)

        for metric in results_run_current_cutoff.keys():
            output_str += "{}: {:.{n_decimals}f}, ".format(metric, results_run_current_cutoff[metric], n_decimals = n_decimals)

        output_str += "\n"

    return output_str


def _remove_item_interactions(URM, item_list):

    URM = sps.csc_matrix(URM.copy())

    for item_index in item_list:

        start_pos = URM.indptr[item_index]
        end_pos = URM.indptr[item_index+1]

        URM.data[start_pos:end_pos] = np.zeros_like(URM.data[start_pos:end_pos])

    URM.eliminate_zeros()
    URM = sps.csr_matrix(URM)

    return URM


class Evaluator(object):
    """Abstract Evaluator"""

    EVALUATOR_NAME = "Evaluator_Base_Class"

    def __init__(self,
                 URM_test_list,
                 cutoff_list,
                 min_ratings_per_user=1,
                 exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None,
                 verbose=True,
                 parallel: bool = False,
                 num_workers: int = None):

        super(Evaluator, self).__init__()

        self.verbose = verbose
        self.parallel = parallel
        self.num_workers = num_workers

        if ignore_items is None:
            self.ignore_items_flag = False
            self.ignore_items_ID = np.array([])
        else:
            self._print("Ignoring {} Items".format(len(ignore_items)))
            self.ignore_items_flag = True
            self.ignore_items_ID = np.array(ignore_items)

        self.cutoff_list = cutoff_list.copy()
        self.max_cutoff = max(self.cutoff_list)

        self.min_ratings_per_user = min_ratings_per_user
        self.exclude_seen = exclude_seen

        if not isinstance(URM_test_list, list):
            self.URM_test = URM_test_list.copy()
            URM_test_list = [URM_test_list]
        else:
            raise ValueError("List of URM_test not supported")

        self.diversity_object = diversity_object

        self.n_users, self.n_items = URM_test_list[0].shape

        # Prune users with an insufficient number of ratings
        # During testing CSR is faster
        self.URM_test_list = []
        users_to_evaluate_mask = np.zeros(self.n_users, dtype=np.bool)

        for URM_test in URM_test_list:

            URM_test = _remove_item_interactions(URM_test, self.ignore_items_ID)

            URM_test = sps.csr_matrix(URM_test)
            self.URM_test_list.append(URM_test)

            rows = URM_test.indptr
            numRatings = np.ediff1d(rows)
            new_mask = numRatings >= min_ratings_per_user

            users_to_evaluate_mask = np.logical_or(users_to_evaluate_mask, new_mask)

        self.users_to_evaluate = np.arange(self.n_users)[users_to_evaluate_mask]

        if ignore_users is not None:
            self._print("Ignoring {} Users".format(len(ignore_users)))
            self.ignore_users_ID = np.array(ignore_users)
            self.users_to_evaluate = set(self.users_to_evaluate) - set(ignore_users)
        else:
            self.ignore_users_ID = np.array([])

        self.users_to_evaluate = list(self.users_to_evaluate)

        # Those will be set at each new evaluation
        self._start_time = np.nan
        self._start_time_print = np.nan
        self._n_users_evaluated = np.nan

    def _print(self, string):

        if self.verbose:
            print("{}: {}".format(self.EVALUATOR_NAME, string))

    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a BaseRecommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        self._start_time = time.time()
        self._start_time_print = time.time()
        self._n_users_evaluated = 0

        results_dict = (self._parallel_run_evaluation_on_selected_users(recommender_object, self.users_to_evaluate)
                        if self.parallel
                        else self._run_evaluation_on_selected_users(recommender_object, self.users_to_evaluate))

        if self._n_users_evaluated > 0:
            for cutoff in self.cutoff_list:
                results_current_cutoff = results_dict[cutoff]

                for key in results_current_cutoff.keys():
                    value = results_current_cutoff[key]

                    if isinstance(value, _Metrics_Object):
                        results_current_cutoff[key] = value.get_metric_value()
                    else:
                        results_current_cutoff[key] = value/self._n_users_evaluated

                if EvaluatorMetrics.F1.value in results_current_cutoff:
                    precision_ = results_current_cutoff[EvaluatorMetrics.PRECISION.value]
                    recall_ = results_current_cutoff[EvaluatorMetrics.RECALL.value]

                    if precision_ + recall_ != 0:
                        # F1 micro averaged: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf
                        results_current_cutoff[EvaluatorMetrics.F1.value] = 2 * (precision_ * recall_) / (precision_ + recall_)

        else:
            logger.warning("No users had a sufficient number of relevant items")
            self._print("WARNING: No users had a sufficient number of relevant items")

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        results_run_string = get_result_string(results_dict)

        return (results_dict, results_run_string)

    def get_user_relevant_items(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in getting relevant items"

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]

    def get_user_test_ratings(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in relevant items ratings"

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]

    def _compute_metrics_on_recommendation_list(self, test_user_batch_array, recommended_items_batch_list, scores_batch, results_dict):

        assert len(recommended_items_batch_list) == len(test_user_batch_array), "{}: recommended_items_batch_list contained recommendations for {} users, expected was {}".format(
            self.EVALUATOR_NAME, len(recommended_items_batch_list), len(test_user_batch_array))

        assert scores_batch.shape[0] == len(test_user_batch_array), "{}: scores_batch contained scores for {} users, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[0], len(test_user_batch_array))

        assert scores_batch.shape[1] == self.n_items, "{}: scores_batch contained scores for {} items, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[1], self.n_items)


        # Compute recommendation quality for each user in batch
        for batch_user_index in range(len(recommended_items_batch_list)):

            test_user = test_user_batch_array[batch_user_index]

            relevant_items = self.get_user_relevant_items(test_user)

            # Add the RMSE to the global object, no need to loop through the various cutoffs
            # This repository is not designed to ensure proper RMSE optimization
            # relevant_items_rating = self.get_user_test_ratings(test_user)
            #
            # all_items_predicted_ratings = scores_batch[batch_user_index]
            # global_RMSE_object = results_dict[self.cutoff_list[0]][EvaluatorMetrics.RMSE.value]
            # global_RMSE_object.add_recommendations(all_items_predicted_ratings, relevant_items, relevant_items_rating)

            # Being the URM CSR, the indices are the non-zero column indexes
            recommended_items = recommended_items_batch_list[batch_user_index]
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            self._n_users_evaluated += 1

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                is_relevant_current_cutoff = is_relevant[0:cutoff]
                recommended_items_current_cutoff = recommended_items[0:cutoff]

                results_current_cutoff[EvaluatorMetrics.ROC_AUC.value]              += roc_auc(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.PRECISION.value]            += precision(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.PRECISION_RECALL_MIN_DEN.value]   += precision_recall_min_denominator(is_relevant_current_cutoff, len(relevant_items))
                results_current_cutoff[EvaluatorMetrics.RECALL.value]               += recall(is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.NDCG.value]                 += ndcg(recommended_items_current_cutoff, relevant_items, relevance=self.get_user_test_ratings(test_user), at=cutoff)
                results_current_cutoff[EvaluatorMetrics.HIT_RATE.value]             += is_relevant_current_cutoff.sum()
                results_current_cutoff[EvaluatorMetrics.ARHR.value]                 += arhr(is_relevant_current_cutoff)

                results_current_cutoff[EvaluatorMetrics.MRR.value].add_recommendations(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.MAP.value].add_recommendations(is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.NOVELTY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.AVERAGE_POPULARITY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_GINI.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.SHANNON_ENTROPY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_USER.value].add_recommendations(recommended_items_current_cutoff, test_user)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].add_recommendations(recommended_items_current_cutoff)

                if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in results_current_cutoff:
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_SIMILARITY.value].add_recommendations(recommended_items_current_cutoff)


        if time.time() - self._start_time_print > 30 or self._n_users_evaluated==len(self.users_to_evaluate):

            elapsed_time = time.time()-self._start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            self._print("Processed {} ( {:.2f}% ) in {:.2f} {}. Users per second: {:.0f}".format(
                          self._n_users_evaluated,
                          100.0* float(self._n_users_evaluated)/len(self.users_to_evaluate),
                          new_time_value, new_time_unit,
                          float(self._n_users_evaluated)/elapsed_time))

            sys.stdout.flush()
            sys.stderr.flush()

            self._start_time_print = time.time()


        return results_dict

    def _parallel_compute_metrics_on_recommendation_list(self,
                                                         test_user_batch_array,
                                                         recommended_items_batch_list,
                                                         scores_batch,
                                                         results_dict):
        """
        TO RUN THIS IN PARALLEL, results_dict MUST be an empty copy of a results dictionary created by
        calling _create_empty_metrics_dict. If not, then race conditions may occur.
        """

        if len(recommended_items_batch_list) != len(test_user_batch_array):
            raise ValueError(f"{self.EVALUATOR_NAME}: recommended_items_batch_list contained recommendations for {len(recommended_items_batch_list)} users, expected was {len(test_user_batch_array)}")

        if scores_batch.shape[0] != len(test_user_batch_array):
            raise ValueError(f"{self.EVALUATOR_NAME}: scores_batch contained scores for {scores_batch.shape[0]} users, expected was {len(test_user_batch_array)}")

        if scores_batch.shape[1] != self.n_items:
            raise ValueError(f"{self.EVALUATOR_NAME}: scores_batch contained scores for {scores_batch.shape[1]} items, expected was {self.n_items}")

        for batch_user_index in range(len(recommended_items_batch_list)):

            test_user = test_user_batch_array[batch_user_index]

            relevant_items = self.get_user_relevant_items(test_user)

            # Being the URM CSR, the indices are the non-zero column indexes
            recommended_items = recommended_items_batch_list[batch_user_index]
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            self._n_users_evaluated += 1

            # self.cutoff_list is an array containing different cut points when measuring Map@K and metrics like that.
            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                is_relevant_current_cutoff = is_relevant[0:cutoff]
                recommended_items_current_cutoff = recommended_items[0:cutoff]

                results_current_cutoff[EvaluatorMetrics.ROC_AUC.value] += roc_auc(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.PRECISION.value] += precision(is_relevant_current_cutoff)
                results_current_cutoff[
                    EvaluatorMetrics.PRECISION_RECALL_MIN_DEN.value] += precision_recall_min_denominator(
                    is_relevant_current_cutoff, len(relevant_items))
                results_current_cutoff[EvaluatorMetrics.RECALL.value] += recall(is_relevant_current_cutoff,
                                                                                relevant_items)
                results_current_cutoff[EvaluatorMetrics.NDCG.value] += ndcg(recommended_items_current_cutoff,
                                                                            relevant_items,
                                                                            relevance=self.get_user_test_ratings(
                                                                                test_user), at=cutoff)
                results_current_cutoff[EvaluatorMetrics.HIT_RATE.value] += is_relevant_current_cutoff.sum()
                results_current_cutoff[EvaluatorMetrics.ARHR.value] += arhr(is_relevant_current_cutoff)

                results_current_cutoff[EvaluatorMetrics.MRR.value].add_recommendations(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.MAP.value].add_recommendations(is_relevant_current_cutoff,
                                                                                       relevant_items)
                results_current_cutoff[EvaluatorMetrics.NOVELTY.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.AVERAGE_POPULARITY.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_GINI.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.SHANNON_ENTROPY.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_USER.value].add_recommendations(
                    recommended_items_current_cutoff, test_user)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].add_recommendations(
                    recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].add_recommendations(
                    recommended_items_current_cutoff)

                if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in results_current_cutoff:
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_SIMILARITY.value].add_recommendations(
                        recommended_items_current_cutoff)

        return results_dict

    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size=None):
        raise NotImplementedError("This is an abstract class. Use EvaluatorHoldout or EvaluatorNegativeItemSample classes.")

    def _parallel_run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size=None):
        raise NotImplementedError("This is an abstract class. Use EvaluatorHoldout or EvaluatorNegativeItemSample classes.")


class EvaluatorHoldout(Evaluator):
    """EvaluatorHoldout"""

    EVALUATOR_NAME = "EvaluatorHoldout"

    def __init__(self,
                 URM_test_list,
                 cutoff_list,
                 min_ratings_per_user=1,
                 exclude_seen=True,
                 diversity_object=None,
                 ignore_items=None,
                 ignore_users=None,
                 verbose=True,
                 parallel: bool = False,
                 num_workers: int = None):

        super(EvaluatorHoldout, self).__init__(URM_test_list,
                                               cutoff_list,
                                               diversity_object=diversity_object,
                                               min_ratings_per_user=min_ratings_per_user,
                                               exclude_seen=exclude_seen,
                                               ignore_items=ignore_items,
                                               ignore_users=ignore_users,
                                               verbose=verbose,
                                               parallel=parallel,
                                               num_workers=num_workers)

    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size=None):

        if block_size is None:
            block_size = min(1000, int(1e8/self.n_items))
            block_size = min(block_size, len(users_to_evaluate))

        logger.debug(f"Block size: {block_size}")

        results_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                  self.n_items, self.n_users,
                                                  recommender_object.get_URM_train(),
                                                  self.URM_test,
                                                  self.ignore_items_ID,
                                                  self.ignore_users_ID,
                                                  self.diversity_object)


        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0

        start_all = time.time()
        logger.info(f"Starting evaluation.")

        while user_batch_start < len(users_to_evaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(users_to_evaluate))

            test_user_batch_array = np.array(users_to_evaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            recommended_items_batch_list, scores_batch = recommender_object.recommend(test_user_batch_array,
                                                                      remove_seen_flag=self.exclude_seen,
                                                                      cutoff = self.max_cutoff,
                                                                      remove_top_pop_flag=False,
                                                                      remove_custom_items_flag=self.ignore_items_flag,
                                                                      return_scores = True
                                                                     )

            results_dict = self._compute_metrics_on_recommendation_list(test_user_batch_array = test_user_batch_array,
                                                         recommended_items_batch_list = recommended_items_batch_list,
                                                         scores_batch = scores_batch,
                                                         results_dict = results_dict)

        logger.info(f"Finished evaluation. Total time: {time.time() - start_all:.3f} seconds.")

        return results_dict

    def _parallel_run_evaluation_on_selected_users(self,
                                                   recommender_object: BaseRecommender,
                                                   users_to_evaluate,
                                                   block_size: int = None,
                                                   num_workers: int = None) -> Dict:
        """Parallel version of _run_evaluation_on_selected_users.

        This method take batches of users of size <block_size> * <num_workers>. These batches are partitioned into
        blocks of size <block_size> and assigned to each worker to process in parallel.

        Parameters
        ----------
        recommender_object: BaseRecommender
            An instance of a recommender object.
        users_to_evaluate: List
            A list of user indexes to evaluate in the URM.
        block_size
            Number of users to evaluate in the same core.
        num_workers
            Number of threads to use.

        Returns
        -------
        dict
            A dictionary containing the results of the evaluation process. This dictionary looks like the one created by
            the method _create_empty_metrics_dict.

        Note
        ----
        This method can easily consume the available RAM in the machine. The amount of RAM used by this method
        mostly depends on the the memory usage of the <recommender_object.recommend> method. If the process is killed by
        insufficient RAM, use smaller block sizes, less number of workers or memory-profile this method.
        """
        # There are memory peaks that are not captured by running the profiler line by line.
        # These are captured if the software is run using mprof.
        # Environment:
        # * Dataset: CIKM2020 Dataset
        # * Recommender: ItemKNNCFRecommender
        # * Num Workers: 16 CPUs
        # Results:
        #  * block_size = 500 -> Peak: 37GB on 16CPU.
        #  * block_size = 1000 -> each job takes 7seg and 900MB -> If parallel on 16CPU would be 900MB*16=14.4GB
        #  * block_size = 400 -> each job takes 3seg and 350MB -> If parallel on 16CPU would be 350MB*16=5.6GB
        #  * block_size = 10000 -> each job takes: X min and 8.8 GiB -> If parallel on 16CPU would be 8.8*16=140GB
        #  * block_size = 100 -> each job takes 1.5s and 120MB -> If parallel on 16CPU would be 120*16=2GB
        #  * block_size = int(len(users_to_evaluate) / 20000)

        block_size = (recommender_object.evaluation_block_size
                      if recommender_object.evaluation_block_size is not None
                      else min(1000, int(1e8/self.n_items), len(users_to_evaluate)))

        num_workers = (self.num_workers
                       if self.num_workers is not None
                       else int(psutil.cpu_count()))

        logger.debug(f"Block size: {block_size} - Number of workers: {num_workers}")

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0

        empty_metrics_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                        self.n_items,
                                                        self.n_users,
                                                        recommender_object.get_URM_train(),
                                                        self.URM_test,
                                                        self.ignore_items_ID,
                                                        self.ignore_users_ID,
                                                        self.diversity_object)
        intermediate_metrics_dict = copy.deepcopy(empty_metrics_dict)

        start_all = time.time()
        while user_batch_start < len(users_to_evaluate):
            user_batch_end = min(user_batch_start + (block_size * num_workers), len(users_to_evaluate))
            logger.info(f"Processing batch from {user_batch_start} to {user_batch_end} - total users {len(users_to_evaluate)}")

            intermediate_metrics_dict_list = []
            for test_user_task_batch_array in np.array_split(users_to_evaluate[user_batch_start:user_batch_end], num_workers):

                empty_results_dict = dask.delayed(copy.deepcopy)(empty_metrics_dict)

                recommended_items_batch_list, scores_batch = dask.delayed(recommender_object.recommend, nout=2)(
                    test_user_task_batch_array,
                    remove_seen_flag=self.exclude_seen,
                    cutoff=self.max_cutoff,
                    remove_top_pop_flag=False,
                    remove_custom_items_flag=self.ignore_items_flag,
                    return_scores=True)

                delayed_results = dask.delayed(self._parallel_compute_metrics_on_recommendation_list)(
                                                      test_user_batch_array=test_user_task_batch_array,
                                                      recommended_items_batch_list=recommended_items_batch_list,
                                                      scores_batch=scores_batch,
                                                      results_dict=empty_results_dict)

                intermediate_metrics_dict_list.append(delayed_results)

                # start = time.time()
                # test_user_batch_array = np.array(users_to_evaluate[user_batch_start:user_batch_end])
                # print(f"finished test_user_batch_array: {time.time() - start}")
                # user_batch_start = user_batch_end
                #
                # start = time.time()
                # empty_results_dict = copy.deepcopy(results_dict)
                # print(f"finished empty_results_dict: {time.time() - start}")
                #
                # start = time.time()
                # recommended_items_batch_list, scores_batch = recommender_object.recommend(
                #     test_user_batch_array,
                #     remove_seen_flag=self.exclude_seen,
                #     cutoff=self.max_cutoff,
                #     remove_top_pop_flag=False,
                #     remove_custom_items_flag=self.ignore_items_flag,
                #     return_scores=True)
                # print(f"finished recommended_items_batch_list: {time.time() - start}")
                #
                #
                #
                # start = time.time()
                # delayed_results = self._parallel_compute_metrics_on_recommendation_list(
                #                                       test_user_batch_array=test_user_batch_array,
                #                                       recommended_items_batch_list=recommended_items_batch_list,
                #                                       scores_batch=scores_batch,
                #                                       results_dict=empty_results_dict)
                #
                # print(f"finished delayed_results: {time.time() - start}")
                #
                # return delayed_results

            start = time.time()
            logger.info(f"Computing {len(intermediate_metrics_dict_list)} tasks.")
            intermediate_metrics_dict_list = dask.compute(*intermediate_metrics_dict_list)
            logger.info(f"Finished tasks in {time.time() - start:.3f} seconds.")

            intermediate_metrics_dict = _merge_metric_dict_with_list(intermediate_metrics_dict,
                                                                     intermediate_metrics_dict_list,
                                                                     self.cutoff_list)
            user_batch_start = user_batch_end

        logger.info(f"Merging parallel results. Time until now: {time.time() - start_all:.3f} seconds.")
        results_dict = _merge_metric_dict_with_list(empty_metrics_dict,
                                                    [intermediate_metrics_dict],
                                                    self.cutoff_list)
        logger.info(f"Finished evaluation. Total time: {time.time() - start_all:.3f} seconds.")
        return results_dict


def _merge_metric_dict_with_list(metrics_dict: dict, metric_dict_list: list, cutoff_list: list) -> dict:
    """WARNING: This method mutates <metrics_dict>."""
    for intermediate_result_dict in metric_dict_list:
        for cutoff in cutoff_list:
            metrics_dict[cutoff][EvaluatorMetrics.ROC_AUC.value] += (
                intermediate_result_dict[cutoff][EvaluatorMetrics.ROC_AUC.value])

            metrics_dict[cutoff][EvaluatorMetrics.PRECISION.value] += (
                intermediate_result_dict[cutoff][EvaluatorMetrics.PRECISION.value])

            metrics_dict[cutoff][EvaluatorMetrics.PRECISION_RECALL_MIN_DEN.value] += (
                intermediate_result_dict[cutoff][EvaluatorMetrics.PRECISION_RECALL_MIN_DEN.value])

            metrics_dict[cutoff][EvaluatorMetrics.RECALL.value] += (
                intermediate_result_dict[cutoff][EvaluatorMetrics.RECALL.value])

            metrics_dict[cutoff][EvaluatorMetrics.NDCG.value] += (
                intermediate_result_dict[cutoff][EvaluatorMetrics.NDCG.value])

            metrics_dict[cutoff][EvaluatorMetrics.HIT_RATE.value] += (
                intermediate_result_dict[cutoff][EvaluatorMetrics.HIT_RATE.value])

            metrics_dict[cutoff][EvaluatorMetrics.ARHR.value] += (
                intermediate_result_dict[cutoff][EvaluatorMetrics.ARHR.value])

            metrics_dict[cutoff][EvaluatorMetrics.MRR.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.MRR.value])

            metrics_dict[cutoff][EvaluatorMetrics.MAP.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.MAP.value])

            metrics_dict[cutoff][EvaluatorMetrics.NOVELTY.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.NOVELTY.value])

            metrics_dict[cutoff][EvaluatorMetrics.AVERAGE_POPULARITY.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.AVERAGE_POPULARITY.value])

            metrics_dict[cutoff][EvaluatorMetrics.DIVERSITY_GINI.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.DIVERSITY_GINI.value])

            metrics_dict[cutoff][EvaluatorMetrics.SHANNON_ENTROPY.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.SHANNON_ENTROPY.value])

            metrics_dict[cutoff][EvaluatorMetrics.COVERAGE_ITEM.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.COVERAGE_ITEM.value])

            metrics_dict[cutoff][EvaluatorMetrics.COVERAGE_USER.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.COVERAGE_USER.value])

            metrics_dict[cutoff][EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value])

            metrics_dict[cutoff][EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].merge_with_other(
                other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.DIVERSITY_HERFINDAHL.value])

            if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in metrics_dict[cutoff]:
                metrics_dict[cutoff][EvaluatorMetrics.DIVERSITY_SIMILARITY.value].merge_with_other(
                    other_metric_object=intermediate_result_dict[cutoff][EvaluatorMetrics.DIVERSITY_SIMILARITY.value]
                )

    return metrics_dict


class EvaluatorNegativeItemSample(Evaluator):
    """EvaluatorNegativeItemSample"""

    EVALUATOR_NAME = "EvaluatorNegativeItemSample"

    def __init__(self, URM_test_list, URM_test_negative, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None):
        """

        The EvaluatorNegativeItemSample computes the recommendations by sorting the test items as well as the test_negative items
        It ensures that each item appears only once even if it is listed in both matrices

        :param URM_test_list:
        :param URM_test_negative: Items to rank together with the test items
        :param cutoff_list:
        :param min_ratings_per_user:
        :param exclude_seen:
        :param diversity_object:
        :param ignore_items:
        :param ignore_users:
        """
        super(EvaluatorNegativeItemSample, self).__init__(URM_test_list, cutoff_list,
                                                          diversity_object = diversity_object,
                                                          min_ratings_per_user = min_ratings_per_user, exclude_seen=exclude_seen,
                                                          ignore_items = ignore_items, ignore_users = ignore_users)


        self.URM_items_to_rank = sps.csr_matrix(self.URM_test.copy().astype(np.bool)) + sps.csr_matrix(URM_test_negative.copy().astype(np.bool))
        self.URM_items_to_rank.eliminate_zeros()
        self.URM_items_to_rank.data = np.ones_like(self.URM_items_to_rank.data)



    def _get_user_specific_items_to_compute(self, user_id):

        start_pos = self.URM_items_to_rank.indptr[user_id]
        end_pos = self.URM_items_to_rank.indptr[user_id+1]

        items_to_compute = self.URM_items_to_rank.indices[start_pos:end_pos]

        return items_to_compute



    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size = None):



        results_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                  self.n_items, self.n_users,
                                                  recommender_object.get_URM_train(),
                                                  self.URM_test,
                                                  self.ignore_items_ID,
                                                  self.ignore_users_ID,
                                                  self.diversity_object)


        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)



        for test_user in users_to_evaluate:

            items_to_compute = self._get_user_specific_items_to_compute(test_user)

            recommended_items, all_items_predicted_ratings = recommender_object.recommend(np.atleast_1d(test_user),
                                                              remove_seen_flag=self.exclude_seen,
                                                              cutoff = self.max_cutoff,
                                                              remove_top_pop_flag=False,
                                                              items_to_compute = items_to_compute,
                                                              remove_custom_items_flag=self.ignore_items_flag,
                                                              return_scores = True
                                                             )


            results_dict = self._compute_metrics_on_recommendation_list(test_user_batch_array = [test_user],
                                                         recommended_items_batch_list = recommended_items,
                                                         scores_batch = all_items_predicted_ratings,
                                                         results_dict = results_dict)


        return results_dict

