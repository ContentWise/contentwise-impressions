#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

import logging

from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps

from Utils.decorator import timeit


logger = logging.getLogger("contentwise-impressions")


class PureSVDRecommender(BaseMatrixFactorizationRecommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train, verbose=True, evaluation_block_size: int = 500):
        super(PureSVDRecommender, self).__init__(URM_train,
                                                 verbose=verbose,
                                                 evaluation_block_size=evaluation_block_size)

    @timeit
    def fit(self, num_factors=100, random_seed=None):
        logger.info("Computing SVD decomposition.")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      random_state=random_seed)

        s_Vt = sps.diags(Sigma)*VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        logger.info("Computing SVD decomposition done!")


