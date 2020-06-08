import json
import logging
import os
from enum import Enum
from typing import Optional, Tuple

import dask
import dask.dataframe as ddf
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_csr

from .decorator import timeit
from .gini import gini

STATISTICS_PATH = "statistics"
DATASET_PATH = "data"

logger = logging.getLogger("contentwise-impressions")


class Dataset:
    def __init__(self):
        self._local_interactions_file_path: str = ""
        self._local_statistics_file_path: str = ""

        self.name = "ABSTRACT_DATASET"
        self.URM: dict = dict()
        self.statistics: dict = dict()

    @staticmethod
    def file_or_dir_exists(path_to_test: str) -> bool:
        return os.path.exists(path_to_test)

    @staticmethod
    def _create_folder(dir_to_create: str):
        os.makedirs(dir_to_create, exist_ok=True)
        logger.info(f"Created folder {dir_to_create}")

    def _read_statistics(self) -> None:
        if not Dataset.file_or_dir_exists(self._local_statistics_file_path):
            self.statistics = dict()
            return

        with open(self._local_statistics_file_path, "r") as f:
            self.statistics = json.load(f)

    def _save_statistics(self) -> None:
        with open(self._local_statistics_file_path, "w") as f:
            json.dump(self.statistics, f, indent=2, sort_keys=True)
        logger.info(f"Saved statistics in {self._local_statistics_file_path}")


class ContentWiseImpressions(Dataset):
    """Dataset published by ContentWise on CIKM2020.

    This dataset contains interactions and impressions from an Over-the-Top media service. The dataset contains
    10 million interactions of users with items related to the cinema and television. It also has impressions of
    items presented to the users, the length of recommendations, row_position. We collected two different types of
    impressions: impressions with and without direct links to interactions.

    The code contains methods to read the dataset from local disk, generate URM matrices, training/validation/testing splits,
     save them URMs to disk, and more.

    This dataset was published in the article "ContentWise Impressions: An industrial dataset with impressions included"
    by F.B. Pérez Maurera, Maurizio Ferrari Dacrema, Lorenzo Saule, Mario Scriminaci, and Paolo Cremonesi. If you use
    this code or the dataset, please reference our work.

    @Article{ContentWiseImpressions,
        author={Pérez Maurera, Fernando Benjamín
            and Ferrari Dacrema, Maurizio
            and Saule, Lorenzo
            and Scriminaci, Mario
            and Cremonesi, Paolo},
        title={ContentWise Impressions: An industrial dataset with impressions included},
        journal={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM 2020)},
        year={2020},
        doi={},
        Eprint={arXiv},
        note={Source: \\url{https://github.com/ContentWise/contentwise-impressions}},
    }

    The following are the types presented in the dataset.

    Interaction columns
    --------
    utc_ts_milliseconds (index): int64
        UTC Unix timestamps of interactions.
    user_id: int32.
        Anonymized identifier of users.
    item_id: int32.
        Anonymized identifier of items.
    item_type`: int8
        Classification of the item. It has 4 possible values.
            0: Movies.
            1: Movies and clips in series.
            2: TV Movies or shows.
            3: Episodes of TV Series.
    series_id: int32.
        Anonymized identifier of series.
    episode_number: int32.
        Episode number of the item inside a series.
    series_length: int32.
        Number of episodes of the series.
    recommendation_id: int32.
        Identifier of recommendation presented to the user.
    interaction_type`: int8
        Classification of the interaction. It has 4 possible values.
            0: The user viewed an item.
            1: The user accessed an item.
            2: The user rated an item.
            3: The user purchased an item.
    explicit_rating: float32
        Rating that the user gave to an item. Ranges from 0 to 5 with steps of 0.5
    vision_factor: float32
        Reflects how much a user viewed an item based on the item's duration. Ranges from 0 to 1.

    Impressions with direct links columns
    --------
    recommendation_id : int32.
        (index) Anonymized identifier of recommendation presented to the user.
    row_position : int32
        Position on screen of recommendation.
    recommendation_list_length : int32
        Number of recommended items.
    recommended_series_list : List[int32]
        Ordered recommendation list of series ids. Series on the first positions are considered to be more
        meaningful to users.

    Impressions without direct links columns
    --------
    user_id : int32.
        (index) Anonymized user identifier that received the recommendations.
    row_position : int32
        Position on screen of recommendation.
    recommendation_list_length : int32
        Number of recommended items.
    recommended_series_list : List[int32]
        Ordered recommendation list of series ids. Series on the first positions are considered to be more more
        meaningful to users.
    """

    class Variant(Enum):
        CW10M = "CW10M"

    def __init__(self, dataset_variant: Variant):
        super().__init__()

        if dataset_variant not in ContentWiseImpressions.Variant:
            raise ValueError(
                f"Dataset variant {dataset_variant} is not valid. Accepted values are: {[variant for variant in ContentWiseImpressions.Variant]}")

        self.statistics_name = "statistics.json"
        self.variant = dataset_variant.value
        self.name = "ContentWiseImpressions"

        self.interactions: Optional[ddf.DataFrame] = None
        self.impressions_direct_link: Optional[ddf.DataFrame] = None
        self.impressions_non_direct_link: Optional[ddf.DataFrame] = None
        self.metadata: Optional[dict] = None

        self._dataset_splits_folder_name = "splits"

        self._interactions_name = "interactions"
        self._impressions_direct_link_name = "impressions-direct-link"
        self._impressions_non_direct_link_name = "impressions-non-direct-link"
        self._metadata_name = "metadata.json"

        self._local_statistics_dir = os.path.join(os.getcwd(),
                                                  STATISTICS_PATH,
                                                  self.name,
                                                  self.variant)
        self._local_statistics_file_path = os.path.join(self._local_statistics_dir,
                                                        self.statistics_name)

        self._local_dataset_dir = os.path.join(os.getcwd(),
                                               DATASET_PATH,
                                               self.name,
                                               self.variant)
        self._local_dataset_splits_dir = os.path.join(self._local_dataset_dir,
                                                      self._dataset_splits_folder_name)

        self._local_interactions_file_path = os.path.join(self._local_dataset_dir,
                                                          self._interactions_name)
        self._local_impressions_direct_link_file_path = os.path.join(self._local_dataset_dir,
                                                                     self._impressions_direct_link_name)
        self._local_non_impressions_non_direct_link_file_path = os.path.join(self._local_dataset_dir,
                                                                             self._impressions_non_direct_link_name)
        self._local_metadata_file_path = os.path.join(self._local_dataset_dir,
                                                      self._metadata_name)

        self.valid_splits = ["train",
                             "test",
                             "validation",
                             "impressions",
                             "impressions_direct_link",
                             "impressions_non_direct_link"]

        Dataset._create_folder(self._local_statistics_dir)
        Dataset._create_folder(self._local_dataset_dir)
        Dataset._create_folder(self._local_dataset_splits_dir)

    @timeit
    def read_dataset(self) -> None:
        def _read_parquet(element: str, local_file_path: str) -> ddf.DataFrame:
            try:
                return ddf.read_parquet(path=local_file_path, engine="pyarrow")
            except OSError as e:
                logger.exception(f"{element} is not present at {local_file_path}. Please download the data folder "
                                 f"following the instructions of the README")
                raise e

        def _read_metadata(local_file_path: str) -> dict:
            try:
                with open(local_file_path, "r") as f:
                    return json.load(f)
            except OSError as e:
                logger.exception(f"Could not open dataset metadata. If it is not present at {local_file_path}, then"
                                 f"please download the data folder following the instructions of the README")
                raise e

        self.interactions = _read_parquet("Interactions",
                                          self._local_interactions_file_path)

        self.impressions_direct_link = _read_parquet("Impressions with a direct link to Interactions",
                                                     self._local_impressions_direct_link_file_path)

        self.impressions_non_direct_link = _read_parquet("Impressions without a direct link to Interactions",
                                                         self._local_non_impressions_non_direct_link_file_path)

        self.metadata = _read_metadata(self._local_metadata_file_path)

    @timeit
    def save_dataset(self) -> None:
        def _save_parquet(element_obj: ddf.DataFrame, element: str, local_file_path: str):
            if element_obj is None:
                raise ValueError(f"Cannot save {element} if it's not loaded")

            ddf.to_parquet(element_obj,
                           local_file_path,
                           engine="pyarrow")

        def _save_metadata(metadata: dict, local_file_path: str):
            if metadata is None:
                raise ValueError("Cannot save metadata if it's not loaded.")

            with open(local_file_path, "w") as f:
                json.dump(obj=metadata, fp=f)

        _save_parquet(self.interactions,
                      "Interactions",
                      self._local_interactions_file_path)

        _save_parquet(self.impressions_direct_link,
                      "Impressions with a direct link to Interactions",
                      self._local_impressions_direct_link_file_path)

        _save_parquet(self.impressions_non_direct_link,
                      "Impressions without a direct link to Interactions",
                      self._local_non_impressions_non_direct_link_file_path)

        _save_metadata(self.metadata, self._local_metadata_file_path)

    @timeit
    def read_urm_splits(self, use_items: bool = True, use_cache: bool = False) -> None:
        logger.debug(f"Use_items: {use_items} - Use_cache: {use_cache}")

        try:
            if use_cache:
                self._read_cached_urm_splits(use_items=use_items)
            else:
                self._generate_urm_interactions(use_items=use_items)
                self._generate_urm_impressions(use_items=use_items)
        except IOError:
            logger.info(f"Splits does not exists locally - Will generate them")
            self._generate_urm_interactions(use_items=use_items)
            self._generate_urm_impressions(use_items=use_items)

    @timeit
    def save_urm(self, use_items: bool = True) -> None:
        logger.debug(f"Use_items: {use_items}")

        for split_name, split in self.URM.items():
            filename = os.path.join(self._local_dataset_splits_dir,
                                    f"urm_{'items' if use_items else 'series'}.{split_name}.npz")

            sp.sparse.save_npz(file=filename, matrix=split, compressed=True)

            logger.info(f"Saved split {split_name} at {filename}")

    def _read_cached_urm_splits(self, use_items: bool) -> None:
        logger.debug(f"Use_items: {use_items}")

        for split_name in self.valid_splits:
            filename = os.path.join(self._local_dataset_splits_dir,
                                    f"urm_{'items' if use_items else 'series'}.{split_name}.npz")

            self.URM[split_name] = sp.sparse.load_npz(filename)

            logger.info(f"Read split {split_name} from {filename}")

    def _generate_urm_interactions(self, use_items: bool) -> None:
        logger.debug(f"Use_items: {use_items}")

        if self.interactions is None or self.metadata is None:
            self.read_dataset()

        user_column = "user_id"
        item_column = "item_id" if use_items else "series_id"

        num_users = self.metadata["num_users"]
        num_items = self.metadata["num_items" if use_items else "num_series"]

        num_interactions = self.interactions.shape[0]

        logger.info(f"Removing duplicates and performing random split")
        train_split, validation_split, test_split = (self.interactions
                                                     .drop_duplicates(subset=[user_column, item_column])
                                                     .random_split(frac=[0.7, 0.1, 0.2], random_state=42))

        train_data = dask.delayed(np.ones)(train_split.shape[0], dtype=np.int32)
        train_split = dask.delayed(csr_matrix)((train_data,
                                                (train_split[user_column],
                                                 train_split[item_column])),
                                               shape=(num_users, num_items),
                                               dtype=np.int32)

        validation_data = dask.delayed(np.ones)(validation_split.shape[0], dtype=np.int32)
        validation_split = dask.delayed(csr_matrix)((validation_data,
                                                     (validation_split[user_column],
                                                      validation_split[item_column])),
                                                    shape=(num_users, num_items),
                                                    dtype=np.int32)

        test_data = dask.delayed(np.ones)(test_split.shape[0], dtype=np.int32)
        test_split = dask.delayed(csr_matrix)((test_data,
                                               (test_split[user_column],
                                                test_split[item_column])),
                                              shape=(num_users, num_items),
                                              dtype=np.int32)

        logger.info("Generating training, validation, testing splits.")
        (train_split,
         validation_split,
         test_split,
         num_users,
         num_items,
         num_interactions) = dask.compute(train_split,
                                          validation_split,
                                          test_split,
                                          num_users,
                                          num_items,
                                          num_interactions)

        logger_message = ("Interaction statistics - "
                          f"# Original interactions: {num_interactions} - "
                          f"Unique users: {num_users} - "
                          f"Unique {'items' if use_items else 'series'}: {num_items} - "
                          f"Unique interactions: {train_split.nnz + validation_split.nnz + test_split.nnz} - "
                          f"# Interactions in train: {train_split.nnz} - "
                          f"# Interactions in validation: {validation_split.nnz} - "
                          f"# Interactions in test: {test_split.nnz}.")
        logger.info(logger_message)

        if train_split.shape != validation_split.shape or train_split.shape != test_split.shape:
            logger_message = ("Reshaping splits - "
                              f"Desired Shape: {(num_users, num_items)} - "
                              f"Current Train Split Shape: {train_split.shape} - "
                              f"Current Validation Split Shape: {validation_split.shape} - "
                              f"Current Test Split Shape: {test_split.shape}.")
            logger.warning(logger_message)

            train_split = train_split.reshape((num_users, num_items))
            validation_split = validation_split.reshape((num_users, num_items))
            test_split = test_split.reshape((num_users, num_items))

        if not isspmatrix_csr(train_split) or not isspmatrix_csr(validation_split) or not isspmatrix_csr(test_split):
            logger_message = ("Reshaping splits - "
                              f"Desired Format: CSR - "
                              f"Current Train Split format: {train_split.getformat()} - "
                              f"Current Validation Split format: {validation_split.getformat()} - "
                              f"Current Test Split format: {test_split.getformat()}.")
            logger.warning(logger_message)

            train_split = train_split.tocsr()
            validation_split = validation_split.tocsr()
            test_split = test_split.tocsr()

        self.URM = {
            "train": train_split,
            "test": validation_split,
            "validation": test_split
        }

    def _generate_urm_impressions(self, use_items: bool) -> None:
        def calculate_urm_impressions_direct_link() -> Tuple[csr_matrix, int]:
            if use_items:
                impressions_direct_link_recommendation_id_series_pairs: ddf.DataFrame = (
                    self.impressions_direct_link[["recommended_series_list"]]
                        .rename(columns={"recommended_series_list": "series_id"})
                        .explode("series_id")
                        .astype({"series_id": "int32"})
                        .reset_index(drop=False)
                )

                impressions_direct_link_recommendation_id_item_pairs = (
                    impressions_direct_link_recommendation_id_series_pairs
                    .merge(right=item_to_series_mapping,
                           how="inner",
                           left_on="series_id",
                           right_on="series_id"))

            else:
                impressions_direct_link_recommendation_id_item_pairs: ddf.DataFrame = (
                    self.impressions_direct_link[["recommended_series_list"]]
                        .rename(columns={"recommended_series_list": "series_id"})
                        .explode("series_id")
                        .astype({"series_id": "int32"})
                        .reset_index(drop=False)
                )

            impressions_direct_link_user_item_pairs: ddf.DataFrame = (
                user_recommendation_id_pairs.merge(right=impressions_direct_link_recommendation_id_item_pairs,
                                                   how="inner",
                                                   left_on="recommendation_id",
                                                   right_on="recommendation_id")
            )

            impressions_direct_link_user_item_pairs: ddf.DataFrame = impressions_direct_link_user_item_pairs[
                [user_column, item_column]]

            data_size = impressions_direct_link_user_item_pairs.shape[0]

            impressions_direct_link_data = dask.delayed(np.ones)(data_size,
                                                                dtype=np.int32)
            impressions_direct_link_urm = dask.delayed(csr_matrix)((impressions_direct_link_data,
                                                                   (impressions_direct_link_user_item_pairs[user_column],
                                                                    impressions_direct_link_user_item_pairs[item_column])),
                                                                  shape=(num_users, num_items),
                                                                  dtype=np.int32)

            logger.info("Generating interacted-impressions split.")
            (impressions_direct_link_urm,
             data_size,) = dask.compute(impressions_direct_link_urm,
                                        data_size)
            return impressions_direct_link_urm, data_size

        def calculate_urm_impressions_non_direct_link() -> Tuple[csr_matrix, int]:
            if use_items:
                impressions_non_direct_link_user_series_pairs: ddf.DataFrame = (
                    self.impressions_non_direct_link[["recommended_series_list"]]
                        .rename(columns={"recommended_series_list": "series_id"})
                        .explode("series_id")
                        .astype({"series_id": "int32"})
                        .reset_index(drop=False)
                )

                impressions_non_direct_link_user_item_pairs = (impressions_non_direct_link_user_series_pairs
                                                              .merge(right=item_to_series_mapping,
                                                                     how="inner",
                                                                     left_on="series_id",
                                                                     right_on="series_id"))
            else:
                impressions_non_direct_link_user_item_pairs: ddf.DataFrame = (
                    self.impressions_non_direct_link[["recommended_series_list"]]
                        .rename(columns={"recommended_series_list": "series_id"})
                        .explode("series_id")
                        .astype({"series_id": "int32"})
                        .reset_index(drop=False)
                )

            impressions_non_direct_link_user_item_pairs: ddf.DataFrame = impressions_non_direct_link_user_item_pairs[
                [user_column, item_column]]

            logger.info("Creating delayed partitions")
            list_of_delayed_partitions = impressions_non_direct_link_user_item_pairs.to_delayed()

            impressions_non_direct_link_urm = csr_matrix((num_users, num_items), dtype=np.int32)
            data_size = 0

            # This processing is MEMORY CONSUMING if done in parallel (more than 120GB RAM with 16 cores). We treat each
            # dask partition sequentially in order to avoid process killing.
            num_partitions = len(list_of_delayed_partitions)
            logger.info(f"Will process {num_partitions} impressions without direct links partitions sequentially. This will "
                        f"take some minutes.")
            for i in range(num_partitions):
                logger.info(f"Processing partition {i} of {num_partitions}")

                delayed = list_of_delayed_partitions[i]
                (impressions_non_direct_link_partition,) = dask.compute(delayed)

                intermediate_data = np.ones(impressions_non_direct_link_partition.shape[0], dtype=np.int32)
                intermediate_split = csr_matrix((intermediate_data,
                                                 (impressions_non_direct_link_partition[user_column],
                                                  impressions_non_direct_link_partition[item_column])),
                                                shape=(num_users, num_items),
                                                dtype=np.int32)

                impressions_non_direct_link_urm += intermediate_split
                data_size += impressions_non_direct_link_partition.shape[0]

            return impressions_non_direct_link_urm, data_size

        logger.debug(f"Use_items: {use_items}")

        if (self.interactions is None
                or self.impressions_direct_link is None
                or self.impressions_non_direct_link is None
                or self.metadata is None):
            self.read_dataset()

        user_column = "user_id"
        item_column = "item_id" if use_items else "series_id"

        num_users = self.metadata["num_users"]
        num_items = self.metadata["num_items" if use_items else "num_series"]

        num_interactions = self.interactions.shape[0]
        num_impressions_direct_link = self.impressions_direct_link.shape[0]
        num_impressions_non_direct_link = self.impressions_non_direct_link.shape[0]

        logger.info("Calculating basic metrics")
        (num_users,
         num_items,
         num_interactions,
         num_impressions_direct_link,
         num_impressions_non_direct_link,) = dask.compute(num_users,
                                                         num_items,
                                                         num_interactions,
                                                         num_impressions_direct_link,
                                                         num_impressions_non_direct_link)

        user_recommendation_id_pairs: ddf.DataFrame = self.interactions[["user_id", "recommendation_id"]]
        item_to_series_mapping: ddf.DataFrame = (self.interactions[["item_id", "series_id"]]
                                                 .drop_duplicates())

        logger.info("Beginning impressions with direct links")
        impressions_direct_link_split, num_impressions_direct_link_user_item_pairs = calculate_urm_impressions_direct_link()

        logger.info("Beginning impressions without direct links")
        impressions_non_direct_link_split, num_impressions_non_direct_link_user_item_pairs = calculate_urm_impressions_non_direct_link()

        impressions_split = impressions_direct_link_split + impressions_non_direct_link_split
        num_impressions_user_item_pairs = num_impressions_direct_link_user_item_pairs + num_impressions_non_direct_link_user_item_pairs

        logger_message = ("Impression statistics"
                          f"\n# Original interactions: {num_interactions}"
                          f"\n# Original impressions with direct links: {num_impressions_direct_link}"
                          f"\n# Original impressions without direct links: {num_impressions_non_direct_link}"
                          f"\nUnique users: {num_users}"
                          f"\nUnique {'items' if use_items else 'series'}: {num_items}"
                          f"\nNum impressions: {num_impressions_user_item_pairs}"
                          f"\nNum interacted-impressions: {num_impressions_direct_link_user_item_pairs}"
                          f"\nNum non-interacted-impressions: {num_impressions_non_direct_link_user_item_pairs}"
                          f"\nNum non-zero impressions: {impressions_split.nnz}."
                          f"\nNum non-zero impressions with direct links: {impressions_direct_link_split.nnz}."
                          f"\nNum non-zero impressions without direct links: {impressions_non_direct_link_split.nnz}."
                          )
        logger.info(logger_message)

        if (impressions_split.shape != (num_users, num_items)
                or impressions_direct_link_split.shape != (num_users, num_items)
                or impressions_non_direct_link_split.shape != (num_users, num_items)):
            logger_message = ("Reshaping splits"
                              f"\nDesired Shape: {(num_users, num_items)}"
                              f"\nCurrent Impressions Split Shape: {impressions_split.shape}"
                              f"\nCurrent impressions with direct links Split Shape: {impressions_direct_link_split.shape}"
                              f"\nCurrent Non-InteractedImpressions Split Shape: {impressions_non_direct_link_split.shape}"
                              )
            logger.warning(logger_message)

            impressions_split = impressions_split.reshape((num_users, num_items))
            impressions_direct_link_split = impressions_direct_link_split.reshape((num_users, num_items))
            impressions_non_direct_link_split = impressions_non_direct_link_split.reshape((num_users, num_items))

        if (not isspmatrix_csr(impressions_split)
                or not isspmatrix_csr(impressions_direct_link_split)
                or not isspmatrix_csr(impressions_non_direct_link_split)):
            logger_message = ("Reshaping splits - "
                              f"\nDesired Format: CSR - "
                              f"\nCurrent Impressions Split format: {impressions_split.getformat()}."
                              f"\nCurrent impressions with direct links Split format: {impressions_direct_link_split.getformat()}."
                              f"\nCurrent impressions without direct links Split format: {impressions_non_direct_link_split.getformat()}."
                              )
            logger.warning(logger_message)

            impressions_split = impressions_split.tocsr()
            impressions_direct_link_split = impressions_direct_link_split.tocsr()
            impressions_non_direct_link_split = impressions_non_direct_link_split.tocsr()

        self.URM["impressions"] = impressions_split
        self.URM["impressions_direct_link"] = impressions_direct_link_split
        self.URM["impressions_non_direct_link"] = impressions_non_direct_link_split

    @timeit
    def initialize_statistics_dictionary(self):
        if self.statistics is None:
            self._read_statistics()

        if len(self.statistics) != 0:
            raise ValueError("The statistics dict cannot be initialized as it already has information.")

        self.statistics["general"] = dict()
        self.statistics["general"]["gini"] = dict()
        self.statistics["general"]["density"] = dict()
        self.statistics["general"]["sparsity"] = dict()
        self.statistics["general"]["residual"] = dict()

        self.statistics["interactions"] = dict()
        self.statistics["interactions"]["user_id"] = dict()
        self.statistics["interactions"]["item_id"] = dict()
        self.statistics["interactions"]["series_id"] = dict()
        self.statistics["interactions"]["recommendation_id"] = dict()
        self.statistics["interactions"]["vision_factor"] = dict()
        self.statistics["interactions"]["explicit_rating"] = dict()

        self.statistics["interactions_only_impressions"] = dict()
        self.statistics["interactions_only_impressions"]["user_id"] = dict()
        self.statistics["interactions_only_impressions"]["item_id"] = dict()
        self.statistics["interactions_only_impressions"]["series_id"] = dict()
        self.statistics["interactions_only_impressions"]["recommendation_id"] = dict()

        self.statistics["impressions_direct_link"] = dict()
        self.statistics["impressions_direct_link"]["row_position"] = dict()
        self.statistics["impressions_direct_link"]["recommendation_list_length"] = dict()
        self.statistics["impressions_direct_link"]["recommended_series_list"] = dict()

        self.statistics["impressions_non_direct_link"] = dict()
        self.statistics["impressions_non_direct_link"]["row_position"] = dict()
        self.statistics["impressions_non_direct_link"]["recommendation_list_length"] = dict()
        self.statistics["impressions_non_direct_link"]["recommended_series_list"] = dict()

        self._save_statistics()

    @timeit
    def paper_statistics(self):
        def statistics_interactions(interactions: ddf.DataFrame):
            agg_user_id_all = (interactions
                               .groupby("user_id")
                               .item_id
                               .agg(["count"]))

            agg_item_id_all = (interactions
                               .groupby("item_id")
                               .user_id
                               .agg(["count"]))

            agg_series_id_all = (interactions
                                 .groupby("series_id")
                                 .item_id
                                 .agg(["count"]))

            agg_recommendation_id_all = (interactions
                                         .groupby("recommendation_id")
                                         .user_id
                                         .agg(["count"]))
            
            mean_user_id = agg_user_id_all["count"].mean()
            min_user_id = agg_user_id_all["count"].min()
            max_user_id = agg_user_id_all["count"].max()

            mean_item_id = agg_item_id_all["count"].mean()
            min_item_id = agg_item_id_all["count"].min()
            max_item_id = agg_item_id_all["count"].max()

            mean_series_id = agg_series_id_all["count"].mean()
            min_series_id = agg_series_id_all["count"].min()
            max_series_id = agg_series_id_all["count"].max()

            mean_recommendation_id = agg_recommendation_id_all["count"].mean()
            min_recommendation_id = agg_recommendation_id_all["count"].min()
            max_recommendation_id = agg_recommendation_id_all["count"].max()

            return (mean_user_id,
                    min_user_id,
                    max_user_id,
                    mean_item_id,
                    min_item_id,
                    max_item_id,
                    mean_series_id,
                    min_series_id,
                    max_series_id,
                    mean_recommendation_id,
                    min_recommendation_id,
                    max_recommendation_id,
                    )

        def statistics_impressions(impressions: ddf.DataFrame):
            agg_row_position = (impressions
                                .groupby("row_position")
                                .recommendation_list_length
                                .agg(["count"]))

            agg_recommendation_list_length = (impressions
                                              .groupby("recommendation_list_length")
                                              .row_position
                                              .agg(["count"]))

            agg_recommended_series = (impressions
                                      .explode("recommended_series_list")
                                      .groupby("recommended_series_list")
                                      .row_position
                                      .agg(["count"]))

            mean_row_position = agg_row_position["count"].mean()
            min_row_position = agg_row_position["count"].min()
            max_row_position = agg_row_position["count"].max()

            mean_recommendation_list_length = agg_recommendation_list_length["count"].mean()
            min_recommendation_list_length = agg_recommendation_list_length["count"].min()
            max_recommendation_list_length = agg_recommendation_list_length["count"].max()

            mean_recommended_series = agg_recommended_series["count"].mean()
            min_recommended_series = agg_recommended_series["count"].min()
            max_recommended_series = agg_recommended_series["count"].max()

            return (mean_row_position,
                    min_row_position,
                    max_row_position,
                    mean_recommendation_list_length,
                    min_recommendation_list_length,
                    max_recommendation_list_length,
                    mean_recommended_series,
                    min_recommended_series,
                    max_recommended_series,
                    )

        self._read_statistics()

        all_interactions: ddf.DataFrame = self.interactions
        interactions_with_impressions: ddf.DataFrame = self.interactions[self.interactions.recommendation_id >= 0]

        (mean_user_id_all_interactions,
         min_user_id_all_interactions,
         max_user_id_all_interactions,
         mean_item_id_all_interactions,
         min_item_id_all_interactions,
         max_item_id_all_interactions,
         mean_series_id_all_interactions,
         min_series_id_all_interactions,
         max_series_id_all_interactions,
         mean_recommendation_id_all_interactions,
         min_recommendation_id_all_interactions,
         max_recommendation_id_all_interactions,
         ) = statistics_interactions(all_interactions)

        (mean_user_id_interactions_with_impressions,
         min_user_id_interactions_with_impressions,
         max_user_id_interactions_with_impressions,
         mean_item_id_interactions_with_impressions,
         min_item_id_interactions_with_impressions,
         max_item_id_interactions_with_impressions,
         mean_series_id_interactions_with_impressions,
         min_series_id_interactions_with_impressions,
         max_series_id_interactions_with_impressions,
         mean_recommendation_id_interactions_with_impressions,
         min_recommendation_id_interactions_with_impressions,
         max_recommendation_id_interactions_with_impressions,
         ) = statistics_interactions(interactions_with_impressions)

        (mean_row_position_impressions_direct_link,
         min_row_position_impressions_direct_link,
         max_row_position_impressions_direct_link,
         mean_recommendation_list_length_impressions_direct_link,
         min_recommendation_list_length_impressions_direct_link,
         max_recommendation_list_length_impressions_direct_link,
         mean_recommended_series_impressions_direct_link,
         min_recommended_series_impressions_direct_link,
         max_recommended_series_impressions_direct_link,
         ) = statistics_impressions(self.impressions_direct_link)

        (mean_row_position_impressions_non_direct_link,
         min_row_position_impressions_non_direct_link,
         max_row_position_impressions_non_direct_link,
         mean_recommendation_list_length_impressions_non_direct_link,
         min_recommendation_list_length_impressions_non_direct_link,
         max_recommendation_list_length_impressions_non_direct_link,
         mean_recommended_series_impressions_non_direct_link,
         min_recommended_series_impressions_non_direct_link,
         max_recommended_series_impressions_non_direct_link,
         ) = statistics_impressions(self.impressions_non_direct_link)

        (mean_user_id_all_interactions,
         min_user_id_all_interactions,
         max_user_id_all_interactions,
         mean_item_id_all_interactions,
         min_item_id_all_interactions,
         max_item_id_all_interactions,
         mean_series_id_all_interactions,
         min_series_id_all_interactions,
         max_series_id_all_interactions,
         mean_recommendation_id_all_interactions,
         min_recommendation_id_all_interactions,
         max_recommendation_id_all_interactions,

         mean_user_id_interactions_with_impressions,
         min_user_id_interactions_with_impressions,
         max_user_id_interactions_with_impressions,
         mean_item_id_interactions_with_impressions,
         min_item_id_interactions_with_impressions,
         max_item_id_interactions_with_impressions,
         mean_series_id_interactions_with_impressions,
         min_series_id_interactions_with_impressions,
         max_series_id_interactions_with_impressions,
         mean_recommendation_id_interactions_with_impressions,
         min_recommendation_id_interactions_with_impressions,
         max_recommendation_id_interactions_with_impressions,

         mean_row_position_impressions_direct_link,
         min_row_position_impressions_direct_link,
         max_row_position_impressions_direct_link,
         mean_recommendation_list_length_impressions_direct_link,
         min_recommendation_list_length_impressions_direct_link,
         max_recommendation_list_length_impressions_direct_link,
         mean_recommended_series_impressions_direct_link,
         min_recommended_series_impressions_direct_link,
         max_recommended_series_impressions_direct_link,

         mean_row_position_impressions_non_direct_link,
         min_row_position_impressions_non_direct_link,
         max_row_position_impressions_non_direct_link,
         mean_recommendation_list_length_impressions_non_direct_link,
         min_recommendation_list_length_impressions_non_direct_link,
         max_recommendation_list_length_impressions_non_direct_link,
         mean_recommended_series_impressions_non_direct_link,
         min_recommended_series_impressions_non_direct_link,
         max_recommended_series_impressions_non_direct_link,

         ) = ddf.compute(mean_user_id_all_interactions,
                         min_user_id_all_interactions,
                         max_user_id_all_interactions,
                         mean_item_id_all_interactions,
                         min_item_id_all_interactions,
                         max_item_id_all_interactions,
                         mean_series_id_all_interactions,
                         min_series_id_all_interactions,
                         max_series_id_all_interactions,
                         mean_recommendation_id_all_interactions,
                         min_recommendation_id_all_interactions,
                         max_recommendation_id_all_interactions,

                         mean_user_id_interactions_with_impressions,
                         min_user_id_interactions_with_impressions,
                         max_user_id_interactions_with_impressions,
                         mean_item_id_interactions_with_impressions,
                         min_item_id_interactions_with_impressions,
                         max_item_id_interactions_with_impressions,
                         mean_series_id_interactions_with_impressions,
                         min_series_id_interactions_with_impressions,
                         max_series_id_interactions_with_impressions,
                         mean_recommendation_id_interactions_with_impressions,
                         min_recommendation_id_interactions_with_impressions,
                         max_recommendation_id_interactions_with_impressions,

                         mean_row_position_impressions_direct_link,
                         min_row_position_impressions_direct_link,
                         max_row_position_impressions_direct_link,
                         mean_recommendation_list_length_impressions_direct_link,
                         min_recommendation_list_length_impressions_direct_link,
                         max_recommendation_list_length_impressions_direct_link,
                         mean_recommended_series_impressions_direct_link,
                         min_recommended_series_impressions_direct_link,
                         max_recommended_series_impressions_direct_link,

                         mean_row_position_impressions_non_direct_link,
                         min_row_position_impressions_non_direct_link,
                         max_row_position_impressions_non_direct_link,
                         mean_recommendation_list_length_impressions_non_direct_link,
                         min_recommendation_list_length_impressions_non_direct_link,
                         max_recommendation_list_length_impressions_non_direct_link,
                         mean_recommended_series_impressions_non_direct_link,
                         min_recommended_series_impressions_non_direct_link,
                         max_recommended_series_impressions_non_direct_link,
                         )

        self.statistics["interactions"]["user_id"]["mean"] = mean_user_id_all_interactions
        self.statistics["interactions"]["user_id"]["min"] = min_user_id_all_interactions
        self.statistics["interactions"]["user_id"]["max"] = max_user_id_all_interactions

        self.statistics["interactions"]["item_id"]["mean"] = mean_item_id_all_interactions
        self.statistics["interactions"]["item_id"]["min"] = min_item_id_all_interactions
        self.statistics["interactions"]["item_id"]["max"] = max_item_id_all_interactions

        self.statistics["interactions"]["series_id"]["mean"] = mean_series_id_all_interactions
        self.statistics["interactions"]["series_id"]["min"] = min_series_id_all_interactions
        self.statistics["interactions"]["series_id"]["max"] = max_series_id_all_interactions

        self.statistics["interactions"]["recommendation_id"]["mean"] = mean_recommendation_id_all_interactions
        self.statistics["interactions"]["recommendation_id"]["min"] = min_recommendation_id_all_interactions
        self.statistics["interactions"]["recommendation_id"]["max"] = max_recommendation_id_all_interactions

        self.statistics["interactions_only_impressions"]["user_id"]["mean"] = mean_user_id_interactions_with_impressions
        self.statistics["interactions_only_impressions"]["user_id"]["min"] = min_user_id_interactions_with_impressions
        self.statistics["interactions_only_impressions"]["user_id"]["max"] = max_user_id_interactions_with_impressions

        self.statistics["interactions_only_impressions"]["item_id"]["mean"] = mean_item_id_interactions_with_impressions
        self.statistics["interactions_only_impressions"]["item_id"]["min"] = min_item_id_interactions_with_impressions
        self.statistics["interactions_only_impressions"]["item_id"]["max"] = max_item_id_interactions_with_impressions

        self.statistics["interactions_only_impressions"]["series_id"]["mean"] = mean_series_id_interactions_with_impressions
        self.statistics["interactions_only_impressions"]["series_id"]["min"] = min_series_id_interactions_with_impressions
        self.statistics["interactions_only_impressions"]["series_id"]["max"] = max_series_id_interactions_with_impressions

        self.statistics["interactions_only_impressions"]["recommendation_id"]["mean"] = mean_recommendation_id_interactions_with_impressions
        self.statistics["interactions_only_impressions"]["recommendation_id"]["min"] = min_recommendation_id_interactions_with_impressions
        self.statistics["interactions_only_impressions"]["recommendation_id"]["max"] = max_recommendation_id_interactions_with_impressions

        self.statistics["impressions_direct_link"]["row_position"]["mean"] = mean_row_position_impressions_direct_link
        self.statistics["impressions_direct_link"]["row_position"]["min"] = min_row_position_impressions_direct_link
        self.statistics["impressions_direct_link"]["row_position"]["max"] = max_row_position_impressions_direct_link

        self.statistics["impressions_direct_link"]["recommendation_list_length"]["mean"] = mean_recommendation_list_length_impressions_direct_link
        self.statistics["impressions_direct_link"]["recommendation_list_length"]["min"] = min_recommendation_list_length_impressions_direct_link
        self.statistics["impressions_direct_link"]["recommendation_list_length"]["max"] = max_recommendation_list_length_impressions_direct_link

        self.statistics["impressions_direct_link"]["recommended_series_list"]["mean"] = mean_recommended_series_impressions_direct_link
        self.statistics["impressions_direct_link"]["recommended_series_list"]["min"] = min_recommended_series_impressions_direct_link
        self.statistics["impressions_direct_link"]["recommended_series_list"]["max"] = max_recommended_series_impressions_direct_link

        self.statistics["impressions_non_direct_link"]["row_position"]["mean"] = mean_row_position_impressions_non_direct_link
        self.statistics["impressions_non_direct_link"]["row_position"]["min"] = min_row_position_impressions_non_direct_link
        self.statistics["impressions_non_direct_link"]["row_position"]["max"] = max_row_position_impressions_non_direct_link

        self.statistics["impressions_non_direct_link"]["recommendation_list_length"]["mean"] = mean_recommendation_list_length_impressions_non_direct_link
        self.statistics["impressions_non_direct_link"]["recommendation_list_length"]["min"] = min_recommendation_list_length_impressions_non_direct_link
        self.statistics["impressions_non_direct_link"]["recommendation_list_length"]["max"] = max_recommendation_list_length_impressions_non_direct_link

        self.statistics["impressions_non_direct_link"]["recommended_series_list"]["mean"] = mean_recommended_series_impressions_non_direct_link
        self.statistics["impressions_non_direct_link"]["recommended_series_list"]["min"] = min_recommended_series_impressions_non_direct_link
        self.statistics["impressions_non_direct_link"]["recommended_series_list"]["max"] = max_recommended_series_impressions_non_direct_link

        self._save_statistics()

    @timeit
    def basic_statistics(self) -> None:
        """Calculates basic statistics on the dataset

        Statistics
        -------------
        First recorded datetime: str
        Last recorded datetime: str
        Number of interactions: int
        Number of columns: int
        Number of users: int
        Number of items: int
        Number of series: int
        Number of recommendations: int
        Number of explicit ratings: int
        Number of recorded vision factors: int
        Naive density: float
            Density of the dataset calculated as <number of interactions> / (<number of items> * <number of users>).
            WARNING: this density will be calculated among all interactions, not unique pairs of (user,
            item) interactions.
        Naive sparsity: float
            1 - <naive density>
        Number of interactions by interaction type: List(Tuple(int, int, float))
            List of tuples that tells the interaction type, the number of interactions in this type, and percentage of
            interactions in this type, respectively.
        Number of interactions by Item type: List(Tuple(int, int, float))
            List of tuples that tells the item type, the number of interactions in this type, and percentage of
            interactions in this type, respectively.
        Number of items by Item type: List(Tuple(int, int, float))
            List of tuples that tells the item type, the number of items in this type, and percentage of items in
            this type, respectively.
        """
        self._read_statistics()

        (number_of_interactions, number_of_columns) = self.interactions.shape

        first_day_of_logs = (self.interactions.index.min())
        last_day_of_logs = (self.interactions.index.max())

        unique_users = (self.interactions.user_id.unique())
        unique_items = (self.interactions.item_id.unique())
        unique_recommendations = (self.interactions.recommendation_id.unique())
        unique_series = (self.interactions.series_id.unique())
        unique_item_types = (self.interactions.item_type.unique())
        unique_interaction_types = (self.interactions.interaction_type.unique())
        unique_items_grouped_by_item_type = (self.interactions.groupby("item_type").item_id.unique())

        naive_density = (number_of_interactions
                         / (unique_users.shape[0] * unique_items.shape[0]))

        interaction_size_grouped_by_item_type = (self.interactions
                                                 .groupby("item_type")
                                                 .size())
        interaction_size_grouped_by_interaction_type = (self.interactions
                                                        .groupby("interaction_type")
                                                        .size())

        non_nan_explicit_ratings = (self.interactions.explicit_rating[self.interactions.explicit_rating != -1.0])
        non_nan_vision_factor = (self.interactions.vision_factor[self.interactions.vision_factor != -1.0])

        number_impressions_direct_link = (self.impressions_direct_link.shape[0])
        number_impressions_non_direct_link = (self.impressions_non_direct_link.shape[0])

        (number_of_interactions,
         number_of_columns,
         unique_users,
         unique_items,
         unique_recommendations,
         unique_series,

         first_day_of_logs,
         last_day_of_logs,

         naive_density,

         unique_item_types,
         unique_items_grouped_by_item_type,

         interaction_size_grouped_by_item_type,
         interaction_size_grouped_by_interaction_type,

         non_nan_explicit_ratings,
         non_nan_vision_factor,

         number_impressions_direct_link,
         number_impressions_non_direct_link) = ddf.compute(number_of_interactions,
                                                           number_of_columns,
                                                           unique_users,
                                                           unique_items,
                                                           unique_recommendations,
                                                           unique_series,

                                                           first_day_of_logs,
                                                           last_day_of_logs,

                                                           naive_density,

                                                           unique_item_types,
                                                           unique_items_grouped_by_item_type,

                                                           interaction_size_grouped_by_item_type,
                                                           interaction_size_grouped_by_interaction_type,

                                                           non_nan_explicit_ratings,
                                                           non_nan_vision_factor,

                                                           number_impressions_direct_link,
                                                           number_impressions_non_direct_link)

        number_of_users = unique_users.shape[0]
        number_of_items = unique_items.shape[0]
        number_of_series = unique_series.shape[0]

        self.statistics["general"]["first_recorded_datetime"] = str(pd.Timestamp(first_day_of_logs, unit="ms"))
        self.statistics["general"]["last_recorded_datetime"] = str(pd.Timestamp(last_day_of_logs, unit="ms"))

        self.statistics["general"]["number_of_interactions"] = number_of_interactions
        self.statistics["general"]["number_of_columns"] = number_of_columns
        self.statistics["general"]["number_of_users"] = number_of_users
        self.statistics["general"]["number_of_items"] = number_of_items
        self.statistics["general"]["number_of_series"] = number_of_series
        self.statistics["general"]["number_of_recommendations"] = unique_recommendations.shape[0]
        self.statistics["general"]["number_of_explicit_ratings"] = non_nan_explicit_ratings.shape[0]
        self.statistics["general"]["number_of_vision_factors"] = non_nan_vision_factor.shape[0]

        self.statistics["general"]["density"]["naive"] = naive_density
        self.statistics["general"]["sparsity"]["naive"] = 1 - naive_density

        self.statistics["general"]["number_of_impressions_direct_link"] = number_impressions_direct_link
        self.statistics["general"]["number_of_impressions_non_direct_link"] = number_impressions_non_direct_link

        self.statistics["general"]["number_of_items_by_item_type"] = [(item_type,
                                                                       int(unique_items_grouped_by_item_type[
                                                                               item_type].shape[0]),
                                                                       float(unique_items_grouped_by_item_type[
                                                                                 item_type].shape[0]
                                                                             * 100
                                                                             / number_of_items))
                                                                      for item_type in unique_item_types]

        self.statistics["general"]["number_of_interactions_by_item_type"] = [(item_type,
                                                                              int(interaction_size_grouped_by_item_type[
                                                                                      item_type]),
                                                                              float(
                                                                                  interaction_size_grouped_by_item_type[
                                                                                      item_type]
                                                                                  * 100
                                                                                  / number_of_interactions))
                                                                             for item_type in unique_item_types]

        self.statistics["general"]["number_of_interactions_by_interaction_type"] = [(interaction_type,
                                                                                     int(
                                                                                         interaction_size_grouped_by_interaction_type[
                                                                                             interaction_type]),
                                                                                     float(
                                                                                         interaction_size_grouped_by_interaction_type[
                                                                                             interaction_type]
                                                                                         * 100
                                                                                         / number_of_interactions))
                                                                                    for interaction_type in
                                                                                    unique_interaction_types]

        self._save_statistics()

    @timeit
    def complex_statistics(self) -> None:
        """Calculates basic statistics on the dataset

        Statistics
        -------------
        Gini score on Users: float
        Gini score on Items: float
        Gini score on Series: float
        Density of users-items: float
            Density of the dataset calculated as <number of unique interactions by (user,item) pairs> / (<number of items> * <number of users>).
        Sparsity of users-items: float
            1 - <density of users-items>
        Density of users-series: float
            Density of the dataset calculated as <number of unique interactions by (user,series) pairs> / (<number of series> * <number of users>).
        Sparsity of users-series: float
            1 - <density of users-series>
        """
        self._read_statistics()

        num_interactions: int = self.interactions.shape[0]
        num_users = self.statistics["general"]["number_of_users"]
        num_items = self.statistics["general"]["number_of_items"]
        num_series = self.statistics["general"]["number_of_series"]

        number_of_user_item_interactions = self.interactions.drop_duplicates(subset=["user_id", "item_id"]).shape[0]
        number_of_user_series_interactions = self.interactions.drop_duplicates(subset=["user_id", "series_id"]).shape[0]

        user_item_density = (number_of_user_item_interactions
                             / (num_users * num_items))

        user_series_density = (number_of_user_series_interactions
                               / (num_users * num_series))

        gini_index_users = dask.delayed(gini)(dask.delayed(np.array)(self.interactions.user_id, dtype=np.float32))
        gini_index_items = dask.delayed(gini)(dask.delayed(np.array)(self.interactions.item_id, dtype=np.float32))
        gini_index_series = dask.delayed(gini)(dask.delayed(np.array)(self.interactions.series_id, dtype=np.float32))

        (num_interactions,

         user_item_density,
         user_series_density,

         gini_index_users,
         gini_index_items,
         gini_index_series) = dask.compute(num_interactions,

                                           user_item_density,
                                           user_series_density,

                                           gini_index_users,
                                           gini_index_items,
                                           gini_index_series)

        logger.info(f"Number of interactions: {num_interactions}")

        self.statistics["general"]["gini"]["users"] = gini_index_users
        self.statistics["general"]["gini"]["items"] = gini_index_items
        self.statistics["general"]["gini"]["series"] = gini_index_series

        self.statistics["general"]["density"]["user_item"] = user_item_density
        self.statistics["general"]["sparsity"]["user_item"] = 1.0 - user_item_density

        self.statistics["general"]["density"]["user_series"] = user_series_density
        self.statistics["general"]["sparsity"]["user_series"] = 1.0 - user_series_density

        self._save_statistics()

    def _distribution(self, category: str, data: ddf.DataFrame, data_column: str, plot_title: str, plot_xlabel: str, metadata_property: str) -> None:
        file_format = "pdf"

        results_dir = os.path.join(self._local_statistics_dir,
                                   category,
                                   data_column,)

        os.makedirs(results_dir, exist_ok=True)

        filepath = os.path.join(results_dir,
                                "distribution")

        interactions_grouped_by_property = data.groupby(data_column).size()

        property_with_most_interactions = interactions_grouped_by_property.nlargest(20)
        property_with_least_interactions = interactions_grouped_by_property.nsmallest(20)

        (interactions_grouped_by_property,
         property_with_most_interactions,
         property_with_least_interactions,) = ddf.compute(interactions_grouped_by_property,
                                                          property_with_most_interactions,
                                                          property_with_least_interactions, )

        ContentWiseImpressions._plot_property_distribution(data=interactions_grouped_by_property,
                                                           title=plot_title,
                                                           xlabel=plot_xlabel,
                                                           file_name=filepath,
                                                           file_format=file_format)

        data_sorted_desc = interactions_grouped_by_property.sort_values(ascending=False, ignore_index=True)
        total_records = data_sorted_desc.sum()

        data_cumsum = data_sorted_desc.cumsum()
        perc20 = data_cumsum[data_cumsum < total_records / 5]
        perc25 = data_cumsum[data_cumsum < total_records / 4]
        perc40 = data_cumsum[data_cumsum < 2 * total_records / 5]
        perc50 = data_cumsum[data_cumsum < total_records / 2]
        perc60 = data_cumsum[data_cumsum < 3 * total_records / 5]
        perc75 = data_cumsum[data_cumsum < 3 * total_records / 4]
        perc80 = data_cumsum[data_cumsum < 4 * total_records / 5]

        idx_perc20 = perc20.shape[0]
        idx_perc25 = perc25.shape[0]
        idx_perc40 = perc40.shape[0]
        idx_perc50 = perc50.shape[0]
        idx_perc60 = perc60.shape[0]
        idx_perc75 = perc75.shape[0]
        idx_perc80 = perc80.shape[0]

        num_metadata = self.metadata[metadata_property]
        num_data = data_sorted_desc.shape[0]

        self.statistics[category][data_column]["distribution"] = dict()

        self.statistics[category][data_column]["distribution"]["idx_perc20"] = (idx_perc20,
                                                                                num_metadata,
                                                                                idx_perc20 * 100 / num_metadata,
                                                                                num_data,
                                                                                idx_perc20 * 100 / num_data,)
        self.statistics[category][data_column]["distribution"]["idx_perc25"] = (idx_perc25,
                                                                                num_metadata,
                                                                                idx_perc25 * 100 / num_metadata,
                                                                                num_data,
                                                                                idx_perc25 * 100 / num_data,)
        self.statistics[category][data_column]["distribution"]["idx_perc40"] = (idx_perc40,
                                                                                num_metadata,
                                                                                idx_perc40 * 100 / num_metadata,
                                                                                num_data,
                                                                                idx_perc40 * 100 / num_data,)
        self.statistics[category][data_column]["distribution"]["idx_perc50"] = (idx_perc50,
                                                                                num_metadata,
                                                                                idx_perc50 * 100 / num_metadata,
                                                                                num_data,
                                                                                idx_perc50 * 100 / num_data,)
        self.statistics[category][data_column]["distribution"]["idx_perc60"] = (idx_perc60,
                                                                                num_metadata,
                                                                                idx_perc60 * 100 / num_metadata,
                                                                                num_data,
                                                                                idx_perc60 * 100 / num_data,)
        self.statistics[category][data_column]["distribution"]["idx_perc75"] = (idx_perc75,
                                                                                num_metadata,
                                                                                idx_perc75 * 100 / num_metadata,
                                                                                num_data,
                                                                                idx_perc75 * 100 / num_data,)
        self.statistics[category][data_column]["distribution"]["idx_perc80"] = (idx_perc80,
                                                                                num_metadata,
                                                                                idx_perc80 * 100 / num_metadata,
                                                                                num_data,
                                                                                idx_perc80 * 100 / num_data,)

        self.statistics[category][data_column]["most_interactions"] = property_with_most_interactions.to_dict()
        self.statistics[category][data_column]["least_interactions"] = property_with_least_interactions.to_dict()

    def _frequency(self, category: str, data: pd.DataFrame, data_column: str, plot_title: str, plot_xlabel: str, plot_xticks, bins: int) -> None:
        file_format = "pdf"

        results_dir = os.path.join(self._local_statistics_dir,
                                   category,
                                   data_column, )

        os.makedirs(results_dir, exist_ok=True)

        filepath = os.path.join(results_dir,
                                "frequency")

        for y_scale in ["linear", "log"]:
            for x_scale in ["linear"]:
                fig, ax = plt.subplots(figsize=(10.24, 7.20))
                ax.hist(data[data_column], label=None, bins=bins, histtype="bar")
                ax.legend(["Interactions"])
                ax.set_title(plot_title)
                ax.set_ylabel("Frequencies")
                ax.set_xlabel(plot_xlabel)
                ax.set_yscale(y_scale)
                ax.set_xscale(x_scale)
                ax.set_xticks(plot_xticks)
                ax.grid()
                fig.savefig(f"{filepath}_{x_scale}_{y_scale}.{file_format}", format=file_format)

    @timeit
    def user_distribution(self) -> None:
        """Calculates statistics and generate plots regarding user distributions

        Statistics
        -------------
        The 20 users with most interactions
        The 20 users with least interactions

        Plots
        -------------
        Desc sorted number of interactions by users. All plots include three vertical lines that indicates where
        the 20, 50 and 80% of interactions has been reached. Plots are automatically saved to disk. Four variants are
        plotted:
            Scale X: Normal - Scale y: Normal.
            Scale X: Normal - Scale y: Log.
            Scale X: Log - Scale y: Normal.
            Scale X: Log - Scale y: Log.
        """
        self._read_statistics()

        self._distribution(category="interactions",
                           data=self.interactions,
                           data_column="user_id",
                           plot_title="Distribution of user interactions",
                           plot_xlabel="Users",
                           metadata_property="num_users")

        self._save_statistics()

    @timeit
    def item_distribution(self) -> None:
        """Calculates statistics and generate plots regarding item distributions

        Statistics
        -------------
        The 20 items with most interactions
        The 20 items with least interactions

        Plots
        -------------
        Desc sorted number of interactions by items. All plots include three vertical lines that indicates where
        the 20, 50 and 80% of interactions has been reached. Plots are automatically saved to disk. Four variants are
        plotted:
            Scale X: Normal - Scale y: Normal.
            Scale X: Normal - Scale y: Log.
            Scale X: Log - Scale y: Normal.
            Scale X: Log - Scale y: Log.
        """
        self._read_statistics()

        self._distribution(category="interactions",
                           data=self.interactions,
                           data_column="item_id",
                           plot_title="Distribution of interactions with items",
                           plot_xlabel="Items",
                           metadata_property="num_items")

        self._save_statistics()

    @timeit
    def series_distribution(self) -> None:
        """Calculates statistics and generate plots regarding series distributions

        Statistics
        -------------
        The 20 series with most interactions
        The 20 series with least interactions

        Plots
        -------------
        Desc sorted number of interactions by series. All plots include three vertical lines that indicates where
        the 20, 50 and 80% of interactions has been reached. Plots are automatically saved to disk. Four variants are
        plotted:
            Scale X: Normal - Scale y: Normal.
            Scale X: Normal - Scale y: Log.
            Scale X: Log - Scale y: Normal.
            Scale X: Log - Scale y: Log.
        """
        self._read_statistics()

        self._distribution(category="interactions",
                           data=self.interactions,
                           data_column="series_id",
                           plot_title="Distribution of interactions with series",
                           plot_xlabel="Series",
                           metadata_property="num_series")

        self._save_statistics()

    @timeit
    def recommendations_distribution(self) -> None:
        """Calculates statistics and generate plots regarding recommendations distributions

        Statistics
        -------------
        The 20 recommendations with most interactions
        The 20 recommendations with least interactions

        Plots
        -------------
        Desc sorted number of interactions by recommendations. Only interactions with recommendations are
        considered. All plots include three vertical lines that indicate where the 20, 50 and 80% of interactions has
        been reached. Plots are automatically saved to disk. Four variants are
        plotted:
            Scale X: Normal - Scale y: Normal.
            Scale X: Normal - Scale y: Log.
            Scale X: Log - Scale y: Normal.
            Scale X: Log - Scale y: Log.
        """
        self._read_statistics()

        self._distribution(category="interactions",
                           data=self.interactions,
                           data_column="recommendation_id",
                           plot_title="Distribution of interactions with the same recommendation",
                           plot_xlabel="Recommendations",
                           metadata_property="num_recommendations")

        self._distribution(category="interactions_only_impressions",
                           data=self.interactions[self.interactions.recommendation_id >= 0],
                           data_column="recommendation_id",
                           plot_title="Distribution of interactions excluding outside impressions with the same recommendation",
                           plot_xlabel="Recommendations",
                           metadata_property="num_recommendations")

        self._save_statistics()

    @timeit
    def vision_factor_distribution(self) -> None:
        self._read_statistics()

        interaction_type_view = 0
        self._distribution(category="interactions",
                           data=self.interactions[self.interactions.interaction_type == interaction_type_view],
                           data_column="vision_factor",
                           plot_title="Distribution of vision factors",
                           plot_xlabel="Recommendations",
                           metadata_property="num_recommendations")

        self._frequency(category="interactions",
                        data=self.interactions[self.interactions.interaction_type == interaction_type_view].compute(),
                        data_column="vision_factor",
                        plot_title="Frequencies of vision factors",
                        plot_xlabel="Vision Factors",
                        plot_xticks=np.linspace(start=0.0, stop=1.0, endpoint=True, num=11),
                        bins=10)

    @timeit
    def explicit_ratings_distribution(self) -> None:
        self._read_statistics()

        interaction_type_view = 2
        self._distribution(category="interactions",
                           data=self.interactions[self.interactions.interaction_type == interaction_type_view],
                           data_column="explicit_rating",
                           plot_title="Distribution of explicit ratings",
                           plot_xlabel="Recommendations",
                           metadata_property="num_recommendations")

        self._frequency(category="interactions",
                        data=self.interactions[self.interactions.interaction_type == interaction_type_view].compute(),
                        data_column="explicit_rating",
                        plot_title="Frequencies of explicit ratings",
                        plot_xlabel="Explicit Ratings",
                        plot_xticks=np.linspace(start=1.0, stop=5.0, endpoint=True, num=9),
                        bins=9)

    @timeit
    def timestamp_distribution(self) -> None:
        """Calculates statistics and generate plots regarding recommendations distributions

        Statistics
        -------------


        Plots
        -------------

        """

        def plot():
            timestamps_filepath = os.path.join(self._local_statistics_dir,
                                               f"contentwise_impressions_distribution_of_interactions")
            ContentWiseImpressions._plot_timestamp_distribution(utc_datetimes,
                                                                title="Frequencies of interactions",
                                                                xlabel="Days",
                                                                file_name=timestamps_filepath,
                                                                file_format=file_format)

            days_filepath = os.path.join(self._local_statistics_dir,
                                         f"contentwise_impressions_distribution_of_interactions_per_day")
            ContentWiseImpressions._plot_timestamp_distribution(utc_datetimes.day,
                                                                title="Frequencies of interactions by day",
                                                                xlabel="Days",
                                                                bins=np.arange(33),
                                                                xticks=range(1, 32, 1),
                                                                file_name=days_filepath,
                                                                file_format=file_format)

            hours_filepath = os.path.join(self._local_statistics_dir,
                                          f"contentwise_impressions_distribution_of_interactions_per_hour")
            ContentWiseImpressions._plot_timestamp_distribution(utc_datetimes.hour,
                                                                title="Frequencies of interactions per hour",
                                                                xlabel="Hours",
                                                                bins=np.arange(25),
                                                                xticks=range(0, 24, 1),
                                                                file_name=hours_filepath,
                                                                file_format=file_format)

            day_of_week_filepath = os.path.join(self._local_statistics_dir,
                                                f"contentwise_impressions_distribution_of_interactions_per_day_of_week")
            ContentWiseImpressions._plot_timestamp_distribution(utc_datetimes.dayofweek,
                                                                title="Frequencies of interactions per Days in Week",
                                                                xlabel="Days in Week",
                                                                bins=np.arange(8),
                                                                xticks=range(0, 7, 1),
                                                                xtickslabels=["Monday",
                                                                              "Tuesday",
                                                                              "Wednesday",
                                                                              "Thursday",
                                                                              "Friday",
                                                                              "Saturday",
                                                                              "Sunday"],
                                                                file_name=day_of_week_filepath,
                                                                file_format=file_format)

            in_each_month_filepath = os.path.join(self._local_statistics_dir,
                                                  f"contentwise_impressions_distribution_of_interactions_in_each_month")
            ContentWiseImpressions._plot_timestamp_distribution(utc_datetimes.month,
                                                                title="Frequencies of interactions in each month",
                                                                xlabel="Months",
                                                                xticks=range(1, 5, 1),
                                                                xtickslabels=["January", "February", "March", "April"],
                                                                file_name=in_each_month_filepath,
                                                                file_format=file_format)

            periods = [("January", "2019-01", "2019-02"),
                       ("January-February", "2019-01", "2019-03"),
                       ("January-March", "2019-01", "2019-04"),
                       ("January-April", "2019-01", "2019-05"),
                       ("January-May", "2019-01", "2019-06"),

                       ("February", "2019-02", "2019-03"),
                       ("February-March", "2019-02", "2019-04"),
                       ("February-April", "2019-02", "2019-05"),
                       ("February-May", "2019-02", "2019-06"),

                       ("March", "2019-03", "2019-04"),
                       ("March-April", "2019-03", "2019-05"),
                       ("March-May", "2019-03", "2019-06"),

                       ("April", "2019-04", "2019-05"),
                       ("April-May", "2019-04", "2019-06"),

                       ("May", "2019-05", "2019-06")
                       ]
            for period_name, lower_bound, upper_bound in periods:
                timestamps = utc_datetimes[utc_datetimes >= lower_bound]
                timestamps = timestamps[timestamps < upper_bound]
                num_days_in_period = (pd.Timestamp(upper_bound) - pd.Timestamp(lower_bound)).days

                filepath = os.path.join(self._local_statistics_dir,
                                        f"contentwise_impressions_distribution_of_interactions_{period_name}")
                ContentWiseImpressions._plot_timestamp_distribution(timestamps,
                                                                    title=f"Frequencies of interactions in {period_name}",
                                                                    xlabel="Days",
                                                                    bins=num_days_in_period,
                                                                    file_name=filepath,
                                                                    file_format=file_format)

        file_format = "pdf"

        utc_datetimes: pd.DatetimeIndex = (ddf.to_datetime(self.interactions.index,
                                                           errors="raise",
                                                           unit="ms",
                                                           origin="unix")
                                           .compute())

        logger.info(f"Computed datetimes for timestamp plots. {type(utc_datetimes)}")
        plot()

    @timeit
    def recommendation_row_position_distribution(self) -> None:
        self._read_statistics()

        impressions_direct_link_row_position = self.impressions_direct_link.row_position
        unique_impressions_direct_link_row_position = self.impressions_direct_link.row_position.unique()

        impressions_non_direct_link_row_position = self.impressions_non_direct_link.row_position
        unique_impressions_non_direct_link_row_position = self.impressions_non_direct_link.row_position.unique()

        impressions_direct_link_grouped_by_row_position = self.impressions_direct_link.groupby("row_position").size()
        impressions_non_direct_link_grouped_by_row_position = self.impressions_non_direct_link.groupby(
            "row_position").size()

        most_interacted_row_positions = impressions_direct_link_grouped_by_row_position.nlargest(5)
        least_interacted_row_positions = impressions_direct_link_grouped_by_row_position.nsmallest(5)

        most_non_interacted_row_positions = impressions_non_direct_link_grouped_by_row_position.nlargest(5)
        least_non_interacted_row_positions = impressions_non_direct_link_grouped_by_row_position.nsmallest(5)

        threshold_maximum_number_of_impressions_direct_link = 1000000000
        interacted_row_positions_inside_threshold = (
            impressions_direct_link_grouped_by_row_position[
                impressions_direct_link_grouped_by_row_position <= threshold_maximum_number_of_impressions_direct_link]
        )

        threshold_maximum_number_of_impressions_non_direct_link = 1000000000
        non_interacted_row_positions_inside_threshold = (
            impressions_non_direct_link_grouped_by_row_position[
                impressions_non_direct_link_grouped_by_row_position <= threshold_maximum_number_of_impressions_non_direct_link]
        )

        (impressions_direct_link_row_position,
         unique_impressions_direct_link_row_position,

         impressions_non_direct_link_row_position,
         unique_impressions_non_direct_link_row_position,

         interacted_row_positions_inside_threshold,
         most_interacted_row_positions,
         least_interacted_row_positions,

         non_interacted_row_positions_inside_threshold,
         most_non_interacted_row_positions,
         least_non_interacted_row_positions) = ddf.compute(impressions_direct_link_row_position,
                                                           unique_impressions_direct_link_row_position,

                                                           impressions_non_direct_link_row_position,
                                                           unique_impressions_non_direct_link_row_position,

                                                           interacted_row_positions_inside_threshold,
                                                           most_interacted_row_positions,
                                                           least_interacted_row_positions,

                                                           non_interacted_row_positions_inside_threshold,
                                                           most_non_interacted_row_positions,
                                                           least_non_interacted_row_positions)

        file_format = "pdf"

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_hist_of_row_position_on_interacted_recommendations")
        ContentWiseImpressions._plot_timestamp_distribution(data=impressions_direct_link_row_position,
                                                            title="Histogram of row positions on interacted "
                                                                  "recommendations",
                                                            xlabel="Row position",
                                                            bins=len(unique_impressions_direct_link_row_position),
                                                            xticks=unique_impressions_direct_link_row_position,
                                                            file_name=filepath,
                                                            file_format=file_format)

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_hist_of_row_position_on_non_interacted_recommendations")
        ContentWiseImpressions._plot_timestamp_distribution(data=impressions_non_direct_link_row_position,
                                                            title="Histogram of row positions on non-interacted "
                                                                  "recommendations",
                                                            xlabel="Row position",
                                                            bins=len(unique_impressions_non_direct_link_row_position),
                                                            xticks=unique_impressions_non_direct_link_row_position,
                                                            file_name=filepath,
                                                            file_format=file_format)

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_distribution_of_row_position_on_interacted_recommendations")
        ContentWiseImpressions._plot_property_distribution(data=interacted_row_positions_inside_threshold,
                                                           title="Distribution of row positions on interacted recommendations",
                                                           xlabel="Row position",
                                                           file_name=filepath,
                                                           file_format=file_format,
                                                           marker="o")

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_distribution_of_row_position_on_non_interacted_recommendations")
        ContentWiseImpressions._plot_property_distribution(data=non_interacted_row_positions_inside_threshold,
                                                           title="Distribution of row positions on non-interacted recommendations",
                                                           xlabel="Row position",
                                                           file_name=filepath,
                                                           file_format=file_format,
                                                           marker="o")

        self.statistics["impressions_direct_link"]["row_position"]["most_impressed"] = most_interacted_row_positions.to_dict()
        self.statistics["impressions_direct_link"]["row_position"]["least_impressed"] = least_interacted_row_positions.to_dict()

        self.statistics["impressions_non_direct_link"]["row_position"]["most_impressed"] = most_non_interacted_row_positions.to_dict()
        self.statistics["impressions_non_direct_link"]["row_position"]["least_impressed"] = least_non_interacted_row_positions.to_dict()

        self._save_statistics()

    @timeit
    def recommendation_recommendation_list_length_distribution(self) -> None:
        self._read_statistics()

        impressions_direct_link_recommendation_list_length = self.impressions_direct_link.recommendation_list_length
        unique_impressions_direct_link_recommendation_list_length = self.impressions_direct_link.recommendation_list_length.unique()

        impressions_non_direct_link_recommendation_list_length = self.impressions_non_direct_link.recommendation_list_length
        unique_impressions_non_direct_link_recommendation_list_length = self.impressions_non_direct_link.recommendation_list_length.unique()

        impressions_direct_link_grouped_by_recommendation_list_length = self.impressions_direct_link.groupby("recommendation_list_length").size()
        impressions_non_direct_link_grouped_by_recommendation_list_length = self.impressions_non_direct_link.groupby(
            "recommendation_list_length").size()

        most_interacted_recommendation_list_lengths = impressions_direct_link_grouped_by_recommendation_list_length.nlargest(5)
        least_interacted_recommendation_list_lengths = impressions_direct_link_grouped_by_recommendation_list_length.nsmallest(5)

        most_non_interacted_recommendation_list_lengths = impressions_non_direct_link_grouped_by_recommendation_list_length.nlargest(5)
        least_non_interacted_recommendation_list_lengths = impressions_non_direct_link_grouped_by_recommendation_list_length.nsmallest(5)

        threshold_maximum_number_of_impressions_direct_link = 1000000000
        interacted_recommendation_list_lengths_inside_threshold = (
            impressions_direct_link_grouped_by_recommendation_list_length[
                impressions_direct_link_grouped_by_recommendation_list_length <= threshold_maximum_number_of_impressions_direct_link]
        )

        threshold_maximum_number_of_impressions_non_direct_link = 1000000000
        non_interacted_recommendation_list_lengths_inside_threshold = (
            impressions_non_direct_link_grouped_by_recommendation_list_length[
                impressions_non_direct_link_grouped_by_recommendation_list_length <= threshold_maximum_number_of_impressions_non_direct_link]
        )

        (impressions_direct_link_recommendation_list_length,
         unique_impressions_direct_link_recommendation_list_length,

         impressions_non_direct_link_recommendation_list_length,
         unique_impressions_non_direct_link_recommendation_list_length,

         interacted_recommendation_list_lengths_inside_threshold,
         most_interacted_recommendation_list_lengths,
         least_interacted_recommendation_list_lengths,

         non_interacted_recommendation_list_lengths_inside_threshold,
         most_non_interacted_recommendation_list_lengths,
         least_non_interacted_recommendation_list_lengths) = ddf.compute(
            impressions_direct_link_recommendation_list_length,
            unique_impressions_direct_link_recommendation_list_length,

            impressions_non_direct_link_recommendation_list_length,
            unique_impressions_non_direct_link_recommendation_list_length,

            interacted_recommendation_list_lengths_inside_threshold,
            most_interacted_recommendation_list_lengths,
            least_interacted_recommendation_list_lengths,

            non_interacted_recommendation_list_lengths_inside_threshold,
            most_non_interacted_recommendation_list_lengths,
            least_non_interacted_recommendation_list_lengths)

        file_format = "pdf"

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_hist_of_recommendation_list_length_on_interacted_recommendations")
        ContentWiseImpressions._plot_timestamp_distribution(data=impressions_direct_link_recommendation_list_length,
                                                            title="Histogram of row positions on interacted "
                                                                  "recommendations",
                                                            xlabel="Row position",
                                                            bins=len(unique_impressions_direct_link_recommendation_list_length),
                                                            xticks=unique_impressions_direct_link_recommendation_list_length,
                                                            file_name=filepath,
                                                            file_format=file_format)

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_hist_of_recommendation_list_length_on_non_interacted_recommendations")
        ContentWiseImpressions._plot_timestamp_distribution(data=impressions_non_direct_link_recommendation_list_length,
                                                            title="Histogram of row positions on non-interacted "
                                                                  "recommendations",
                                                            xlabel="Row position",
                                                            bins=len(unique_impressions_non_direct_link_recommendation_list_length),
                                                            xticks=unique_impressions_non_direct_link_recommendation_list_length,
                                                            file_name=filepath,
                                                            file_format=file_format)

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_distribution_of_recommendation_list_length_on_interacted_recommendations")
        ContentWiseImpressions._plot_property_distribution(data=interacted_recommendation_list_lengths_inside_threshold,
                                                           title="Distribution of row positions on interacted recommendations",
                                                           xlabel="Row position",
                                                           file_name=filepath,
                                                           file_format=file_format,
                                                           marker="o")

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_distribution_of_recommendation_list_length_on_non_interacted_recommendations")
        ContentWiseImpressions._plot_property_distribution(data=non_interacted_recommendation_list_lengths_inside_threshold,
                                                           title="Distribution of row positions on non-interacted recommendations",
                                                           xlabel="Row position",
                                                           file_name=filepath,
                                                           file_format=file_format,
                                                           marker="o")

        self.statistics["impressions_direct_link"]["recommendation_list_length"]["most_impressed"] = most_interacted_recommendation_list_lengths.to_dict()
        self.statistics["impressions_direct_link"]["recommendation_list_length"]["least_impressed"] = least_interacted_recommendation_list_lengths.to_dict()

        self.statistics["impressions_non_direct_link"]["recommendation_list_length"]["most_impressed"] = most_non_interacted_recommendation_list_lengths.to_dict()
        self.statistics["impressions_non_direct_link"]["recommendation_list_length"]["least_impressed"] = least_non_interacted_recommendation_list_lengths.to_dict()

        self._save_statistics()

    @timeit
    def recommendation_series_distribution(self) -> None:
        self._read_statistics()

        impressions_direct_link_series: ddf.DataFrame = self.impressions_direct_link.explode(
            column="recommended_series_list")
        unique_impressions_direct_link_series = self.impressions_direct_link.recommended_series_list.unique()

        impressions_non_direct_link_series: ddf.DataFrame = self.impressions_non_direct_link.explode(
            column="recommended_series_list")
        unique_impressions_non_direct_link_series = self.impressions_non_direct_link.recommended_series_list.unique()

        impressions_direct_link_grouped_by_series = impressions_direct_link_series.groupby(
            "recommended_series_list").size()
        impressions_non_direct_link_grouped_by_series = impressions_non_direct_link_series.groupby(
            "recommended_series_list").size()

        most_interacted_series = impressions_direct_link_grouped_by_series.nlargest(5)
        least_interacted_series = impressions_direct_link_grouped_by_series.nsmallest(5)

        most_non_interacted_series = impressions_non_direct_link_grouped_by_series.nlargest(5)
        least_non_interacted_series = impressions_non_direct_link_grouped_by_series.nsmallest(5)

        threshold_maximum_number_of_impressions_direct_link = 1000000000
        interacted_series_inside_threshold = (
            impressions_direct_link_grouped_by_series[
                impressions_direct_link_grouped_by_series <= threshold_maximum_number_of_impressions_direct_link]
        )

        threshold_maximum_number_of_impressions_non_direct_link = 1000000000
        non_interacted_series_inside_threshold = (
            impressions_non_direct_link_grouped_by_series[
                impressions_non_direct_link_grouped_by_series <= threshold_maximum_number_of_impressions_non_direct_link]
        )

        (impressions_direct_link_series,
         unique_impressions_direct_link_series,

         impressions_non_direct_link_series,
         unique_impressions_non_direct_link_series,

         interacted_series_inside_threshold,
         most_interacted_series,
         least_interacted_series,

         non_interacted_series_inside_threshold,
         most_non_interacted_series,
         least_non_interacted_series) = ddf.compute(impressions_direct_link_series,
                                                    unique_impressions_direct_link_series,

                                                    impressions_non_direct_link_series,
                                                    unique_impressions_non_direct_link_series,

                                                    interacted_series_inside_threshold,
                                                    most_interacted_series,
                                                    least_interacted_series,

                                                    non_interacted_series_inside_threshold,
                                                    most_non_interacted_series,
                                                    least_non_interacted_series)

        file_format = "pdf"

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_hist_of_row_position_on_interacted_recommendations")
        ContentWiseImpressions._plot_timestamp_distribution(data=impressions_direct_link_series,
                                                            title="Histogram of row positions on interacted "
                                                                  "recommendations",
                                                            xlabel="Row position",
                                                            bins=len(unique_impressions_direct_link_series),
                                                            xticks=unique_impressions_direct_link_series,
                                                            file_name=filepath,
                                                            file_format=file_format)

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_hist_of_row_position_on_non_interacted_recommendations")
        ContentWiseImpressions._plot_timestamp_distribution(data=impressions_non_direct_link_series,
                                                            title="Histogram of row positions on non-interacted "
                                                                  "recommendations",
                                                            xlabel="Row position",
                                                            bins=len(unique_impressions_non_direct_link_series),
                                                            xticks=unique_impressions_non_direct_link_series,
                                                            file_name=filepath,
                                                            file_format=file_format)

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_distribution_of_row_position_on_interacted_recommendations")
        ContentWiseImpressions._plot_property_distribution(data=interacted_series_inside_threshold,
                                                           title="Distribution of row positions on interacted recommendations",
                                                           xlabel="Row position",
                                                           file_name=filepath,
                                                           file_format=file_format,
                                                           marker="o")

        filepath = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_distribution_of_row_position_on_non_interacted_recommendations")
        ContentWiseImpressions._plot_property_distribution(data=non_interacted_series_inside_threshold,
                                                           title="Distribution of row positions on non-interacted recommendations",
                                                           xlabel="Row position",
                                                           file_name=filepath,
                                                           file_format=file_format,
                                                           marker="o")

        self.statistics["impressions_direct_link"]["recommended_series_list"]["most_impressed"] = most_interacted_series.to_dict()
        self.statistics["impressions_direct_link"]["recommended_series_list"]["least_impressed"] = least_interacted_series.to_dict()

        self.statistics["impressions_non_direct_link"]["recommended_series_list"]["most_impressed"] = most_non_interacted_series.to_dict()
        self.statistics["impressions_non_direct_link"]["recommended_series_list"]["least_impressed"] = least_non_interacted_series.to_dict()

        self._save_statistics()

    @timeit
    def heatmap_interacted_recommendations(self) -> None:
        self._read_statistics()

        def get_interactions_with_recommendation_index() -> pd.DataFrame:
            def get_series_index_on_recommendation_list(row):
                results: np.ndarray = np.where(row.recommended_series_list == row.series_id)
                indices: np.ndarray = results[0]

                if len(indices) == 0:
                    return -1
                return indices[0]

            data = self.interactions[self.interactions.recommendation_id >= 0]

            data: ddf.DataFrame = data.merge(right=self.impressions_direct_link,
                                             how="inner",
                                             left_on="recommendation_id",
                                             right_index=True)

            data["recommendation_index"] = data.apply(get_series_index_on_recommendation_list,
                                                      axis="columns",
                                                      meta=("recommendation_index", "int32"))

            data = data[data.recommendation_index >= 0]

            (data,) = ddf.compute(data)

            return data

        interactions_with_recommendation_index = get_interactions_with_recommendation_index()
        interactions_with_recommendation_index: pd.DataFrame = (interactions_with_recommendation_index
                                                                .groupby(by=["row_position", "recommendation_index"])
                                                                .size()
                                                                .reset_index(drop=False, name="count"))

        file_format = "pdf"
        filename = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_heatmap_interacted_recommendations_row_position_by_recommendation_index")
        ContentWiseImpressions._plot_heatmap(
            data=interactions_with_recommendation_index,
            pivot_index_name="row_position",
            pivot_columns_name="recommendation_index",
            pivot_values_name="count",
            xlabel="Recommendation Position",
            ylabel="Row Position",
            file_name=filename,
            file_format=file_format)

        filename = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_heatmap_interacted_recommendations_row_position_greater5")
        ContentWiseImpressions._plot_heatmap(
            data=interactions_with_recommendation_index[interactions_with_recommendation_index.row_position > 5],
            pivot_index_name="row_position",
            pivot_columns_name="recommendation_index",
            pivot_values_name="count",
            xlabel="Recommendation Position",
            ylabel="Row Position",
            file_name=filename,
            file_format=file_format)

        filename = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_heatmap_interacted_recommendations_recommendation_index_greater20")
        ContentWiseImpressions._plot_heatmap(
            data=interactions_with_recommendation_index[interactions_with_recommendation_index.recommendation_index > 20],
            pivot_index_name="row_position",
            pivot_columns_name="recommendation_index",
            pivot_values_name="count",
            xlabel="Recommendation Position",
            ylabel="Row Position",
            file_name=filename,
            file_format=file_format)

        filename = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_heatmap_interacted_recommendations_row_position_grater5_by_recommendation_index_greater20")
        ContentWiseImpressions._plot_heatmap(
            data=interactions_with_recommendation_index[(interactions_with_recommendation_index.row_position > 5) & (interactions_with_recommendation_index.recommendation_index > 20)],
            pivot_index_name="row_position",
            pivot_columns_name="recommendation_index",
            pivot_values_name="count",
            xlabel="Recommendation Position",
            ylabel="Row Position",
            file_name=filename,
            file_format=file_format)

        filename = os.path.join(self._local_statistics_dir,
                                f"contentwise_impressions_heatmap_interacted_recommendations_row_position_less5_by_recommendation_index_less20")
        ContentWiseImpressions._plot_heatmap(
            data=interactions_with_recommendation_index[(interactions_with_recommendation_index.row_position <= 5) & (interactions_with_recommendation_index.recommendation_index <= 20)],
            pivot_index_name="row_position",
            pivot_columns_name="recommendation_index",
            pivot_values_name="count",
            xlabel="Recommendation Position",
            ylabel="Row Position",
            file_name=filename,
            file_format=file_format)

    @timeit
    def calculate_residual_interacted_impressions(self) -> None:
        def calculate_residual(impressions_type: str) -> int:
            URM_impressions = self.URM[impressions_type].copy()

            URM_impressions_non_interacted = URM_impressions - URM_impressions.multiply(URM_all_bool)
            URM_impressions_non_interacted.eliminate_zeros()

            return URM_impressions.nnz - URM_impressions_non_interacted.nnz

        self._read_statistics()

        URM_all = (self.URM["train"] + self.URM["validation"] + self.URM["test"])
        URM_all_bool = URM_all.astype(np.bool)

        residual_interacted_impressions = calculate_residual("impressions")
        residual_interacted_impressions_direct_link = calculate_residual("impressions_direct_link")
        residual_interacted_impressions_non_direct_link = calculate_residual("impressions_non_direct_link")

        logger.info(f"Residual interacted impressions: {residual_interacted_impressions}"
                    f"\nResidual interacted impressions direct link: {residual_interacted_impressions_direct_link}"
                    f"\nResidual interacted impressions non-direct: {residual_interacted_impressions_non_direct_link}")

        self.statistics["general"]["residual"]["interacted_impressions"] = residual_interacted_impressions
        self.statistics["general"]["residual"]["interacted_impressions_direct_link"] = residual_interacted_impressions_direct_link
        self.statistics["general"]["residual"]["interacted_impressions_non_direct_link"] = residual_interacted_impressions_non_direct_link

        self._save_statistics()

    @staticmethod
    def _plot_property_distribution(data,
                                    title,
                                    xlabel,
                                    file_name,
                                    file_format,
                                    marker: Optional[str] = None):

        data_sorted_desc = data.sort_values(ascending=False, ignore_index=True)
        x_data = np.arange(1, data_sorted_desc.shape[0] + 1)
        y_data = data_sorted_desc

        fig: Figure
        ax: plt.Axes
        for y_scale in ["linear", "log"]:
            for x_scale in ["linear", "log"]:
                fig, ax = plt.subplots(figsize=(10.24, 7.20))
                ax.plot(x_data, y_data, marker=marker)
                ax.set_title(f"{title} - Scale Y: {y_scale} - Scale X: {x_scale}")
                ax.set_yscale(y_scale)
                ax.set_xscale(x_scale)
                ax.set_ylabel("Interactions")
                ax.set_xlabel(xlabel)
                ax.grid()
                fig.savefig(f"{file_name}_{y_scale}_{x_scale}.{file_format}", format=file_format)

    @staticmethod
    def _plot_timestamp_distribution(data,
                                     title,
                                     xlabel,
                                     file_name,
                                     file_format,
                                     bins=None,
                                     xticks=None,
                                     xtickslabels=None,
                                     histtype="bar"):
        fig, ax = plt.subplots(figsize=(10.24, 7.20))
        ax.hist(data, label=None, bins=bins, histtype=histtype)
        ax.legend(["Interactions"])
        ax.set_title(title)
        ax.set_ylabel("Frequencies")
        ax.set_xlabel(xlabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        if xtickslabels is not None:
            ax.set_xticklabels(xtickslabels)
        ax.grid()
        fig.savefig(f"{file_name}.{file_format}", format=file_format)

    @staticmethod
    def _plot_heatmap(data: pd.DataFrame,
                      pivot_index_name: str,
                      pivot_columns_name: str,
                      pivot_values_name: str,
                      xlabel: str,
                      ylabel: str,
                      file_name: str,
                      file_format: str,):

        data = data.pivot(index=pivot_index_name,
                          columns=pivot_columns_name,
                          values=pivot_values_name)

        min_value = data.min().min()
        max_value = data.max().max()

        fig: Figure
        ax: Axes

        sns.set(font_scale=1.2)
        with sns.axes_style("white"):
            fig, ax = plt.subplots(figsize=(10.24, 7.20))
            sns.heatmap(data=data,
                        ax=ax,
                        vmin=min_value,
                        vmax=max_value,
                        )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.savefig(f"{file_name}_linear.{file_format}", format=file_format)

            fig, ax = plt.subplots(figsize=(10.24, 7.20))
            sns.heatmap(data=data,
                        ax=ax,
                        vmin=min_value,
                        vmax=max_value,
                        annot=True,
                        fmt=".2g",
                        )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.savefig(f"{file_name}_linear_annotated.{file_format}", format=file_format)

            fig, ax = plt.subplots(figsize=(10.24, 7.20))
            sns.heatmap(data=data,
                        ax=ax,
                        vmin=min_value,
                        vmax=max_value,
                        norm=LogNorm(vmin=min_value, vmax=max_value),
                        )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.savefig(f"{file_name}_log.{file_format}", format=file_format)

            fig, ax = plt.subplots(figsize=(10.24, 7.20))
            sns.heatmap(data=data,
                        ax=ax,
                        vmin=min_value,
                        vmax=max_value,
                        annot=True,
                        fmt=".2g",
                        norm=LogNorm(vmin=min_value, vmax=max_value),
                        )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.savefig(f"{file_name}_log_annotated.{file_format}", format=file_format)


def read_dataset(dataset_variant: ContentWiseImpressions.Variant, use_items: bool = True) -> ContentWiseImpressions:
    contentwise_impressions = ContentWiseImpressions(dataset_variant=dataset_variant)
    contentwise_impressions.read_dataset()
    contentwise_impressions.read_urm_splits(use_cache=True, use_items=use_items)

    if any(split_name not in contentwise_impressions.URM
           for split_name in contentwise_impressions.valid_splits):
        raise ValueError(f"Dataset has an invalid split. Valid splits: {contentwise_impressions.valid_splits}")

    return contentwise_impressions
