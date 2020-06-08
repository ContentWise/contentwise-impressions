import dask.dataframe as ddf
import numpy as np
import pandas as pd
import pytest
from distributed import Client
from scipy.sparse import csr_matrix

from Utils.dataset import ContentWiseImpressions, read_dataset
from Utils.config import configure_dask_cluster


@pytest.fixture
def dataset() -> ContentWiseImpressions:
    dataset = read_dataset(ContentWiseImpressions.Variant.CW10M, use_items=True)
    return dataset


@pytest.fixture(scope="module")
def dask_client() -> Client:
    client, cluster = configure_dask_cluster(use_processes=True)
    yield client
    client.close(), cluster.close()


def test_consistency_interactions_index(dataset: ContentWiseImpressions):
    na_index_mask: ddf.Series = dataset.interactions.index.isna()
    min_index: int = dataset.interactions.index.min()
    max_index: int = dataset.interactions.index.max()

    (na_index_mask,
     min_index,
     max_index,) = ddf.compute(na_index_mask,
                               min_index,
                               max_index,)

    assert not na_index_mask.any()


def test_consistency_interactions_user_ids(dataset: ContentWiseImpressions):
    expected_number_users: int = dataset.metadata["num_users"]
    na_user_id_mask: ddf.Series = dataset.interactions.user_id.isna()
    invalid_user_id_mask: ddf.Series = ((dataset.interactions.user_id < 0) |
                                        (dataset.interactions.user_id > expected_number_users))

    (na_user_id_mask,
     invalid_user_id_mask,) = ddf.compute(na_user_id_mask,
                                          invalid_user_id_mask,)

    assert not na_user_id_mask.any()
    assert not invalid_user_id_mask.any()


def test_consistency_interactions_item_ids(dataset: ContentWiseImpressions):
    expected_number_items: int = dataset.metadata["num_items"]
    na_item_id_mask: ddf.Series = dataset.interactions.item_id.isna()
    invalid_item_id_mask: ddf.Series = ((dataset.interactions.item_id < 0) |
                                        (dataset.interactions.item_id > expected_number_items))

    (na_item_id_mask,
     invalid_item_id_mask,) = ddf.compute(na_item_id_mask,
                                          invalid_item_id_mask,)

    assert not na_item_id_mask.any()
    assert not invalid_item_id_mask.any()


def test_consistency_interactions_item_types(dataset: ContentWiseImpressions):
    na_item_type_mask: ddf.Series = dataset.interactions.item_type.isna()
    invalid_item_types_mask: ddf.Series = (dataset.interactions
                                           .item_type
                                           .map(lambda item_type: item_type not in {0, 1, 2, 3}))

    (na_item_type_mask,
     invalid_item_types_mask,) = ddf.compute(na_item_type_mask,
                                             invalid_item_types_mask,)

    assert not na_item_type_mask.any()
    assert not invalid_item_types_mask.any()


def test_consistency_interactions_series_ids(dataset: ContentWiseImpressions):
    expected_number_series: int = dataset.metadata["num_series"]
    na_series_id_mask: ddf.Series = dataset.interactions.series_id.isna()
    invalid_series_id_mask: ddf.Series = ((dataset.interactions.series_id < 0) |
                                          (dataset.interactions.series_id > expected_number_series))

    (na_series_id_mask,
     invalid_series_id_mask,) = ddf.compute(na_series_id_mask,
                                            invalid_series_id_mask,)

    assert not na_series_id_mask.any()
    assert not invalid_series_id_mask.any()


def test_consistency_interactions_episode_numbers(dataset: ContentWiseImpressions):
    na_episode_number_mask: ddf.Series = dataset.interactions.episode_number.isna()
    invalid_episode_number_mask: ddf.Series = (dataset.interactions.episode_number < 0)

    (na_episode_number_mask,
     invalid_episode_number_mask,) = ddf.compute(na_episode_number_mask,
                                                 invalid_episode_number_mask,)

    assert not na_episode_number_mask.any()
    assert not invalid_episode_number_mask.any()


def test_consistency_interactions_series_length(dataset: ContentWiseImpressions):
    na_series_length_mask: ddf.Series = dataset.interactions.series_length.isna()
    invalid_series_length_mask: ddf.Series = (dataset.interactions.series_length < 0)

    (na_series_length_mask,
     invalid_series_length_mask,) = ddf.compute(na_series_length_mask,
                                                invalid_series_length_mask,)

    assert not na_series_length_mask.any()
    assert not invalid_series_length_mask.any()


def test_consistency_interactions_recommendation_ids(dataset: ContentWiseImpressions):
    expected_number_recommendations: int = dataset.metadata["num_recommendations"]
    na_recommendation_id_mask: ddf.Series = dataset.interactions.recommendation_id.isna()
    invalid_recommendation_id_mask: ddf.Series = ((dataset.interactions.recommendation_id < -1) |
                                                  (dataset.interactions.recommendation_id > expected_number_recommendations))

    (na_recommendation_id_mask,
     invalid_recommendation_id_mask) = ddf.compute(na_recommendation_id_mask,
                                                   invalid_recommendation_id_mask,)

    assert not na_recommendation_id_mask.any()
    assert not invalid_recommendation_id_mask.any()


def test_consistency_interactions_interaction_types(dataset: ContentWiseImpressions):
    na_interaction_type_mask: ddf.Series = dataset.interactions.interaction_type.isna()
    invalid_interaction_type_mask: ddf.Series = (dataset.interactions
                                                 .interaction_type
                                                 .map(lambda interaction_type: interaction_type not in {0, 1, 2, 3}))

    (na_interaction_type_mask,
     invalid_interaction_type_mask,) = ddf.compute(na_interaction_type_mask,
                                                   invalid_interaction_type_mask,)

    assert not na_interaction_type_mask.any()
    assert not invalid_interaction_type_mask.any()


def test_consistency_interactions_explicit_ratings(dataset: ContentWiseImpressions):
    # Explicit ratings values should only be set when the interaction type is "Rated" (2).
    # We verify that all "rated" interactions have valid values (from 0.0 to 5.0 with steps of 0.5)
    # For any other interaction, we verify that the value is -1.
    na_explicit_ratings_mask: ddf.Series = dataset.interactions.explicit_rating.isna()

    rated_interactions_explicit_ratings: ddf.Series = dataset.interactions[dataset.interactions.interaction_type ==
                                                                           2].explicit_rating
    rated_invalid_explicit_ratings_mask: ddf.Series = (rated_interactions_explicit_ratings
                                                       .map(lambda rating: rating not in np.linspace(0.0, 5.0,
                                                                                                     num=11)))

    other_interactions_explicit_rating = dataset.interactions[
        dataset.interactions.interaction_type != 2].explicit_rating
    other_invalid_explicit_ratings_mask: ddf.Series = (other_interactions_explicit_rating != -1.0)

    (na_explicit_ratings_mask,
     rated_invalid_explicit_ratings_mask,
     other_invalid_explicit_ratings_mask,) = ddf.compute(na_explicit_ratings_mask,
                                                         rated_invalid_explicit_ratings_mask,
                                                         other_invalid_explicit_ratings_mask,)
    assert not na_explicit_ratings_mask.any()
    assert not rated_invalid_explicit_ratings_mask.any()
    assert not other_invalid_explicit_ratings_mask.any()


def test_consistency_interactions_vision_factor(dataset: ContentWiseImpressions):
    # Vision factor values should only be set when the interaction type is "Viewed" (0).
    # We verify that all "viewed" interactions have valid values (from 0.0 to 1.0)
    # For any other interaction, we verify that the value is -1.
    na_vision_factor_mask: ddf.Series = dataset.interactions.vision_factor.isna()
    viewed_interactions_vision_factors = dataset.interactions[dataset.interactions.interaction_type == 0].vision_factor
    viewed_invalid_vision_factor_mask: ddf.Series = ((viewed_interactions_vision_factors < 0) |
                                                     (viewed_interactions_vision_factors > 5.0))

    other_interactions_vision_factors = dataset.interactions[dataset.interactions.interaction_type != 0].vision_factor
    other_invalid_vision_factor_mask: ddf.Series = (other_interactions_vision_factors != -1.0)

    (na_vision_factor_mask,
     viewed_invalid_vision_factor_mask,
     other_invalid_vision_factor_mask,) = ddf.compute(na_vision_factor_mask,
                                                      viewed_invalid_vision_factor_mask,
                                                      other_invalid_vision_factor_mask,)

    assert not na_vision_factor_mask.any()
    assert not viewed_invalid_vision_factor_mask.any()
    assert not other_invalid_vision_factor_mask.any()


def test_consistency_interactions_items_have_only_one_series(dataset: ContentWiseImpressions):
    pairs_item_id_with_series_id = (dataset
                                    .interactions[["item_id", "series_id"]]
                                    .groupby("item_id")
                                    .series_id
                                    .agg(["min", "max"]))

    invalid_pairs_mask = (pairs_item_id_with_series_id["min"] != pairs_item_id_with_series_id["max"])

    (invalid_pairs_mask,) = ddf.compute(invalid_pairs_mask, scheduler="threads")

    assert not invalid_pairs_mask.any()


def test_consistency_interactions_items_have_only_one_type(dataset: ContentWiseImpressions):
    pairs_item_id_with_item_type = (dataset
                                    .interactions[["item_id", "item_type"]]
                                    .groupby("item_id")
                                    .item_type
                                    .agg(["min", "max"]))

    invalid_pairs_mask = (pairs_item_id_with_item_type["min"] != pairs_item_id_with_item_type["max"])

    (invalid_pairs_mask,) = ddf.compute(invalid_pairs_mask, scheduler="threads")

    assert not invalid_pairs_mask.any()


def test_consistency_interactions_items_have_only_one_episode_number(dataset: ContentWiseImpressions):
    pairs_item_id_with_episode_number = (dataset
                                         .interactions[["item_id", "episode_number"]]
                                         .groupby("item_id")
                                         .episode_number
                                         .agg(["min", "max"]))

    invalid_pairs_mask = (pairs_item_id_with_episode_number["min"] != pairs_item_id_with_episode_number["max"])

    (invalid_pairs_mask, ) = ddf.compute(invalid_pairs_mask, scheduler="threads")

    assert not invalid_pairs_mask.any()


def test_consistency_interactions_items_have_same_series_length(dataset: ContentWiseImpressions):
    pairs_item_id_with_series_length = (dataset
                                        .interactions[["item_id", "series_length"]]
                                        .groupby("item_id")
                                        .series_length
                                        .agg(["min", "max"]))

    invalid_pairs_mask = (pairs_item_id_with_series_length["min"] != pairs_item_id_with_series_length["max"])

    (invalid_pairs_mask,) = ddf.compute(invalid_pairs_mask, scheduler="threads")

    assert not invalid_pairs_mask.any()


def test_consistency_interactions_episode_number_lower_than_series_length(dataset: ContentWiseImpressions):
    invalid_pairs_mask = (dataset.interactions.episode_number >
                          dataset.interactions.series_length)

    (invalid_pairs_mask,) = ddf.compute(invalid_pairs_mask, scheduler="threads")

    assert not invalid_pairs_mask.any()


def test_consistency_impressions_direct_link_index(dataset: ContentWiseImpressions):
    na_index_mask: ddf.Series = dataset.impressions_direct_link.index.isna()

    (na_index_mask,) = ddf.compute(na_index_mask,)

    assert not na_index_mask.any()


def test_consistency_impressions_direct_link_row_position(dataset: ContentWiseImpressions):
    na_row_position_mask: ddf.Series = dataset.impressions_direct_link.row_position.isna()
    row_position_less_than_zero_mask: ddf.Series = (dataset.impressions_direct_link.row_position < 0)

    (na_row_position_mask,
     row_position_less_than_zero_mask,) = ddf.compute(na_row_position_mask,
                                                      row_position_less_than_zero_mask,)

    assert not na_row_position_mask.any(skipna=False)
    assert not row_position_less_than_zero_mask.any(skipna=False)


def test_consistency_impressions_direct_link_recommendation_list_length(dataset: ContentWiseImpressions):
    na_recommendation_list_length_mask: ddf.Series = dataset.impressions_direct_link.recommendation_list_length.isna()
    recommendation_list_length_less_than_zero_mask: ddf.Series = (
            dataset.impressions_direct_link.recommendation_list_length < 0)

    (na_recommendation_list_length_mask,
     recommendation_list_length_less_than_zero_mask,) = ddf.compute(na_recommendation_list_length_mask,
                                                                    recommendation_list_length_less_than_zero_mask,)

    assert not na_recommendation_list_length_mask.any(skipna=False)
    assert not recommendation_list_length_less_than_zero_mask.any(skipna=False)


def test_consistency_impressions_direct_link_recommended_series(dataset: ContentWiseImpressions):
    na_recommended_series_mask: ddf.Series = (dataset
                                              .impressions_direct_link
                                              .recommended_series_list
                                              .map(lambda recommended_series_list: recommended_series_list.shape[0] > 0 and
                                                                                   np.any(np.isnan(recommended_series_list)),
                                                   meta=("na_recommended_series_mask", "bool")))

    (na_recommended_series_mask,) = ddf.compute(na_recommended_series_mask,)

    assert not na_recommended_series_mask.any(skipna=False)


def test_consistency_impressions_direct_link_recommended_lists_with_at_least_one_item(dataset: ContentWiseImpressions):
    empty_recommendation_list_mask = (dataset
                                      .impressions_direct_link
                                      .recommended_series_list
                                      .map(lambda recommended_series_list: recommended_series_list.shape[0] == 0,
                                           meta=("actual_length_of_recommended_series", "bool")))

    (empty_recommendation_list_mask,) = ddf.compute(empty_recommendation_list_mask)

    assert not empty_recommendation_list_mask.any(skipna=False)


def test_consistency_impressions_direct_link_reported_length_equal_to_actual_length(dataset: ContentWiseImpressions):
    recommendation_list_length = dataset.impressions_direct_link.recommendation_list_length

    actual_length_of_recommended_series = (dataset
                                           .impressions_direct_link
                                           .recommended_series_list
                                           .map(lambda series: series.shape[0],
                                                meta=("actual_length_of_recommended_series", "int")))

    impressions_with_mismatching_length_mask = (recommendation_list_length != actual_length_of_recommended_series)

    (impressions_with_mismatching_length_mask,) = ddf.compute(impressions_with_mismatching_length_mask)

    assert not impressions_with_mismatching_length_mask.any(skipna=False)


def test_consistency_impressions_non_direct_link_index(dataset: ContentWiseImpressions):
    na_index_mask: ddf.Series = dataset.impressions_non_direct_link.index.isna()

    (na_index_mask,) = ddf.compute(na_index_mask, )

    assert not na_index_mask.any()


def test_consistency_impressions_non_direct_link_row_position(dataset: ContentWiseImpressions):
    na_row_position_mask: ddf.Series = dataset.impressions_non_direct_link.row_position.isna()
    row_position_less_than_zero_mask: ddf.Series = (dataset.impressions_direct_link.row_position < 0)

    (na_row_position_mask,
     row_position_less_than_zero_mask,) = ddf.compute(na_row_position_mask,
                                                      row_position_less_than_zero_mask, )

    assert not na_row_position_mask.any(skipna=False)
    assert not row_position_less_than_zero_mask.any(skipna=False)


def test_consistency_impressions_non_direct_link_recommendation_list_length(dataset: ContentWiseImpressions):
    na_recommendation_list_length_mask: ddf.Series = (dataset
                                                      .impressions_non_direct_link
                                                      .recommendation_list_length
                                                      .isna())
    recommendation_list_length_less_than_zero_mask: ddf.Series = (dataset
                                                                  .impressions_non_direct_link
                                                                  .recommendation_list_length < 0)

    (na_recommendation_list_length_mask,
     recommendation_list_length_less_than_zero_mask,) = ddf.compute(na_recommendation_list_length_mask,
                                                                    recommendation_list_length_less_than_zero_mask, )

    assert not na_recommendation_list_length_mask.any(skipna=False)
    assert not recommendation_list_length_less_than_zero_mask.any(skipna=False)


def test_consistency_impressions_non_direct_link_recommended_series(dataset: ContentWiseImpressions):
    na_recommended_series_map_mask: ddf.Series = (dataset
                                                  .impressions_non_direct_link
                                                  .recommended_series_list
                                                  .map(lambda recommended_series_list: np.any(np.isnan(recommended_series_list)),
                                                       meta=("na_recommended_series_mask", "bool")))

    (na_recommended_series_map_mask,) = ddf.compute(na_recommended_series_map_mask)

    assert not na_recommended_series_map_mask.any(skipna=False)


def test_consistency_impressions_non_direct_link_recommended_lists_with_at_least_one_item(dataset: ContentWiseImpressions):
    empty_recommendation_list_mask = (dataset
                                      .impressions_non_direct_link
                                      .recommended_series_list
                                      .map(lambda recommended_series_list: recommended_series_list.shape[0] == 0,
                                           meta=("empty_recommendation_list_mask", "bool")))

    (empty_recommendation_list_mask,) = ddf.compute(empty_recommendation_list_mask,)

    assert not empty_recommendation_list_mask.any(skipna=False)


def test_consistency_impressions_non_direct_link_reported_length_equal_to_actual_length(dataset: ContentWiseImpressions):
    recommendation_list_length = dataset.impressions_non_direct_link.recommendation_list_length

    actual_length_of_recommended_series = (dataset
                                           .impressions_non_direct_link
                                           .recommended_series_list
                                           .map(lambda series: series.shape[0],
                                                meta=("actual_length_of_recommended_series", "int")))

    impressions_with_mismatching_length_mask = (recommendation_list_length != actual_length_of_recommended_series)

    (impressions_with_mismatching_length_mask,) = ddf.compute(impressions_with_mismatching_length_mask)

    assert not impressions_with_mismatching_length_mask.any(skipna=False)


def test_consistency_interactions_impressions_direct_link_interacted_items_are_inside_recommendation_list(dataset: ContentWiseImpressions):
    def get_series_index_on_recommendation_list(row) -> int:
        results: np.ndarray = np.where(row.recommended_series_list == row.series_id)
        indices: np.ndarray = results[0]

        if len(indices) == 0:
            return -1
        return indices[0]

    dataset: ddf.DataFrame = dataset.interactions.merge(right=dataset.impressions_direct_link,
                                                        how="inner",
                                                        left_on="recommendation_id",
                                                        right_index=True)

    dataset["recommendation_index"] = dataset.apply(get_series_index_on_recommendation_list,
                                                    axis="columns",
                                                    meta=("recommendation_index", "int32"))

    series_not_found_on_recommendation_mask: ddf.Series = (dataset.recommendation_index == -1)

    (series_not_found_on_recommendation_mask,) = ddf.compute(series_not_found_on_recommendation_mask)

    assert not series_not_found_on_recommendation_mask.any(skipna=False)


def test_consistency_interactions_impressions_non_direct_link_indirect_impressions_exist(dataset: ContentWiseImpressions):
    interactions: csr_matrix = dataset.URM["train"] + dataset.URM["validation"] + dataset.URM["test"]

    impressions_non_direct_link: csr_matrix = dataset.URM["impressions_non_direct_link"].copy()
    impressions_non_direct_link.data = np.ones_like(impressions_non_direct_link.data, dtype=np.int32)

    indirect_impressions_mask: np.ndarray = (interactions + impressions_non_direct_link).data > 1

    print(indirect_impressions_mask, indirect_impressions_mask.shape)
    print(indirect_impressions_mask[indirect_impressions_mask].shape)
    assert indirect_impressions_mask.any()


def test_consistency_interactions_impressions_direct_link_only_common_recommendation_ids(dataset: ContentWiseImpressions):
    unique_shared_recommendation_ids = (dataset
                                        .interactions
                                        .merge(right=dataset.impressions_direct_link,
                                               how="inner",
                                               left_on="recommendation_id",
                                               right_index=True)
                                        .recommendation_id
                                        .unique())

    # We add the missing recommendation id (-1) as part of a different recommendation id. The merge above removes this
    # value, we add its count here.
    num_unique_shared_recommendation_ids = unique_shared_recommendation_ids.shape[0] + 1

    (num_unique_shared_recommendation_ids,) = ddf.compute(num_unique_shared_recommendation_ids)

    assert num_unique_shared_recommendation_ids == dataset.metadata["num_recommendations"]


def test_consistency_interactions_impressions_non_direct_link_only_common_user_ids(dataset: ContentWiseImpressions):
    # NOTE: We calculate uniqueness of user_ids on the impressions_non_direct_link due to the high impact on memory
    # that the merges take if not done in this way.
    unique_user_ids_on_impressions_non_direct_link = (dataset
                                                     .impressions_non_direct_link
                                                     .reset_index(drop=False)
                                                     .user_id
                                                     .unique()
                                                     .to_frame(name='user_id'))

    unique_shared_user_ids = (dataset
                              .interactions
                              .merge(right=unique_user_ids_on_impressions_non_direct_link,
                                     how="inner",
                                     left_on="user_id",
                                     right_on="user_id")
                              .user_id
                              .unique())

    num_unique_shared_user_ids = unique_shared_user_ids.shape[0]

    (num_unique_shared_user_ids,) = ddf.compute(num_unique_shared_user_ids)

    assert num_unique_shared_user_ids == dataset.metadata["num_users"]


def test_na():
    with pytest.raises(AssertionError):
        should_fail = np.array([1, np.NaN, 2, 3])
        should_fail_mask = np.isnan(np.array(should_fail))

        assert not should_fail_mask.any()


def test_na_pd():
    with pytest.raises(AssertionError):
        should_fail = pd.Series(np.array([1, np.NaN, 2, 3]))
        should_fail_mask = should_fail.isna()

        assert not should_fail_mask.any()


def test_na_dask():
    with pytest.raises(AssertionError):
        should_fail = ddf.from_pandas(pd.Series(np.array([1, np.NaN, 2, 3])), npartitions=1)
        should_fail_mask = should_fail.isna().compute()

        assert not should_fail_mask.any()


def test_na_2_np():
    with pytest.raises(AssertionError):
        should_fail = np.array([np.NaN, np.NaN])
        should_fail_mask = np.isnan(np.array(should_fail))

        assert not should_fail_mask.any()


def test_na_2_pd():
    with pytest.raises(AssertionError):
        should_fail = pd.Series(np.array([np.NaN, np.NaN]))
        should_fail_mask = should_fail.isna()

        assert not should_fail_mask.any()


def test_na_2_dask():
    with pytest.raises(AssertionError):
        should_fail = ddf.from_pandas(pd.Series(np.array([np.NaN, np.NaN])), npartitions=1)
        should_fail_mask = should_fail.isna().compute()

        assert not should_fail_mask.any()


def test_na_3_np():
    should_pass = np.array([])
    should_pass_mask = np.isnan(np.array(should_pass))

    assert not should_pass_mask.any()


def test_na_3_pd():
    should_pass = pd.Series(np.array([]))
    should_pass_mask = should_pass.isna()

    assert not should_pass_mask.any()


def test_na_3_dask():
    should_pass = ddf.from_pandas(pd.Series(np.array([])), npartitions=1)
    should_pass_mask = should_pass.isna().compute()

    assert not should_pass_mask.any()


def test_na_4_np():
    should_pass = np.array([1, 2, 3])
    should_pass_mask = np.isnan(np.array(should_pass))

    assert not should_pass_mask.any()


def test_na_4_pd():
    should_pass = pd.Series(np.array([1, 2, 3]))
    should_pass_mask = should_pass.isna()

    assert not should_pass_mask.any()


def test_na_4_dask():
    should_pass = ddf.from_pandas(pd.Series(np.array([1, 2, 3])), npartitions=1)
    should_pass_mask = should_pass.isna().compute()

    assert not should_pass_mask.any()
