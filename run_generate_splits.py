# -*- coding: utf-8 -*-
"""
@author: F. B. Pérez Maurera
"""
import logging
import os
import time
import argparse

from Utils.config import configure_logger
from Utils.dataset import ContentWiseImpressions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        default=False,
                        help="Write dataset on disk",
                        action='store_true')
    parser.add_argument('-i',
                        '--items',
                        default=False,
                        help="Load URM using items",
                        action='store_true')
    parser.add_argument('-s',
                        '--series',
                        help="Load URM using series.",
                        default=False,
                        action='store_true'
                        )
    input_flags = parser.parse_args()
    print(input_flags)

    configure_logger(logs_dir=os.path.join(".", "logs"),
                     root_filename=os.path.basename(__file__))

    logger = logging.getLogger("contentwise-impressions")

    ts = time.time()

    store_dataset = input_flags.dataset
    use_items = input_flags.items
    use_series = input_flags.series

    logger.info("Dataset initialization")
    dataset = ContentWiseImpressions(dataset_variant=ContentWiseImpressions.Variant.CW10M)
    dataset.read_dataset()

    if store_dataset:
        dataset.save_dataset()

    if use_items:
        dataset.read_urm_splits(use_items=True, use_cache=False)
        dataset.save_urm(use_items=True)
        logger.info(f"Generated and saved URM splits using items")

    if use_series:
        dataset.read_urm_splits(use_items=False, use_cache=False)
        dataset.save_urm(use_items=False)
        logger.info(f"Generated and saved URM splits using series")

    te = time.time()

    logger.info(f"Success - Time elapsed: {te - ts:.2f}s")
