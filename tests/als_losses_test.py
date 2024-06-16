import pickle
import unittest

import implicit.cpu._als
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from recommender_base_test import RecommenderBaseTestMixin, get_checker_board
from scipy.sparse import coo_matrix, csr_matrix, random

import implicit
from implicit.als import AlternatingLeastSquares
from implicit.gpu import HAS_CUDA
from implicit.datasets.movielens import get_movielens, generate_dataset
import implicit.cpu
import logging

log = logging.getLogger("implicit")

if HAS_CUDA:

    class GPUALSTest(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return AlternatingLeastSquares(
                factors=32, regularization=0, random_state=23, use_gpu=True
            )

    class GPUALSTestFloat16(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return AlternatingLeastSquares(
                factors=32, regularization=0, random_state=23, use_gpu=True, dtype=np.float16
            )


def test_calculate_loss_simple(use_gpu):

    generate_dataset(variant="100k", num_test_ratings=10, eval_percent=0.1, min_rating=4.0, path="/home/kamenskaya-el/adaptive-als/", outputpath="~/adaptive-als/implicit_datasets/")
    titles, ratings = get_movielens(variant="100k", split="train")
    # ratings.data[ratings.data <= 2.0] = 0
    # ratings.data[ratings.data >= 4.0] = 1
    # ratings.data[(ratings.data < 4.0)&(ratings.data > 2.0)] = 0
    ratings.eliminate_zeros()
    ratings.data = np.ones(len(ratings.data))
    user_ratings = ratings.T.tocsr()

    model = AlternatingLeastSquares(
        factors=40,
        regularization=0.1,
        iterations=30,
        dtype=np.float32,
        random_state=23,
        use_native=True,
        use_cg=True,
        use_gpu=use_gpu,
        calculate_training_loss=True,
        alpha=40,
    )

    model.fit(user_ratings, show_progress=True,  xavier_init="uniform", gamma=0.02, zero_padding=True, min_embedding=2)

    item_factors, user_factors = model.item_factors, model.user_factors
    print(f"Ratings {user_ratings.shape} | User factors {user_factors.shape} | Item factors {item_factors.shape}")

    _, ratings = get_movielens(variant="100k", split="test")
    # ratings.data[ratings.data <= 2.0] = 0
    # ratings.data[ratings.data >= 4.0] = 1
    # ratings.data[(ratings.data < 4.0)&(ratings.data > 2.0)] = 0
    ratings.eliminate_zeros()
    ratings.data = np.ones(len(ratings.data))
    user_ratings_test = ratings.T.tocsr()

    # counts = csr_matrix(
    #     [
    #         [4, 1, 0, 2, 5, 3],
    #         [2, 0, 3, 0, 1, 0],
    #         [1, 0, 4, 0, 3, 2],
    #         [0, 5, 0, 4, 0, 5],
    #     ],
    #     dtype=np.float32,
    # )
    # user_ratings_test = counts.T.tocsr()

    # print(f"AUC: {implicit.cpu._als.calculate_auc_loss(user_ratings_test, user_factors, item_factors)}")
    print(f"RMSE: {implicit.cpu._als.calculate_rmse_loss(user_ratings_test, user_factors, item_factors)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_calculate_loss_simple(False)