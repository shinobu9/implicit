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
    if use_gpu:
        print("gpu loss")
        calculate_loss = implicit.gpu.als.calculate_loss

    else:
        print("cpu loss")
        calculate_loss = implicit.cpu.als.calculate_loss

    # Ciu = random(
    #     m=100,
    #     n=100,
    #     density=0.05,
    #     format="coo",
    #     dtype=np.float32,
    #     random_state=42,
    #     data_rvs=None,
    # ).T.tocsr()

    # generate_dataset(path="/home/kamenskaya-el/adaptive-als/ml-100k", variant="100k", num_test_ratings=19)
    # titles, ratings = get_movielens(variant="100k", split="train")
    # ratings.data[ratings.data < 4.0] = 0
    # ratings.eliminate_zeros()
    # ratings.data = np.ones(len(ratings.data))
    # user_ratings = ratings.T.tocsr()

    counts = csr_matrix(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    user_ratings = counts.T.tocsr()


    model = AlternatingLeastSquares(
        factors=4,
        regularization=10,
        iterations=10,
        dtype=np.float32,
        random_state=23,
        use_native=False,
        use_cg=False,
        use_gpu=use_gpu,
        calculate_training_loss=True
    )

    model.fit(user_ratings, show_progress=True)

    item_factors, user_factors = model.item_factors, model.user_factors

    # titles, ratings = get_movielens(variant="100k", split="test")
    # ratings.data[ratings.data < 4.0] = 0
    # ratings.eliminate_zeros()
    # ratings.data = np.ones(len(ratings.data))
    # user_ratings = ratings.T.tocsr()

    counts = csr_matrix(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    user_ratings_test = counts.T.tocsr()

    # loss, rmse, auc = calculate_loss(user_ratings, user_factors, item_factors, regularization=10)
    print(implicit.cpu._als.calculate_auc_loss(user_ratings+user_ratings_test, user_factors, item_factors))
    # print(calculate_loss(user_ratings+user_ratings_test, user_factors, item_factors, regularization=10))
    # print(loss)
    # print(rmse)
    # print(auc)



if __name__ == "__main__":
    test_calculate_loss_simple(False)