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
    variant = "100k"
    generate_dataset(variant=variant, num_test_ratings=10, eval_percent=0.1, min_rating=4.0, 
                     path="/home/kamenskaya-el/adaptive-als/", 
                     outputpath="~/adaptive-als/implicit_datasets/")
    titles, ratings = get_movielens(variant=variant, split="train")
    # ratings.data[ratings.data <= 2.0] = 0
    # ratings.data[ratings.data >= 4.0] = 1
    # ratings.data[(ratings.data < 4.0)&(ratings.data > 2.0)] = 0
    ratings.eliminate_zeros()
    # ratings.data = np.ones(len(ratings.data))
    user_ratings = ratings.T.tocsr()

    model = AlternatingLeastSquares(
        factors=6,
        regularization=0.1,
        iterations=5,
        dtype=np.float32,
        random_state=23,
        use_native=False, #ONLY with zeroes and vanilla
        use_cg=True,
        use_gpu=use_gpu,
        calculate_training_loss=True,
        alpha=40,
        use_projections=True
    )

    # counts = csr_matrix(
    #     [
    #         [1, 1, 1, -1, 1, 0],
    #         [-1, 1, 0, 1, 0, 1],
    #         [-1, 0, 1, 1, 0, -1],
    #         [0, 1, 1, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, -1, 0],
    #         [0, 1, 1, 1, 0, 0],
    #     ],
    #     dtype=np.float32,
    # )
    # user_ratings = counts.T.tocsr()

    projections = False
    zero_padding = False

    setup = "proj"
    if setup == "zero":
        zero_padding = True
    elif setup == "proj":
        projections = True

    name = f"{variant}-{setup}"

    model.fit(user_ratings, show_progress=True,  xavier_init="uniform", 
              zero_padding=zero_padding, projections=projections, gamma=0.2, min_embedding=2, beta=1,
              loss_csv="/home/kamenskaya-el/adaptive-als/implicit/111losses/{name}.csv", 
              loss_png=f"/home/kamenskaya-el/adaptive-als/implicit/111plots/{name}.png")

    item_factors, user_factors = model.item_factors, model.user_factors

    _, ratings = get_movielens(variant="100k", split="test")
    ratings.eliminate_zeros()
    user_ratings_test = ratings.T.tocsr()

    # print(f"AUC: {implicit.cpu._als.calculate_auc_loss(user_ratings_test, user_factors, item_factors)}")
    print(f"RMSE: {implicit.cpu._als.calculate_rmse_loss(user_ratings_test, user_factors, item_factors)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_calculate_loss_simple(False)