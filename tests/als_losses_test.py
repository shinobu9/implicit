import pickle
import unittest

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from recommender_base_test import RecommenderBaseTestMixin, get_checker_board
from scipy.sparse import coo_matrix, csr_matrix, random

import implicit
from implicit.als import AlternatingLeastSquares
from implicit.gpu import HAS_CUDA


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

    Ciu = random(
        m=100,
        n=100,
        density=0.05,
        format="coo",
        dtype=np.float32,
        random_state=42,
        data_rvs=None,
    ).T.tocsr()

    model = AlternatingLeastSquares(
        factors=32,
        regularization=10,
        iterations=10,
        dtype=np.float32,
        random_state=23,
        use_native=False,
        use_cg=False,
        use_gpu=use_gpu,
    )

    model.fit(Ciu, show_progress=True)

    item_factors, user_factors = model.item_factors, model.user_factors
    print(item_factors)

    print(user_factors)

    loss = calculate_loss(Ciu, user_factors, item_factors, regularization=10)
    print(loss)


if __name__ == "__main__":
    test_calculate_loss_simple(False)