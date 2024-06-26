""" Implicit Alternating Least Squares """
import functools
import heapq
import logging
import time

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from ..utils import check_blas_config, check_csr, check_random_state, nonzeros
from . import _als
from .matrix_factorization_base import MatrixFactorizationBase

log = logging.getLogger("implicit")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)


class AlternatingLeastSquares(MatrixFactorizationBase):
    """Alternating Least Squares

    A Recommendation Model based off the algorithms described in the paper 'Collaborative
    Filtering for Implicit Feedback Datasets' with performance optimizations described in
    'Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative
    Filtering.'

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    regularization : float, optional
        The regularization factor to use
    alpha : float, optional
        The weight to give to positive examples.
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    use_native : bool, optional
        Use native extensions to speed up model fitting
    use_cg : bool, optional
        Use a faster Conjugate Gradient solver to calculate factors
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
    num_threads : int, optional
        The number of threads to use for fitting the model and batch recommend calls.
        Specifying 0 means to default to the number of cores on the machine.
    random_state : int, numpy.random.RandomState, np.random.Generator or None, optional
        The random state for seeding the initial item and user factors.
        Default is None.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """

    def __init__(
        self,
        factors=100,
        regularization=0.01,
        alpha=1.0,
        dtype=np.float32,
        use_native=True,
        use_cg=True,
        iterations=15,
        calculate_training_loss=True,
        num_threads=0,
        random_state=None,
        use_projections=False,
    ):
        super().__init__(num_threads=num_threads)

        # parameters on how to factorize
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha

        # options on how to fit the model
        self.dtype = np.dtype(dtype)
        self.use_native = use_native
        self.use_cg = use_cg
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.fit_callback = None
        self.cg_steps = 3
        self.random_state = random_state
        self.use_projections = use_projections


        # cache for item factors squared
        self._YtY = None
        # cache for user factors squared
        self._XtX = None

        # adaptive embedding parameters
        self._du = None
        self._di = None
        self._A = None
        self._B = None
        self._Xhat = None
        self._Yhat = None

        check_blas_config()

    def fit(self, user_items, show_progress=True, callback=None, xavier_init=None, 
            zero_padding=False, projections=False, gamma=0.02, min_embedding=2, beta=1.,
            loss_csv="loss.csv", loss_png="loss.png"):
        """Factorizes the user_items matrix.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The user_items matrix does double duty here. It defines which items are liked by which
        users (P_ui in the original paper), as well as how much confidence we have that the user
        liked the item (C_ui).

        The negative items are implicitly defined: This code assumes that positive items in the
        user_items matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.
        Negative items can also be passed with a higher confidence value by passing a negative
        value, indicating that the user disliked the item.

        Parameters
        ----------
        user_items: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the users, the columns are the items liked that user,
            and the value is the confidence that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        callback: Callable, optional
            Callable function on each epoch with such arguments as epoch, elapsed time and progress
        xavier_init: string, optional
            Type of matrix initialization to use. If set to None uses default random initialization,
            other options are "normal" and "uniform"
        zero_padding: bool, optional
            Whether to pad adaptive embeddings with zeroes
        projections: bool, optional
            Whether to project adaptive embeddings onto common space
        gamma : float, optional
            The hyperparameter to scale the embedding size.
            Only used when zero_padding or projections is True
        min_embedding: int, optional
            Minimal embedding size when training with adaptive embeddings.
            Only used when zero_padding or projections is True
        """
        # initialize the random state
        random_state = check_random_state(self.random_state)

        self.losses = []

        Cui = check_csr(user_items)
        if Cui.dtype != np.float32:
            Cui = Cui.astype(np.float32)

        # Give the positive examples more weight if asked for
        if self.alpha != 1.0:
            Cui = self.alpha * Cui

        s = time.time()
        Ciu = Cui.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()
        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            if xavier_init is None:
                self.user_factors = random_state.random((users, self.factors), dtype=self.dtype) * 0.01
            elif xavier_init == "normal":
                self.user_factors = random_state.normal(loc=0, scale=np.sqrt(2/(self.factors+items)), size=(users, self.factors)).astype(self.dtype)
            elif xavier_init == "uniform":
                scale = np.sqrt(6/(self.factors+items))
                self.user_factors = random_state.uniform(low=-scale, high=scale, size=(users, self.factors)).astype(self.dtype)

        if self.item_factors is None:
            if xavier_init is None:
                self.item_factors = random_state.random((items, self.factors), dtype=self.dtype) * 0.01
            elif xavier_init == "normal":
                self.item_factors = random_state.normal(loc=0, scale=np.sqrt(2/(self.factors+users)), size=(items, self.factors)).astype(self.dtype)
            elif xavier_init == "uniform":
                scale = np.sqrt(6/(self.factors+users))
                self.item_factors = random_state.uniform(low=-scale, high=scale, size=(items, self.factors)).astype(self.dtype)

        log.debug("Initialized factors in %s", time.time() - s)

# --------------------------------------------------------------------------------------------------
        if zero_padding or projections:
            s = time.time()
            user_interactions = Ciu.getnnz(axis=0)
            item_interactions = Ciu.getnnz(axis=1)
            fu_med = np.median(user_interactions)
            fi_med = np.median(item_interactions)
            self._du = np.round(user_interactions / (gamma * fu_med)).astype(np.int32)
            self._di = np.round(item_interactions / (gamma * fi_med)).astype(np.int32)

            for u in range(users):
                if self._du[u] < self.factors:
                    if self._du[u] == 0: self._du[u] = min_embedding
                    self.user_factors[u,self._du[u]:] = 0.0
                else:
                    self._du[u] = self.factors

            for i in range(items):
                if self._di[i] < self.factors:
                    if self._di[i] == 0: self._di[i] = min_embedding
                    self.item_factors[i,self._di[i]:] = 0.0
                else:
                    self._di[i] = self.factors
            print(f"Median user pop: {fu_med}, median item pop {fi_med}")
            print(f"Small user embedding after padding: {sum(self._du<self.factors)}/{len(self._du)}")
            print(f"Small item embedding after padding: {sum(self._di<self.factors)}/{len(self._di)}")

            log.debug("Initialized embedding sizes in %s", time.time() - s)


        if projections:
            s = time.time()
            self._A = []
            self._B = []
            if xavier_init == "uniform":
                for i in range(self.factors):
                    scale = np.sqrt(6/(self.factors+i))
                    A = random_state.uniform(low=-scale, high=scale, size=(self.factors, self.factors)).astype(self.dtype)
                    A[:, i:] = 0
                    self._A.append(A)
            elif xavier_init == "normal":
                for i in range(self.factors):
                    A = random_state.normal(loc=0, scale=np.sqrt(2/(self.factors+i)), size=(self.factors, self.factors, self.factors)).astype(self.dtype)
                    A[:, i:] = 0
                    self._A.append(A)
            elif xavier_init is None:
                for i in range(self.factors):
                    A = random_state.random((self.factors, self.factors), dtype=self.dtype) * 0.01
                    A[:, i:] = 0
                    self._A.append(A)
            self._A.append(np.eye(self.factors, self.factors))
            self._A = np.array(self._A)
            self._B = self._A.copy()

            self._Xhat = self.user_factors.copy()
            self._Yhat = self.item_factors.copy()

            log.debug("Initialized projection matrices in %s", time.time() - s)

        # print(f"\nUser factors BEFORE \n{self.user_factors}\nItem factors BEFORE \n{self.item_factors}\n")
        # print(f"\nAp BEFORE \n{self._A}\nBp BEFORE \n{self._B}\n")
        
# --------------------------------------------------------------------------------------------------
        # invalidate cached norms and squared factors
        self._item_norms = self._user_norms = None
        self._YtY = None
        self._XtX = None
        loss = None

        solver = self.solver

        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            # alternate between learning the user_factors from the item_factors and vice-versa
            for iteration in range(self.iterations):
                s = time.time()
                if projections:
                    # 1. Compute all y_hat_i
                    AduXu(self._B, self.item_factors, self._Yhat, self._di)
                    # 2. Compute all Ap
                    for p in range(1,self.factors):
                        update_Ap(self._A, p, self._du, self.user_factors, self._Yhat, Cui, beta, self.alpha)
                    # 3. Compute all x_hat_u
                    AduXu(self._A, self.user_factors, self._Xhat, self._du)
                    # 4. Compute all Bp
                    for p in range(1,self.factors):
                        update_Ap(self._B, p, self._di, self.item_factors, self._Xhat, Ciu, beta, self.alpha)
                    # 5. Compute all y_hat_i
                    AduXu(self._B, self.item_factors, self._Yhat, self._di)
                    # 6. Compute all x_u
                    solver(
                        Cui,
                        self.user_factors,
                        self._Yhat,
                        self.regularization,
                        self._A,
                        self._du,
                        num_threads=self.num_threads,
                    )
                    # pad_with_zeroes(self.user_factors, self._du, self.factors, min_embedding)
                    # 7. Compute all x_hat_u
                    AduXu(self._A, self.user_factors, self._Xhat, self._du)
                    # 8. Compute all y_i
                    solver(
                        Ciu,
                        self.item_factors,
                        self._Xhat,
                        self.regularization,
                        self._B,
                        self._di,                        
                        num_threads=self.num_threads,
                    )
                    # pad_with_zeroes(self.item_factors, self._di, self.factors, min_embedding)
                    # print(f"\nITERATION {iteration}\nUser factors \n{self.user_factors}\nItem factors \n{self.item_factors}\n")

                else:
                    solver(
                        Cui,
                        self.user_factors,
                        self.item_factors,
                        self.regularization,
                        num_threads=self.num_threads,
                    )
                    if zero_padding:
                        pad_with_zeroes(self.user_factors, self._du, self.factors, min_embedding)
                    solver(
                        Ciu,
                        self.item_factors,
                        self.user_factors,
                        self.regularization,
                        num_threads=self.num_threads,
                    )
                    if zero_padding:
                        pad_with_zeroes(self.item_factors, self._di, self.factors, min_embedding)
                    progress.update(1)

                if self.calculate_training_loss:
                    loss = _als.calculate_loss(
                        Cui,
                        self.user_factors,
                        self.item_factors,
                        self.regularization,
                        num_threads=self.num_threads,
                    )
                    progress.set_postfix({"loss": loss})

                    if not show_progress:
                        log.info("loss %.4f", loss)
                    print(f"loss : {loss}")
                    self.losses.append(loss)

                # Backward compatibility
                if not callback:
                    callback = self.fit_callback
                if callback:
                    callback(iteration, time.time() - s, loss)

        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

        self._check_fit_errors()
        self.save_losses_to_csv(self.losses, loss_csv)
        self.plot_losses(self.losses, loss_png)
        # print(f"\nAp AFTER \n{self._A}\nBp AFTER \n{self._B}\n")

    def save_losses_to_csv(self, losses, output_file):
        df = pd.DataFrame({'Iteration': range(1, len(losses) + 1), 'Loss': losses})
        df.to_csv(output_file, index=False)

    def plot_losses(self, losses, output_image):
        df = pd.DataFrame({'Iteration': range(1, len(losses) + 1), 'Loss': losses})
        plt.figure(figsize=(8, 6))
        plt.plot(df['Iteration'], df['Loss'], marker='o', color='b', linestyle='-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        # plt.title('Loss per Iteration')
        plt.grid(True)
        plt.savefig(output_image)


    def recalculate_user(self, userid, user_items):
        """Recalculates factors for a batch of users

        This method recalculates factors for a batch of users and returns
        the factors without storing on the object. For updating the model
        using 'partial_fit_users'

        Parameters
        ----------
        userid : Union[array_like, int]
            The userid or array of userids to recalculate
        user_items : csr_matrix
            Sparse matrix of (users, items) that contain the users that liked
            each item.
        """
        user_items = check_csr(user_items)

        # we're using the cholesky solver here on purpose, since for a full recompute
        users = 1 if np.isscalar(userid) else len(userid)

        if user_items.shape[0] != users:
            raise ValueError("user_items should have one row for every item in user")

        if self.alpha != 1.0:
            user_items = self.alpha * user_items

        user_factors = np.zeros((users, self.factors), dtype=self.dtype)
        _als._least_squares(
            self.YtY,
            user_items.indptr,
            user_items.indices,
            user_items.data.astype("float32"),
            user_factors,
            self.item_factors,
            self.regularization,
            num_threads=self.num_threads,
        )
        return user_factors[0] if np.isscalar(userid) else user_factors
    
    def recalculate_item(self, itemid, item_users):
        """Recalculates factors for a batch of items

        This recalculates factors for a batch of items, returns the newly
        calculated values without storing.

        Parameters
        ----------
        itemid : Union[array_like, int]
            The itemid or array of itemids to recalculate
        item_users : csr_matrix
            Sparse matrix of (items, users) that contain the users that liked
            each item
        """
        item_users = check_csr(item_users)

        if self.alpha != 1.0:
            item_users = self.alpha * item_users

        items = 1 if np.isscalar(itemid) else len(itemid)
        item_factors = np.zeros((items, self.factors), dtype=self.dtype)
        _als._least_squares(
            self.XtX,
            item_users.indptr,
            item_users.indices,
            item_users.data.astype("float32"),
            item_factors,
            self.user_factors,
            self.regularization,
            num_threads=self.num_threads,
        )
        return item_factors[0] if np.isscalar(itemid) else item_factors

    def partial_fit_users(self, userids, user_items):
        """Incrementally updates user factors

        This method updates factors for users specified by userids, given a
        sparse matrix of items that they have interacted with before. This
        allows you to retrain only parts of the model with new data, and
        avoid a full retraining when new users appear - or the liked
        items for an existing user change.

        Parameters
        ----------
        userids : array_like
            An array of userids to calculate new factors for
        user_items : csr_matrix
            Sparse matrix containing the liked items for each user. Each row in this
            matrix corresponds to a row in userids.
        """
        if len(userids) != user_items.shape[0]:
            raise ValueError("user_items must contain 1 row for every user in userids")

        # recalculate factors for each user in the input
        user_factors = self.recalculate_user(userids, user_items)

        # ensure that we have enough storage for any new users
        users, factors = self.user_factors.shape
        max_userid = max(userids)
        if max_userid >= users:
            self.user_factors = np.concatenate(
                [self.user_factors, np.zeros((max_userid - users + 1, factors), dtype=self.dtype)]
            )

        # update the stored factors with the newly calculated values
        self.user_factors[userids] = user_factors

        # clear any cached properties that are invalidated by this update
        self._user_norms = None
        self._XtX = None

    def partial_fit_items(self, itemids, item_users):
        """Incrementally updates item factors

        This method updates factors for items specified by itemids, given a
        sparse matrix of users that have interacted with them. This
        allows you to retrain only parts of the model with new data, and
        avoid a full retraining when new users appear - or the liked
        users for an existing item change.

        Parameters
        ----------
        itemids : array_like
            An array of itemids to calculate new factors for
        item_users : csr_matrix
            Sparse matrix containing the liked users for each item in itemids
        """
        if len(itemids) != item_users.shape[0]:
            raise ValueError("item_users must contain 1 row for every user in itemids")

        # recalculate factors for each item in the input
        item_factors = self.recalculate_item(itemids, item_users)

        # ensure that we have enough storage for any new items
        items, factors = self.item_factors.shape
        max_itemid = max(itemids)
        if max_itemid >= items:
            self.item_factors = np.concatenate(
                [self.item_factors, np.zeros((max_itemid - items + 1, factors), dtype=self.dtype)]
            )

        # update the stored factors with the newly calculated values
        self.item_factors[itemids] = item_factors

        # clear any cached properties that are invalidated by this update
        self._item_norms = None
        self._YtY = None

    def explain(self, userid, user_items, itemid, user_weights=None, N=10):
        """Provides explanations for why the item is liked by the user.

        Parameters
        ----------
        userid : int
            The userid to explain recommendations for
        user_items : csr_matrix
            Sparse matrix containing the liked items for the user
        itemid : int
            The itemid to explain recommendations for
        user_weights : ndarray, optional
            Precomputed Cholesky decomposition of the weighted user liked items.
            Useful for speeding up repeated calls to this function, this value
            is returned
        N : int, optional
            The number of liked items to show the contribution for

        Returns
        -------
        total_score : float
            The total predicted score for this user/item pair
        top_contributions : list
            A list of the top N (itemid, score) contributions for this user/item pair
        user_weights : ndarray
            A factorized representation of the user. Passing this in to
            future 'explain' calls will lead to noticeable speedups
        """

        user_items = check_csr(user_items)
        if self.alpha != 1.0:
            user_items = self.alpha * user_items

        # user_weights = Cholesky decomposition of Wu^-1
        # from section 5 of the paper CF for Implicit Feedback Datasets
        if user_weights is None:
            A, _ = user_linear_equation(
                self.item_factors, self.YtY, user_items, userid, self.regularization, self.factors
            )
            user_weights = scipy.linalg.cho_factor(A)
        seed_item = self.item_factors[itemid]

        # weighted_item = y_i^t W_u
        weighted_item = scipy.linalg.cho_solve(user_weights, seed_item)

        total_score = 0.0
        h = []
        h_len = 0
        for other_itemid, confidence in nonzeros(user_items, userid):
            if confidence < 0:
                continue

            factor = self.item_factors[other_itemid]
            # s_u^ij = (y_i^t W^u) y_j
            score = weighted_item.dot(factor) * confidence
            total_score += score
            contribution = (score, other_itemid)
            if h_len < N:
                heapq.heappush(h, contribution)
                h_len += 1
            else:
                heapq.heappushpop(h, contribution)

        items = (heapq.heappop(h) for i in range(len(h)))
        top_contributions = list((i, s) for s, i in items)[::-1]
        return total_score, top_contributions, user_weights

    @property
    def solver(self):
        if self.use_projections:
            if self.use_cg:
                solver = _als.least_squares_cg_proj if self.use_native else least_squares_cg_proj
                return functools.partial(solver, cg_steps=self.cg_steps)
            else:
                return _als.least_squares_proj if self.use_native else least_squares_proj
        else:
            if self.use_cg:
                solver = _als.least_squares_cg if self.use_native else least_squares_cg
                return functools.partial(solver, cg_steps=self.cg_steps)
            else:
                return _als.least_squares if self.use_native else least_squares

    @property
    def YtY(self):
        if self._YtY is None:
            Y = self.item_factors
            self._YtY = Y.T.dot(Y)
        return self._YtY

    @property
    def XtX(self):
        if self._XtX is None:
            X = self.user_factors
            self._XtX = X.T.dot(X)
        return self._XtX

    def to_gpu(self):
        """Converts this model to an equivalent version running on the gpu"""
        import implicit.gpu.als

        ret = implicit.gpu.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            dtype=self.dtype,
            iterations=self.iterations,
            calculate_training_loss=self.calculate_training_loss,
            random_state=self.random_state,
        )
        if self.user_factors is not None:
            ret.user_factors = implicit.gpu.Matrix(self.user_factors)
        if self.item_factors is not None:
            ret.item_factors = implicit.gpu.Matrix(self.item_factors)
        return ret

    def save(self, fileobj_or_path):
        args = {
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "regularization": self.regularization,
            "factors": self.factors,
            "num_threads": self.num_threads,
            "iterations": self.iterations,
            "use_native": self.use_native,
            "use_cg": self.use_cg,
            "cg_steps": self.cg_steps,
            "calculate_training_loss": self.calculate_training_loss,
            "dtype": self.dtype.name,
            "random_state": self.random_state,
            "alpha": self.alpha,
        }
        # filter out 'None' valued args, since we can't go np.load on
        # them without using pickle
        args = {k: v for k, v in args.items() if v is not None}
        np.savez(fileobj_or_path, **args)


def pad_with_zeroes(X, d_u, d, min_d):
    for u in range(X.shape[0]):
        X[u, d_u[u]:] = 0.0

def pad_with_zeroes_matrix(A):
    for u in range(A.shape[0]):
        A[u, :, u:] = 0

def AduXu(A, X, Xhat, d_u):
    for u in range(X.shape[0]):
        Xhat[u] = A[d_u[u]].dot(X[u])

def update_Ap(A, p, d_u, X, Yhat, Cui, beta, alpha):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    # YtCuY + regularization * I = YtY + regularization * I + Yt(Cu-I)
    users, _ = Cui.shape
    d = X.shape[1]
    QtQ = np.zeros((d*d, d*d)).astype(np.float32)
    I = np.eye(d*d, d*d).astype(np.float32)*beta
    Qtr = np.zeros((d*d)).astype(np.float32)

    # s = time.time()
    for u in range(users):
        if d_u[u] != p:
            continue
        for i, confidence in nonzeros(Cui, u):
            if confidence == 0:
                continue
            Qp = np.outer(Yhat[i], X[u]).flatten().astype(np.float32)
            assert Qp.shape[0] == d*d, "Qp shape does not equal (d*d,)"

            QtQ += np.outer(Qp,Qp).astype(np.float32)
            assert QtQ.shape == (d*d, d*d), "QtQ shape does not equal (d*d, d*d)"

            Qtr += confidence / alpha * Qp
            assert Qtr.shape[0] == d*d, "Qtr shape does not equal (d*d,)"

    # log.debug("Calculated Qp in %.3fs", time.time() - s)
    
    # s = time.time()
    QtQreg_inv = np.linalg.inv(QtQ + I)
    # log.debug("Calculated inverse in %.3fs", time.time() - s)
    A[p] = QtQreg_inv.dot(Qtr).reshape(d,d)

def least_squares(Cui, X, Y, regularization, num_threads=0):
    """For each user in Cui, calculate factors Xu for them
    using least squares on Y.

    Note: this is at least 10 times slower than the cython version included
    here.
    """
    users, n_factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        X[u] = user_factor(Y, YtY, Cui, u, regularization, n_factors)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def least_squares_proj(Cui, X, Y, regularization, A, d_u, num_threads=0):
    """For each user in Cui, calculate factors Xu for them
    using least squares on Y.

    Note: this is at least 10 times slower than the cython version included
    here.
    """
    users, n_factors = X.shape

    for u in range(users):
        Ywave = Y.dot(A[d_u[u]])
        YtY = Ywave.T.dot(Ywave)
        X[u] = user_factor(Y, A[d_u[u]].T.dot(YtY), Cui, u, regularization, n_factors)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------


def user_linear_equation(Y, YtY, Cui, u, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    # YtCuY + regularization * I = YtY + regularization * I + Yt(Cu-I)

    # accumulate YtCuY + regularization*I in A
    A = YtY + regularization * np.eye(n_factors)

    # accumulate YtCuPu in b
    b = np.zeros(n_factors)

    for i, confidence in nonzeros(Cui, u):
        factor = Y[i]

        if confidence > 0:
            b += confidence * factor
        else:
            confidence *= -1

        A += (confidence - 1) * np.outer(factor, factor)
    return A, b


def user_factor(Y, YtY, Cui, u, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    A, b = user_linear_equation(Y, YtY, Cui, u, regularization, n_factors)
    return np.linalg.solve(A, b)


# def item_factor(X, XtX, Cui, u, regularization, n_factors):
#     # Yu = (XtCuX + regularization * I)^-1 (XtCuPu)
#     A, b = user_linear_equation(X, XtX, Cui, u, regularization, n_factors)
#     return np.linalg.solve(A, b)


def least_squares_cg(Cui, X, Y, regularization, num_threads=0, cg_steps=3):
    users, factors = X.shape
    YtY = Y.T.dot(Y) + regularization * np.eye(factors, dtype=Y.dtype)

    for u in range(users):
        # start from previous iteration
        x = X[u]

        # calculate residual error r = (YtCuPu - (YtCuY.dot(Xu)
        r = -YtY.dot(x)
        for i, confidence in nonzeros(Cui, u):
            if confidence > 0:
                r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]
            else:
                confidence *= -1
                r += -(confidence - 1) * Y[i].dot(x) * Y[i]

        p = r.copy()
        rsold = r.dot(r)
        if rsold < 1e-20:
            continue

        for _ in range(cg_steps):
            # calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                if confidence < 0:
                    confidence *= -1

                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            # standard CG update
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            if rsnew < 1e-20:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

def least_squares_cg_proj(Cui, X, Y, regularization, A, d_u, num_threads=0, cg_steps=3):
    users, factors = X.shape

    for u in range(users):
        # start from previous iteration
        x = X[u]
        Ywave = Y.dot(A[d_u[u]])
        YtY = Ywave.T.dot(Ywave) + regularization * np.eye(factors, dtype=Ywave.dtype)

        # calculate residual error r = (YtCuPu - (YtCuY.dot(Xu)
        r = -YtY.dot(x)
        for i, confidence in nonzeros(Cui, u):
            if confidence > 0:
                r += (confidence - (confidence - 1) * Ywave[i].dot(x)) * Ywave[i]
            else:
                confidence *= -1
                r += -(confidence - 1) * Ywave[i].dot(x) * Ywave[i]

        p = r.copy()
        rsold = r.dot(r)
        if rsold < 1e-20:
            continue

        for _ in range(cg_steps):
            # calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                if confidence < 0:
                    confidence *= -1

                Ap += (confidence - 1) * Ywave[i].dot(p) * Ywave[i]

            # standard CG update
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            if rsnew < 1e-20:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x


calculate_loss = _als.calculate_loss
