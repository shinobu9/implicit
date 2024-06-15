import logging
import os

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from implicit.datasets import _download

log = logging.getLogger("implicit")


URL_BASE = "https://github.com/benfred/recommender_data/releases/download/v1.0/"
# URL_BASE = "https://github.com/shinobu9/implicit/blob/80e2193a4cb953ae776972b38cb38cf1153ed262/"


def get_movielens(path="~/adaptive-als/implicit_datasets/", variant="100k", split="train"):
    import pandas

    if variant == "20m":
        ratings = pandas.read_csv(os.path.join(path, f"movielens_20m_{split}.csv"))
        movies = pandas.read_csv(os.path.join("/home/kamenskaya-el/adaptive-als/ml-20m/", "movies.csv"))
    elif variant == "100k":
        ratings = pandas.read_csv(os.path.join(path, f"movielens_100k_{split}.csv"))
        movies = pandas.read_csv(
            os.path.join("/home/kamenskaya-el/adaptive-als/ml-100k/", "u.item"),
            names=["movieId", "title"],
            usecols=[0, 1],
            delimiter="|",
            encoding="ISO-8859-1",
        )
    else:
        ratings = pandas.read_csv(os.path.join(path, f"movielens_25m_{split}.csv"))
        movies = pandas.read_csv(os.path.join("/home/kamenskaya-el/adaptive-als/ml-25m/", "movies.csv"))

    return _csr_from_dataframe(ratings, movies)



def generate_dataset(path, variant="100k", num_test_ratings=10, outputpath="~/adaptive-als/implicit_datasets/"):
    if variant == "20m":
        ratings, _ = _read_dataframes_20M(path)
    elif variant == "100k":
        ratings, _ = _read_dataframes_100k(path)
    else:
        ratings, _ = _read_dataframes(path)

    split_and_save_csv(ratings, variant, num_test_ratings, outputpath)


def _read_dataframes_20M(path):
    """reads in the movielens 20M"""
    import pandas

    ratings = pandas.read_csv(os.path.join(path, "ratings.csv"))
    movies = pandas.read_csv(os.path.join(path, "movies.csv"))

    return ratings, movies


def _read_dataframes_100k(path):
    """reads in the movielens 100k dataset"""
    import pandas

    ratings = pandas.read_table(
        os.path.join(path, "u.data"), names=["userId", "movieId", "rating", "timestamp"]
    )

    movies = pandas.read_csv(
        os.path.join(path, "u.item"),
        names=["movieId", "title"],
        usecols=[0, 1],
        delimiter="|",
        encoding="ISO-8859-1",
    )

    return ratings, movies


def _read_dataframes(path):
    import pandas

    ratings = pandas.read_csv(
        os.path.join(path, "ratings.dat"),
        delimiter="::",
        names=["userId", "movieId", "rating", "timestamp"],
    )

    movies = pandas.read_table(
        os.path.join(path, "movies.dat"), delimiter="::", names=["movieId", "title", "genres"]
    )
    return ratings, movies


def _hfd5_from_dataframe(ratings, movies, outputfilename):
    # transform ratings dataframe into a sparse matrix
    m = coo_matrix(
        (ratings["rating"].astype(np.float32), (ratings["movieId"], ratings["userId"]))
    ).tocsr()

    with h5py.File(outputfilename, "w") as f:
        # write out the ratings matrix
        g = f.create_group("movie_user_ratings")
        g.create_dataset("data", data=m.data)
        g.create_dataset("indptr", data=m.indptr)
        g.create_dataset("indices", data=m.indices)

        # write out the titles as a numpy array
        titles = np.empty(shape=(movies.movieId.max() + 1,), dtype=np.object_)
        titles[movies.movieId] = movies.title
        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset("movie", (len(titles),), dtype=dt)
        dset[:] = titles


def split_and_save_csv(ratings, variant, num_test_ratings, outputpath):
    import pandas

    assert num_test_ratings < 20, "num_test_ratings must be less than 20, not all users have enough ratings"

    test_ratings = ratings.groupby('userId', group_keys=False).apply(lambda x: x.sort_values('timestamp').tail(num_test_ratings))

    ratings_train = ratings.copy()
    ratings_train.loc[ratings_train.index.isin(test_ratings.index), 'rating'] = 0

    ratings_test = ratings.copy()
    ratings_test.loc[~ratings_test.index.isin(test_ratings.index), 'rating'] = 0

    ratings_train = ratings_train.sample(frac=1, random_state=42).reset_index(drop=True)
    ratings_test = ratings_test.sample(frac=1, random_state=42).reset_index(drop=True)

    ratings_train.to_csv(os.path.join(outputpath, f"movielens_{variant}_train.csv"), index=False)
    print(f'Saved train data to {os.path.join(outputpath, f"movielens_{variant}_train.csv")}')
    print(f'Num items {len(ratings_train["movieId"].unique())}')
    print(f'Num users {len(ratings_train["userId"].unique())}')
    print(f"Num non null ratings {len(ratings_train[ratings_train['rating'] != 0])}")

    ratings_test.to_csv(os.path.join(outputpath, f"movielens_{variant}_test.csv"), index=False)
    print(f'Saved test data to {os.path.join(outputpath, f"movielens_{variant}_test.csv")}')
    print(f'Num items {len(ratings_test["movieId"].unique())}')
    print(f'Num users {len(ratings_test["userId"].unique())}')
    print(f"Num non null ratings {len(ratings_test[ratings_test['rating'] != 0])}")


def _csr_from_dataframe(ratings, movies):
    # transform ratings dataframe into a sparse matrix
    m = coo_matrix(
        (ratings["rating"].astype(np.float32), (ratings["movieId"], ratings["userId"]))
    ).tocsr()
    titles = np.empty(shape=(movies.movieId.max() + 1,), dtype=np.object_)
    titles[movies.movieId] = movies.title
    return titles, m