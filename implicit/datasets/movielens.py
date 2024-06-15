import logging
import os

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from implicit.datasets import _download

log = logging.getLogger("implicit")


URL_BASE = "https://github.com/benfred/recommender_data/releases/download/v1.0/"
# URL_BASE = "https://github.com/shinobu9/implicit/blob/80e2193a4cb953ae776972b38cb38cf1153ed262/"


# def get_movielens(variant="100k"):
#     """Gets movielens datasets

#     Parameters
#     ---------
#     variant : string
#         Which version of the movielens dataset to download. Should be one of '20m', '10m',
#         '1m' or '100k'.

#     Returns
#     -------
#     movies : ndarray
#         An array of the movie titles.
#     ratings : csr_matrix
#         A sparse matrix where the row is the movieId, the column is the userId and the value is
#         the rating.
#     """
#     filename = f"movielens_{variant}_train.hdf5"

#     path = os.path.join(_download.LOCAL_CACHE_DIR, filename)
#     if not os.path.isfile(path):
#         log.info("Downloading dataset to '%s'", path)
#         _download.download_file(URL_BASE + filename, path)
#     else:
#         log.info("Using cached dataset at '%s'", path)

#     with h5py.File(path, "r") as f:
#         m = f.get("movie_user_ratings")
#         plays = csr_matrix((m.get("data"), m.get("indices"), m.get("indptr")))
#         return np.array(f["movie"].asstr()[:]), plays

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



def generate_dataset(path, variant="100k", outputpath="~/adaptive-als/implicit_datasets/", split=0.9):
    if variant == "20m":
        ratings, movies = _read_dataframes_20M(path)
    elif variant == "100k":
        ratings, movies = _read_dataframes_100k(path)
    else:
        ratings, movies = _read_dataframes(path)

    _csv_from_dataframe_split(ratings, variant, split, outputpath)


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


def _csv_from_dataframe_split(ratings, variant, split, outputpath):
    ratings = ratings.sort_values(by='timestamp')
    split_index = int(split * len(ratings))
    ratings_train = ratings.iloc[:split_index]
    ratings_test = ratings.iloc[split_index:]
    ratings_train = ratings_train.sample(frac=1, random_state=42).reset_index(drop=True)
    ratings_test = ratings_test.sample(frac=1, random_state=42).reset_index(drop=True)

    ratings_train.to_csv(os.path.join(outputpath, f"movielens_{variant}_train.csv"))
    print(f'Saved train data to {os.path.join(outputpath, f"movielens_{variant}_train.csv")}')

    ratings_test.to_csv(os.path.join(outputpath, f"movielens_{variant}_test.csv"))
    print(f'Saved test data to {os.path.join(outputpath, f"movielens_{variant}_test.csv")}')


def _csr_from_dataframe(ratings, movies):
    # transform ratings dataframe into a sparse matrix
    m = coo_matrix(
        (ratings["rating"].astype(np.float32), (ratings["movieId"], ratings["userId"]))
    ).tocsr()
    titles = np.empty(shape=(movies.movieId.max() + 1,), dtype=np.object_)
    titles[movies.movieId] = movies.title
    return titles, m