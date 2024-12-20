import numpy as np
import recommender_als as als
import recommender_lmafit as lmafit
import pandas as pd
import numpy.ma as ma
from abc import ABCMeta, abstractmethod

# Code by Bashir Rastegarpanah
# From https://github.com/rastegarpanah/antidote-data-framework
# Written for paper
# Bashir Rastegarpanah, Krishna P. Gummadi and Mark Crovella (2019).
# Fighting Fire with Fire: Using Antidote Data to Improve Polarization and Fairness of Recommender Systems.
# In: Proceedings of WSDM. Melbourne, Australia. doi:10.1145/3289600.3291002

def read_movielens_small(n_movies, n_users, data_dir='Data/MovieLens-small'):
    """
    Reads and processes the MovieLens small dataset to extract a subset of movie ratings and genres.

    Parameters:
    n_movies (int): The number of top movies to select based on the number of ratings.
    n_users (int): The number of top users to select based on the number of ratings.
    data_dir (str): The directory path where the MovieLens small dataset is stored. Defaults to 'Data/MovieLens-small'.

    Returns:
    tuple: A tuple containing:
        - ratings (DataFrame): A DataFrame with user IDs as rows and movie titles as columns, containing the ratings.
        - movie_genres (Series): A Series with movie titles as index and a list of genres as values.
    """
    # get ratings
    df = pd.read_csv('{}/ratings.csv'.format(data_dir))

    # create a dataframe with movie IDs on the rows
    # and user IDs on the columns
    ratings = df.pivot(index='movieId', columns='userId', values='rating')

    # put movie titles as index on rows
    movies = pd.read_csv('Data/{}/movies.csv'.format(data_dir))
    movieSeries = pd.Series(list(movies['title']),
                             index=movies['movieId'])
    ratings = ratings.rename(index=movieSeries)
    
    #read movie genres
    movie_genres = pd.Series(list(movies['genres']),index=movies['title'])
    movie_genres = movie_genres.apply(lambda s:s.split('|'))

    # select the top n_movies that have the most number of ratings
    num_ratings = (~ratings.isnull()).sum(axis=1)
    rows = num_ratings.nlargest(n_movies)
    ratings = ratings.loc[rows.index]
    
    # select the top n_users that have the most number of ratings
    num_ratings = (~ratings.isnull()).sum(axis=0)
    cols = num_ratings.nlargest(n_users)
    ratings = ratings[cols.index]

    # eliminate the users that have no ratings in this set
    null_columns = ratings.isnull().all(axis=0)
    null_column_ids = null_columns.index[null_columns]
    ratings = ratings.drop(null_column_ids, axis=1)
    ratings = ratings.T
    return ratings, movie_genres


def read_movielens_1M(n_movies, n_users, top_users, data_dir='Data/MovieLens-1M'):
    # get ratings
    df = pd.read_table('{}/ratings.dat'.format(data_dir),names=['UserID','MovieID','Rating','Timestamp'], 
                       sep='::', engine='python')

    # create a dataframe with movie IDs on the rows
    # and user IDs on the columns
    ratings = df.pivot(index='MovieID', columns='UserID', values='Rating')

    
    movies = pd.read_table('{}/movies.dat'.format(data_dir),
                         names=['MovieID', 'Title', 'Genres'], 
                         sep='::', engine='python')
                         
    user_info = pd.read_table('{}/users.dat'.format(data_dir),
                            names=['UserID','Gender','Age','Occupation','Zip-code'], 
                            sep='::', engine='python')
    user_info = user_info.rename(index=user_info['UserID'])[['Gender','Age','Occupation','Zip-code']]
    
    # put movie titles as index on rows
    movieSeries = pd.Series(list(movies['Title']), index=movies['MovieID'])
    ratings = ratings.rename(index=movieSeries)
    
    #read movie genres
    movie_genres = pd.Series(list(movies['Genres']),index=movies['Title'])
    movie_genres = movie_genres.apply(lambda s:s.split('|'))

    # select the top n_movies that have the most number of ratings
    num_ratings = (~ratings.isnull()).sum(axis=1)
    rows = num_ratings.nlargest(n_movies)
    ratings = ratings.loc[rows.index]
    
    if top_users:
        #select the top n_users that have the most number of ratings
        num_ratings = (~ratings.isnull()).sum(axis=0)
        cols = num_ratings.nlargest(n_users)
        ratings = ratings[cols.index]
    else:
        #pick first users in order 
        cols = ratings.columns[0:n_users]
        ratings = ratings[cols]

    ratings = ratings.T
    return ratings, movie_genres, user_info
    
def train_val_split(omega,alpha,split_axis=1):
    if split_axis==0:
        omega = omega.T
    
    def split_known_indices(x,alpha):
        known_ratings = x[x.values].index
        val = list(np.random.choice(known_ratings, size =int(np.rint(alpha*len(known_ratings))), replace=False))
        train = list(set(known_ratings) - set(val))
        return {'train':train, 'val':val}
        
    def f1(x, indices):
        a = x.copy()
        a[indices[x.name]] = True
        return a
    
    splits = omega.apply(split_known_indices, args=[alpha], axis=1)
    n,d = omega.shape
    T = np.zeros((n,d), dtype=bool)
    T = pd.DataFrame(T, index=omega.index, columns=omega.columns).copy()

    train_indices = splits.apply(lambda d:d['train'])
    val_indices =   splits.apply(lambda d:d['val'])
    
    omega_train = T.apply(f1, args=[train_indices], axis=1)
    omega_val = T.apply(f1, args=[val_indices], axis=1)
    
    if split_axis==0:
        omega_train = omega_train.T
        omega_val = omega_val.T
    return omega_train, omega_val

def compute_RMSE(X1,X2,omega):
    X1 = X1.mask(~omega)
    X2 = X2.mask(~omega)
    MSE = ((X1 - X2).pow(2).sum().sum()*1.0)/omega.sum().sum()
    return np.sqrt(MSE)
    
def antidote_effect(RS, X, X_antidote):
    ratings = pd.concat([X,X_antidote])
    pred,error = RS.fit_model(ratings)
    V = RS.get_V()
    U = RS.get_U().loc[X.index]
    return U,V,U.dot(V)


class MF():
    """
    Matrix Factorization (MF) base class.

    This class serves as a base class for various matrix factorization techniques. It provides common
    functionalities and attributes that can be used by derived classes to implement specific matrix
    factorization algorithms.

    Attributes:
        rank (int): The rank of the factorization.
        lambda_ (float): Regularization parameter.
        ratings (pd.DataFrame): The user-item ratings matrix.
        num_of_known_ratings_per_user (pd.Series): Number of known ratings per user.
        num_of_known_ratings_per_movie (pd.Series): Number of known ratings per movie.

    Methods:
        set_ratings(ratings):
            Sets the ratings matrix and updates the number of known ratings per user and movie.
        get_U():
            Returns the user feature matrix as a DataFrame.
        get_V():
            Returns the item feature matrix as a DataFrame.
        fit_model():
            Abstract method to fit the model to the provided ratings matrix.
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, rank, lambda_=1e-6, ratings=None):
        self.rank = rank
        self.lambda_ = lambda_
        if ratings is not None:
            self.ratings = ratings
            self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
            self.num_of_known_ratings_per_movie = (~self.ratings.isnull()).sum(axis=0)
    
    def set_ratings(self, ratings):
        """
        Sets the ratings matrix and updates the number of known ratings per user and movie.

        Args:
            ratings (pd.DataFrame): The user-item ratings matrix.
        """
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_movie = (~self.ratings.isnull()).sum(axis=0)
    
    def get_U(self):
        """
        Returns the user feature matrix as a DataFrame.

        Returns:
            pd.DataFrame: The user feature matrix.
        """
        return pd.DataFrame(self.U, index = self.ratings.index)
    
    def get_V(self):
        """
        Returns the item feature matrix as a DataFrame.

        Returns:
            pd.DataFrame: The item feature matrix.
        """
        return pd.DataFrame(self.V, columns = self.ratings.columns)
    
    @abstractmethod
    def fit_model():
        pass
    
        
class als_MF(MF):
    """
    Alternating Least Squares Matrix Factorization (ALS-MF) class.

    This class implements the ALS algorithm for matrix factorization. It inherits from the MF base class
    and overrides the fit_model method to provide the specific implementation of the ALS algorithm.

    Attributes:
        rank (int): The rank of the factorization.
        lambda_ (float): Regularization parameter.
        ratings (pd.DataFrame): The user-item ratings matrix.
        U (np.ndarray): User feature matrix.
        V (np.ndarray): Item feature matrix.
        pred (pd.DataFrame): Predicted ratings matrix.
        error (float): The error of the predicted ratings.

    Methods:
        fit_model(ratings=None, max_iter=50, threshold=1e-5):
            Fits the ALS model to the provided ratings matrix.
    """
    
    def fit_model(self, ratings=None, max_iter=50, threshold=1e-5):
        X = self.ratings if ratings is None else ratings
        self.ratings = X
        self.U, self.V = als.als(X, self.rank, self.lambda_, max_iter, threshold)
        self.pred = pd.DataFrame(self.U.dot(self.V),
                                 index = X.index,
                                 columns = X.columns)
        self.error = ma.power(ma.masked_invalid(X-self.pred),2).sum()
        return self.pred, self.error


class lmafit_MF(MF):
    """
    Low-rank Matrix Fitting (LMAFIT) class.

    This class implements the LMAFIT algorithm for matrix factorization. It inherits from the MF base class
    and overrides the fit_model method to provide the specific implementation of the LMAFIT algorithm.

    Attributes:
        rank (int): The rank of the factorization.
        ratings (pd.DataFrame): The user-item ratings matrix.
        U (np.ndarray): User feature matrix.
        V (np.ndarray): Item feature matrix.
        pred (pd.DataFrame): Predicted ratings matrix.
        error (float): The error of the predicted ratings.

    Methods:
        fit_model(ratings=None, init=None):
            Fits the LMAFIT model to the provided ratings matrix.
    """

    def fit_model(self, ratings=None, init=None):
        X = self.ratings if ratings is None else ratings
        self.ratings = X
        m, n = X.shape
        known_elements = np.where(~np.isnan(X.values))
        list_of_known_elements = zip(*known_elements)
        data = [X.values[coordinate] for coordinate in list_of_known_elements]
        self.U, self.V, opts = lmafit.lmafit_mc_adp(m, n, self.rank, known_elements, data, opts=init)        
        self.pred = pd.DataFrame(self.U.dot(self.V), index=X.index, columns=X.columns)
        self.error = ma.power(ma.masked_invalid(X-self.pred),2).sum()
        return self.pred, self.error
