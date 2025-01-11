import numpy as np
from numba import jit

@jit(nopython = True)
def get_UEROR(uidx,movie_indices, rating,movies_factors,users_factors, users_biases,movies_biases):
        # Predicted rating
      preds = (movies_factors[movie_indices] @ users_factors[uidx]) + users_biases[uidx] + movies_biases[movie_indices]

      # Compute squared errors
      squared_errors = (rating - preds) ** 2
      return np.sum(squared_errors)

# @jit(nopython = True)
def calc_metrics(users_list, users_train_idxes,users_factors, movies_factors, users_biases, movies_biases, lambda_reg, gamma, tau):
    total_squared_error = 0
    total_count = 0

    for i,uidx in enumerate(users_train_idxes):
      movie_indices, rating = users_list[i]

      total_squared_error +=  get_UEROR(uidx,movie_indices, rating,movies_factors,users_factors, users_biases,movies_biases)
      total_count += len(rating)

    rmse = np.sqrt(total_squared_error / total_count)
    cost = 0.5 * total_squared_error
    # Regularization for biases
    cost += (gamma / 2) * (np.sum(users_biases ** 2) + np.sum(movies_biases ** 2))
    # Regularization for latent factors
    cost += (tau / 2) * (np.sum(users_factors ** 2) + np.sum(movies_factors ** 2))
    return cost, rmse


@jit(nopython = True)
def Update_user_biases(  current_user_idx,  movie_indices, rating,movies_biases,users_biases,lambda_reg,gamma,users_factors = None,movies_factors = None):
    #  Update the user Biases

      n = len(rating)
      if n == 0:
        return  # No rating to update

        # for Biases update formula
      r_hat = movies_factors[movie_indices] @ users_factors[current_user_idx] + movies_biases[movie_indices]
      # Now accumulate summ
      numerator =  lambda_reg * (rating  - r_hat) / ((lambda_reg * n) + gamma)
      sum_bias = np.sum(numerator)
      users_biases[current_user_idx] = sum_bias
      
# THE Movies BIAS UPDATES LOOP - one movie only
@jit(nopython = True)
def Update_movie_biases(current_movie_idx,user_indices, rating ,users_biases, movies_biases,lambda_reg , gamma, users_factors= None,movies_factors=None ):
    #Update the movie Biases
    n = len(rating)
    if n == 0 :
      return

    r_hat = (users_factors[user_indices] @ movies_factors[current_movie_idx]) + users_biases[user_indices]
    numerator = lambda_reg* (rating - r_hat)/ ((lambda_reg * n) + gamma)
    sum_bias = np.sum(numerator)
    movies_biases[current_movie_idx] = sum_bias

# THE USERS FACTORS UPDATES LOOP - one USER only
@jit(nopython = True)
def Update_user_factors(current_user_idx, movie_indices, rating ,movies_factors,users_factors,user_bias,item_bias,lambda_reg,taw,K_factors):
    # Update user factors with matrix calculations

  #  Update the user factors
    n = len(rating)
    if n == 0:
        return  # No rating to update

    Vn = movies_factors[movie_indices]
    errors      =  rating - user_bias[current_user_idx] - item_bias[movie_indices]
    errors = errors[:, np.newaxis] # Shape: (n_rating, 1)

    # Compute sums
    sum_VnVnT = Vn.T @ Vn  # Shape: (K_factors, K_factors)
    sum_rmn_Vn = (Vn * errors).sum(axis=0)
    reg_matrix = taw * np.eye(K_factors)
    matrix_to_invert = (lambda_reg * sum_VnVnT) + reg_matrix
    rhs = lambda_reg * sum_rmn_Vn

    # Update user factors
    users_factors[current_user_idx] = np.linalg.solve(matrix_to_invert, rhs)

@jit(nopython = True)
def Update_movie_factors( current_movie_idx, user_indices, rating, users_factors, movies_factors, user_bias, item_bias, lambda_reg, taw,K_factors):
    # current_movie_idx , user_indices, rating = list_of_fans
    n = len(rating)
    if n == 0:
        return  # No rating to update

    Um = users_factors[user_indices]  # Shape: (n_rating, K_factors)
    errors = rating - user_bias[user_indices] - item_bias[current_movie_idx]
    errors = errors[:, np.newaxis] # Shape: (n_rating, 1)

    # Compute sums
    sum_UmUmT = Um.T @ Um  # Shape: (K_factors, K_factors)
    sum_rmn_Um = (Um * errors).sum(axis=0)  # Shape: (K_factors,)

    reg_matrix = taw * np.eye(K_factors)
    matrix_to_invert = (lambda_reg * sum_UmUmT )+ reg_matrix
    rhs = lambda_reg * sum_rmn_Um

    # Update movie factors
    movies_factors[current_movie_idx] = np.linalg.solve(matrix_to_invert, rhs)


@jit(nopython=True)
def Update_movie_factors_with_features(
    current_movie_idx, user_indices, rating, users_factors, movies_factors,
    user_bias, item_bias, lambda_reg, taw, K_factors, movies_genres_array, genre_factors, jitter):
    n = len(rating)
    if n == 0:
        return  # No rating to update

    Um = users_factors[user_indices]  # Shape: (n_rating, K_factors)
    errors = rating - user_bias[user_indices] - item_bias[current_movie_idx]
    errors = errors[:, np.newaxis]  # Shape: (n_rating, 1)

    # Compute sum_UmUmT using np.dot
    sum_UmUmT = Um.T @ Um  # Equivalent to np.einsum('ij,il->jl', Um, Um)

    # Compute the right-hand side vector for errors using np.dot
    sum_UmErrors = Um.T @ errors[:, 0]  # Equivalent to np.einsum('ij,i->j', Um, errors[:, 0])

    # Ensure compatible data types for genre computation
    genres_float = movies_genres_array[current_movie_idx].astype(np.float64)
    genre_contribution = np.dot(genres_float, genre_factors)
    genre_contribution *= taw / np.sqrt(np.sum(genres_float) + jitter)

    # Final update computation
    left = lambda_reg * sum_UmUmT + taw * np.eye(K_factors)
    total_right = lambda_reg * sum_UmErrors + genre_contribution
    movies_factors[current_movie_idx] = np.linalg.solve(left, total_right)

def update_feature_factors(num_features, movies_genres_array,V,F, tau):
    for i in range(num_features):
      indices = np.where(movies_genres_array[:, i] == 1) # get movies with ith feature
      feature_sums = np.sum(movies_genres_array[indices], axis=1) # get number of features for movies containing ith feature
      features_except_i = np.delete(movies_genres_array, i, axis=1)[indices] # get other features except ith feature for movies with ith feature
      F_except_i = np.delete(F, i, axis=0) # get feature embeddings for other features except ith feature
      # calculate feature update
      total_right = np.sum(V[indices] * 1/np.sqrt(feature_sums)[..., None] - 1/feature_sums[..., None] * np.einsum('ij, jk -> ik', features_except_i, F_except_i), axis=0)         
      F[i] = total_right/(np.sum(1/np.sqrt(feature_sums)) - 1)