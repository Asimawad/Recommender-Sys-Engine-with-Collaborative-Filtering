import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from Data_processing import MovieLookups
from Helper_Functions import (
    plot_likelihood,
    plot_rmse,
    setup_logging,
    setup_experiment_folder,
    save_model,
)


# === Utility Functions ===

# Positive Log-Likelihood
def likelihood(users_list, users_biases, movies_biases, lambda_reg, gamma_reg):
    cost = 0
    for uidx, user_fav_movies in enumerate(users_list):
        for movie in user_fav_movies:
            if movie != ():  # movie = (midx, rating)
                midx = movie[0]
                rating = movie[1]
                cost += (rating - (users_biases[uidx] + movies_biases[midx])) ** 2
    cost *= lambda_reg / 2
    cost += gamma_reg / 2 * sum([x ** 2 for x in users_biases])
    cost += gamma_reg / 2 * sum([x ** 2 for x in movies_biases])
    return cost

# Root Mean Squared Error (RMSE)
def RMSError(users_list, users_biases, movies_biases):
    error = 0
    n = 0
    for uidx, favourite_movies in enumerate(users_list):
        for movie in favourite_movies:
            if movie != ():
                midx = movie[0]
                rate = movie[1]
                error += (rate - (movies_biases[midx] + users_biases[uidx])) ** 2
                n += 1
    error = error / n
    return np.sqrt(error), n

# Update User Biases
def update_user_biases(current_user_idx, list_of_favourite_movies, movies_biases, users_biases, lambda_reg, gamma):
    sum_bias = 0
    n = 0
    for midx, rate in list_of_favourite_movies:
        sum_bias += (rate - movies_biases[midx])
        n += 1
    if n != 0:  # Update only if user has data
        users_biases[current_user_idx] = (lambda_reg * sum_bias) / ((lambda_reg * n) + gamma)

# Update Movie Biases
def update_movie_biases(current_movie_idx, list_of_fans, users_biases, movies_biases, lambda_reg, gamma):
    sum_bias = 0
    n = 0
    for uidx, rating in list_of_fans:
        sum_bias += (rating - users_biases[uidx])
        n += 1
    if n != 0:  # Update only if movie has data
        movies_biases[current_movie_idx] = (lambda_reg * sum_bias) / ((lambda_reg * n) + gamma)

# === Main Script ===
if __name__ == "__main__":
    experiment_name = "Bias_only_model"  # Customize this name
    experiment_folder = setup_experiment_folder(experiment_name)
    logger = setup_logging(experiment_folder)

    movies = pd.read_csv('./Data/movies.csv')
    ratings = pd.read_csv('./Data/ratings.csv')


    # Initialize the MovieLookups class
    ratings_lookups = MovieLookups(ratings)

    All_users, All_movies = ratings_lookups.create_lookups()
    users_train_data, users_val_data = ratings_lookups.Users_split()
    movies_train_data, movies_val_data = ratings_lookups.Movies_split()

    # Parameters
    n_users = len(users_train_data)
    n_movies = len(movies_train_data)
    n_Epochs = 10
    lambda_reg = 1
    gamma = 0.01

    # Initialize Bias Terms
    users_biases = np.random.randn(n_users)
    movies_biases = np.random.randn(n_movies)

    train_loss = []
    valid_loss = []

    train_rmse = []
    valid_rmse = []

    # Training Loop
    for epoch in range(n_Epochs):
        # Update User Biases movie_indices, rating)
        for current_user_idx, list_of_favourite_movies in enumerate(users_train_data):
            update_user_biases(current_user_idx, list_of_favourite_movies, movies_biases, users_biases, lambda_reg, gamma)

        # Update Movie Biases 
        for current_movie_idx, list_of_fans in enumerate(movies_train_data):
            update_movie_biases(current_movie_idx, list_of_fans, users_biases, movies_biases, lambda_reg, gamma)

        # Calculate Loss
        training_loss = likelihood(users_train_data, users_biases, movies_biases, lambda_reg, gamma)
        validation_loss = likelihood(users_val_data, users_biases, movies_biases, lambda_reg, gamma)

        train_loss.append(training_loss)
        valid_loss.append(validation_loss)

        # Calculate RMSE
        trmse, _ = RMSError(users_train_data, users_biases, movies_biases)
        vrmse, _ = RMSError(users_val_data, users_biases, movies_biases)

        train_rmse.append(trmse)
        valid_rmse.append(vrmse)

        logger.info(f"Epoch {epoch + 1}: Train RMSE = {trmse:.4f}, Valid RMSE = {vrmse:.4f} | Train Loss = {training_loss:.4f}, Validation Loss = {validation_loss:.4f}")

    # Plot Results
    plot_likelihood(x_axis=n_Epochs, y_axis=[train_loss, valid_loss],experiment_folder=experiment_folder)
    plot_rmse(x_axis=n_Epochs, y_axis=[train_rmse, valid_rmse],experiment_folder=experiment_folder)
