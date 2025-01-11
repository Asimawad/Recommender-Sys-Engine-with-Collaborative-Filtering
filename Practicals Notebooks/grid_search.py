import time
import numpy as np
from src.Latent_Factor_Updates import Update_movie_factors_with_features, Update_user_biases, Update_user_factors, Update_movie_biases, update_feature_factors, calc_metrics
from src.Helper_Functions import plot_likelihood, plot_rmse, setup_logging, setup_experiment_folder, save_model, Load_training_data, Load_idx_maps, Load_test_data, get_possible_movie_indices
from Main import RecommendationModel

#  saving plots and training logs
experiment_name = "grid_search"  # Customize this name
experiment_folder = setup_experiment_folder('ml-32m',experiment_name)
logger = setup_logging(experiment_folder)

# Define the hyperparameter grid to search
param_grid = {
    # 'lambda_reg': [0.1, 0.5, 1.0, 5.0],  # Regularization parameter
    'lambda_reg': [0.5],  # Regularization parameter
    # 'gamma': [0.01, 0.1, 0.5],           # Learning rate for users' factors
    'gamma': [0.5],           # Learning rate for users' factors
    # 'taw': [1, 10, 100],                 # Weighting factor for genre features
    'taw': [10],                 # Weighting factor for genre features
    'K_factors':    [8,16,25,32,40,50,60]        # Number of latent factors [8,16, 25,32,40, 50] 
}

# source_data_folder = f"Training_data/{Dataset}"
source_data_folder = "Training_data/ml-32m"

users_train, movies_train, movies_train_idxes, users_train_idxes, movies_genres_array = Load_training_data(source_data_folder)
users_test, movies_test, users_test_idxes, movies_test_idxes = Load_test_data(source_data_folder)

user_idx_map, movie_idx_map, idx_to_user, idx_to_movie, genre_to_idx, specific_indices = Load_idx_maps(source_data_folder)

# Function to perform grid search
def grid_search(param_grid):
    best_params = None
    best_valid_rmse = float('inf')
    best_train_rmse = float('inf')
    best_train_loss = float('inf')
    best_valid_loss = float('inf')

    # Iterate over all combinations of hyperparameters
    for lambda_reg in param_grid['lambda_reg']:
        for gamma in param_grid['gamma']:
            for taw in param_grid['taw']:
                for K_factors in param_grid['K_factors']:
                    logger.info(f"Training with lambda_reg={lambda_reg}, gamma={gamma}, taw={taw}, K_factors={K_factors}")
                    
                    # Initialize all variables inside the grid search loop to reset them for each combination
                    n_users = len(user_idx_map)
                    n_movies = len(movie_idx_map)
                    jitter = 1e-8
                    num_features = movies_genres_array.shape[-1]

                    # Re-initialize Latent Factors
                    users_factors = np.random.normal(loc=0, scale=1/np.sqrt(K_factors), size=(n_users, K_factors))
                    movies_factors = np.random.normal(loc=0, scale=1/np.sqrt(K_factors), size=(n_movies, K_factors))
                    genre_factors = np.random.normal(loc=0, scale=1/np.sqrt(K_factors), size=(num_features, K_factors))

                    # Initialize Bias Terms
                    user_bias = np.random.randn(n_users)
                    item_bias = np.random.randn(n_movies)

                    train_loss = []
                    valid_loss = []
                    train_rmse = []
                    valid_rmse = []

                    # Start training loop for grid search
                    for EPOCH in range(10):  # We can keep it fixed at 20 epochs for the grid search
                        epoch_start_time = time.time()

                        # Perform user updates
                        for current_user_idx, (movie_indices, rating) in zip(users_train_idxes, users_train):
                            Update_user_biases(current_user_idx, movie_indices, rating, item_bias, user_bias, lambda_reg, gamma, users_factors, movies_factors)
                            Update_user_factors(current_user_idx, movie_indices, rating, movies_factors, users_factors, user_bias, item_bias, lambda_reg, taw, K_factors)

                        # Perform movie updates
                        for current_movie_idx, (user_indices, rating) in zip(movies_train_idxes, movies_train):
                            Update_movie_biases(current_movie_idx, user_indices, rating, user_bias, item_bias, lambda_reg, gamma, users_factors, movies_factors)
                            Update_movie_factors_with_features(current_movie_idx, user_indices, rating, users_factors, movies_factors, user_bias, item_bias, lambda_reg, taw, K_factors, movies_genres_array, genre_factors, jitter)

                        # Perform feature updates
                        update_feature_factors(num_features, movies_genres_array, movies_factors, genre_factors, taw)

                        # Calculate metrics
                        train_loss_epoch, trmse = calc_metrics(users_train, users_train_idxes, users_factors, movies_factors, user_bias, item_bias, lambda_reg, gamma, taw)
                        valid_loss_epoch, vrmse = calc_metrics(users_test, users_test_idxes, users_factors, movies_factors, user_bias, item_bias, lambda_reg, gamma, taw)

                        train_loss.append(train_loss_epoch)
                        valid_loss.append(valid_loss_epoch)
                        train_rmse.append(trmse)
                        valid_rmse.append(vrmse)

                        # Check if this combination is the best
                        if vrmse < best_valid_rmse:
                            best_valid_rmse = vrmse
                            best_train_rmse = trmse
                            best_train_loss = train_loss_epoch
                            best_valid_loss = valid_loss_epoch
                            best_params = {
                                'lambda_reg': lambda_reg,
                                'gamma': gamma,
                                'taw': taw,
                                'K_factors': K_factors
                            }

                        # Log the results
                        logger.info(f"Epoch {EPOCH + 1}, Valid RMSE: {vrmse:.4f}")

                    # Print best results for this set of hyperparameters
                    logger.info(f"Best Params: {best_params}")
                    logger.info(f"Best Train RMSE: {best_train_rmse:.4f}, Best Valid RMSE: {best_valid_rmse:.4f}")
    
    return best_params, best_train_rmse, best_valid_rmse, best_train_loss, best_valid_loss

# Execute grid search
best_params, best_train_rmse, best_valid_rmse, best_train_loss, best_valid_loss = grid_search(param_grid)

# Print best results
logger.info("Best Hyperparameters:")
logger.info(f"lambda_reg: {best_params['lambda_reg']}, gamma: {best_params['gamma']}, taw: {best_params['taw']}, K_factors: {best_params['K_factors']}")
logger.info(f"Best Train RMSE: {best_train_rmse:.4f}")
logger.info(f"Best Valid RMSE: {best_valid_rmse:.4f}")
