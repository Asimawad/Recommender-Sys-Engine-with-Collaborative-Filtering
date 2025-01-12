import time
import numpy as np
from src.Latent_Factor_Updates import  Update_movie_factors_with_features,Update_user_biases,Update_user_factors,Update_movie_biases,Update_movie_factors,update_feature_factors,calc_metrics
from src.Helper_Functions import plot_likelihood, plot_rmse  ,setup_logging, setup_experiment_folder ,save_model,Load_training_data,Load_idx_maps,Load_test_data,get_possible_movie_indices

#  saving plots and training logs
dataset = "ml-25m"
experiment_name = "BIAS_U_V"  # Customize this name
experiment_folder = setup_experiment_folder(dataset,experiment_name)
logger = setup_logging(experiment_folder)


# by running Data_preprocessing, this data will be created in the data folder
data_folder = f"Training_data/{dataset}"
users_train,movies_train,movies_train_idxes,users_train_idxes,movies_genres_array = Load_training_data(data_folder)
users_test,movies_test,users_test_idxes,movies_test_idxes  = Load_test_data(data_folder)
user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie,genre_to_idx,_=  Load_idx_maps(data_folder)
print("Training data is loaded")

#  users + bias update only
n_users = len(user_idx_map) ; n_movies = len(movie_idx_map)
K_factors = 25 ; n_Epochs = 20 ;  lambda_reg = 1 ; gamma = 0.1 ; taw =  10 ; std = np.sqrt(K_factors)
num_features = movies_genres_array.shape[-1]
# Initializing Latent Factors
users_factors  = np.random.normal(loc=0, scale = 1/std, size=(n_users ,K_factors))
movies_factors = np.random.normal(loc=0, scale = 1/std, size=(n_movies,K_factors))
#2 - Initialize Bias Terms:
user_bias = np.random.randn(n_users)
item_bias = np.random.randn(n_movies)

train_loss = []
valid_loss = []

train_rmse = []
valid_rmse = []
# MAIN LOOP - - - - > in this code, users without data will still be random, not updated to zeros like biases, should the biases be randomized? idk
print("Training Started ...")
# Tracking the time taken for different updates
for EPOCH in range(n_Epochs):
    epoch_start_time = time.time()  # Start time for the epoch

    # Timing for user updates
    user_update_time = 0
    start_time = time.time()  # Start timing the user updates

    # Perform the updates on users
    for  current_user_idx, (movie_indices, rating) in zip(users_train_idxes, users_train):
      Update_user_biases(current_user_idx, movie_indices, rating , item_bias, user_bias, lambda_reg, gamma, users_factors, movies_factors)
      Update_user_factors(current_user_idx, movie_indices, rating,  movies_factors, users_factors, user_bias, item_bias, lambda_reg, taw, K_factors)
    # Accumulate the time taken for these updates
    user_update_time += time.time() - start_time

    # Timing for movie updates
    movie_update_time = 0
    start_time = time.time()  # Start timing the movie updates

    # Perform the updates onscikit-learn movies
    for  current_movie_idx,(user_indices, rating) in zip(movies_train_idxes,movies_train):

      Update_movie_biases( current_movie_idx,user_indices, rating , user_bias, item_bias, lambda_reg, gamma, users_factors, movies_factors)
      Update_movie_factors( current_movie_idx,user_indices, rating , users_factors, movies_factors, user_bias, item_bias, lambda_reg, taw, K_factors)
    movie_update_time += time.time() - start_time


    #  calculate the likelihood 
    
    training_loss,trmse = calc_metrics(users_train,users_train_idxes ,users_factors,movies_factors, user_bias, item_bias, lambda_reg, gamma,taw)
    validation_loss,vrmse = calc_metrics(users_test,users_test_idxes ,users_factors,movies_factors,user_bias,item_bias,lambda_reg,gamma,taw)

    train_loss.append(training_loss)
    valid_loss.append(validation_loss)

    train_rmse.append(trmse)
    valid_rmse.append(vrmse)

    # End time for the epoch
    epoch_time = time.time() - epoch_start_time
  # Now log information & Print the time taken for each part

    logger.info(f"Epoch {EPOCH + 1}/{n_Epochs} :Train RMSE = {trmse:.4f} --Valid RMSE = {vrmse:.4f} |Train loss = {training_loss/1000000:.1f} million -- validation likelihood {validation_loss/1000000:.1f} million")
    logger.info(f" - User Updates  Time: {user_update_time:.2f} seconds")
    logger.info(f" - Movie Updates Time: {movie_update_time:.2f} seconds")
    logger.info(f" - Total Epoch   Time: {epoch_time:.2f} seconds\n")

save_model(experiment_folder,movies_factors,users_factors,user_bias,item_bias)

#  PLOTTING THE RESULTS
plot_likelihood(x_axis= n_Epochs , y_axis= [train_loss,valid_loss],experiment_folder=experiment_folder )
plot_rmse(x_axis= n_Epochs , y_axis= [train_rmse,valid_rmse] ,experiment_folder=experiment_folder )
