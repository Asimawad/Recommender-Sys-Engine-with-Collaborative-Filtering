import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
# plotting the loss and the RMSE curves
def plot_likelihood(x_axis, y_axis, experiment_folder = "./" , plot_name = "Negative Log Likelihood Curves"):
  plt.figure(figsize=(10, 6))
  plt.subplot(1,2,1)
  plt.plot(range(x_axis) , y_axis[0])
  plt.title('Training Likelihood Curve')
  plt.xlabel('Iterations')
  plt.ylabel('loss')
  plt.subplot(1,2,2)
  plt.plot(range(x_axis) , y_axis[1] , c ='r')
  plt.title('Validation Likelihood Curve')
  plt.xlabel('Iterations')
  plt.ylabel('loss')
  plot_filename = os.path.join(experiment_folder, f'_{plot_name}_experiment_plot.pdf')    
  print(f"Plot saved as {plot_name} Curves")
  plt.savefig(plot_filename)
  plt.show()
  plt.close() 

# Movie degrees
def plot_rmse(x_axis,y_axis,experiment_folder = './',plot_name = "RMSE Curves"):
  plt.figure(figsize=(10, 6))
  plt.plot(range(x_axis) , y_axis[0])
  plt.plot(range(x_axis) , y_axis[1] , c ='r')
  plt.title('Training & Validation RMSE Curves')
  plt.xlabel('Iterations')
  plt.ylabel('RMSE')
  plot_filename = os.path.join(experiment_folder, f'_{plot_name}_experiment_plot.pdf')    
  print(f"Plot saved as {plot_name}  Curves")
  plt.savefig(plot_filename)
  plt.show()
  plt.close() 

def setup_logging(experiment_folder):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create a file handler to log to a file in the experiment folder
    log_filename = os.path.join(experiment_folder, 'experiment_log.txt')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Create a stream handler to print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_experiment_folder(dataset, experiment_name):
    # Create a directory for the experiment if it doesn't exist
    experiment_folder = f"./Experiments_{dataset}/{experiment_name}/"
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    return experiment_folder

def get_possible_movie_indices(query = "scooby doo" ,movies_df = None):
  movies = movies_df['title'].values.tolist()
  movies_df['title'] = movies_df['title'].str.lower()
  def check_movie(value, movies):
    matched_movies = []
    for movie in movies:
      if re.search(value, movie):
        matched_movies.append(movie)
    return matched_movies
  title_to_id_moviesmap = {a:b for (a, b) in zip(movies_df['title'], movies_df['movieId'])}
  id_to_title_moviesmap = {a:b for (a, b) in zip(movies_df['movieId'], movies_df['title'])}
  def get_possible_movie_indices(search_movie, movies_map = title_to_id_moviesmap):
    possible_movies = movies_map.keys()
    for val in search_movie.split():
      possible_movies = check_movie(val, possible_movies)

    return possible_movies
  
  return get_possible_movie_indices(query, title_to_id_moviesmap)


def create_fake_user(list_of_favourite_movies,item_bias,movies_factors,K_factors,lambda_reg,gamma,taw):
    movie_indices , rating = list_of_favourite_movies
    movie_bias = item_bias[movie_indices]
    movie_factor = movies_factors[movie_indices]
    fake_user = np.zeros((1 ,K_factors))
    b_fake_user = lambda_reg *( rating -  movie_bias )/(lambda_reg +gamma)

    for _ in range(5):
        # update fake_user

        sum_VnVnT = lambda_reg * np.outer(movie_factor,movie_factor)
        reg_term = taw*np.eye(K_factors)
        sum_rmn_Vn = lambda_reg * (movie_factor * (rating - b_fake_user - movie_bias  ))
        fake_user = np.linalg.solve(sum_VnVnT + reg_term, sum_rmn_Vn)
        # update b_fake_user again
        rhat = fake_user@movie_factor
        b_fake_user = lambda_reg *(rating - rhat- movie_bias )/(lambda_reg +gamma)
    return fake_user 


def save_model(source_data_folder,movies_factors,users_factors,user_bias,item_bias, feature_vectors = np.zeros((19,30))):
  if not os.path.exists(f'{source_data_folder}model_params'):
        os.makedirs(f'{source_data_folder}model_params')
  np.save(f'{source_data_folder}model_params/user_bias.npy', user_bias,allow_pickle=True)
  np.save(f'{source_data_folder}model_params/item_bias.npy', item_bias,allow_pickle=True)
  np.save(f'{source_data_folder}model_params/users_factors.npy', users_factors,allow_pickle=True)
  np.save(f'{source_data_folder}model_params/movies_factors.npy', movies_factors,allow_pickle=True)
  np.save(f'{source_data_folder}model_params/feature_vectors.npy', feature_vectors,allow_pickle=True)

def load_model(source_data_folder):
    #   for testing the model
    user_bias       = np.load(f'{source_data_folder}model_params/user_bias.npy',allow_pickle=True)
    item_bias       = np.load(f'{source_data_folder}model_params/item_bias.npy',allow_pickle=True)
    users_factors   = np.load(f'{source_data_folder}model_params/users_factors.npy',allow_pickle=True)
    movies_factors  = np.load(f'{source_data_folder}model_params/movies_factors.npy',allow_pickle=True)
    feature_vectors = np.load(f'{source_data_folder}model_params/feature_vectors.npy',allow_pickle=True)
    return movies_factors,users_factors,user_bias,item_bias,feature_vectors
def Load_idx_maps(load_dir):
    with open(os.path.join(load_dir, 'user_idx_map.pkl'), 'rb') as f:
        user_idx_map = pickle.load(f)

    with open(os.path.join(load_dir, 'movie_idx_map.pkl'), 'rb') as f:
        movie_idx_map = pickle.load(f)

    with open(os.path.join(load_dir, 'idx_to_user.pkl'), 'rb') as f:
        idx_to_user = pickle.load(f)

    with open(os.path.join(load_dir, 'idx_to_movie.pkl'), 'rb') as f:
        idx_to_movie = pickle.load(f)

    with open(os.path.join(load_dir, 'genre_to_idx.pkl'), 'rb') as f:
        genre_to_idx = pickle.load(f)

    with open(os.path.join(load_dir, 'specific_indices.pkl'), 'rb') as f:
        specific_indices = pickle.load(f)

    return user_idx_map, movie_idx_map, idx_to_user, idx_to_movie, genre_to_idx, specific_indices

def Load_training_data(source_data_folder):
    users_train        = np.load(os.path.join(source_data_folder,'users_train.npy'),allow_pickle=True)
    movies_train       = np.load(os.path.join(source_data_folder,'movies_train.npy'),allow_pickle=True)
    movies_train_idxes = np.load(os.path.join(source_data_folder,'movies_train_idxes.npy'),allow_pickle=True)
    users_train_idxes  = np.load(os.path.join(source_data_folder,'users_train_idxes.npy'), allow_pickle=True)
    movies_genres_array= np.load(os.path.join(source_data_folder,'movies_genres_array.npy'), allow_pickle=True)
    return users_train,movies_train,movies_train_idxes,users_train_idxes,movies_genres_array

def Load_test_data(source_data_folder):
    users_test         = np.load(os.path.join(source_data_folder,'users_test.npy' ),allow_pickle=True)
    movies_test        = np.load(f'./{source_data_folder}/movies_test.npy', allow_pickle=True)
    users_test_idxes   = np.load(f'./{source_data_folder}/users_test_idxes.npy', allow_pickle=True)
    movies_test_idxes  = np.load(f'./{source_data_folder}/movies_test_idxes.npy', allow_pickle=True)
    return users_test,movies_test,users_test_idxes,movies_test_idxes 

# movieId,title,genres
