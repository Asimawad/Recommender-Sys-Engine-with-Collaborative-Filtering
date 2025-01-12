import time
import numpy as np
from src.Latent_Factor_Updates import (
    Update_movie_factors_with_features,
    Update_user_biases,
    Update_user_factors,
    Update_movie_biases,
    update_feature_factors,
    Update_movie_factors,
    calc_metrics,
)
from src.Helper_Functions import (
    plot_likelihood,
    plot_rmse,
    setup_logging,
    setup_experiment_folder,
    save_model,
    Load_training_data,
    Load_idx_maps,
    Load_test_data,
)

# === Class Definitions ===

class RecommendationModel:
    def __init__(self,n_users,n_movies,num_features,K_factors=32,lambda_reg=2,gamma=1,taw=100,n_epochs=30,jitter=1e-8):
                self.n_users = n_users
                self.n_movies = n_movies
                self.num_features = num_features
                self.K_factors = K_factors
                self.lambda_reg = lambda_reg
                self.gamma = gamma
                self.taw = taw
                self.n_epochs = n_epochs
                self.jitter = jitter

        # Initialize latent factors and biases
                std = np.sqrt(K_factors)
                self.users_factors = np.random.normal(0, 1 / std, (n_users, K_factors))
                self.movies_factors = np.random.normal(0, 1 / std, (n_movies, K_factors))
                self.genre_factors = np.random.normal(0, 1 / std, (num_features, K_factors))
                self.user_bias = np.random.randn(n_users)
                self.item_bias = np.random.randn(n_movies)

                self.train_loss = []
                self.valid_loss = []
                self.train_rmse = []
                self.valid_rmse = []

    def train(self, users_train, movies_train, users_train_idxes, movies_train_idxes, movies_genres_array, users_test, movies_test, users_test_idxes, movies_test_idxes, logger):
                print("Training Started ...")
                for epoch in range(self.n_epochs):
                        epoch_start_time = time.time()

                        # User updates
                        user_update_time = self._update_users(users_train, users_train_idxes)

                        # Movie updates
                        movie_update_time = self._update_movies(movies_train, movies_train_idxes, movies_genres_array)

                        # Calculate metrics
                        training_loss, train_rmse = self._calculate_metrics(users_train, users_train_idxes)
                        validation_loss, valid_rmse = self._calculate_metrics(users_test, users_test_idxes)

                        self.train_loss.append(training_loss)
                        self.valid_loss.append(validation_loss)
                        self.train_rmse.append(train_rmse)
                        self.valid_rmse.append(valid_rmse)

                        epoch_time = time.time() - epoch_start_time

                        logger.info(f"Epoch {epoch + 1}/{self.n_epochs} : Train RMSE = {train_rmse:.4f} -- Valid RMSE = {valid_rmse:.4f} | Train Loss = {training_loss / 1e6:.1f}M -- Validation Loss = {validation_loss / 1e6:.1f}M")
                        logger.info(f" - User Updates Time: {user_update_time:.2f}s")
                        logger.info(f" - Movie Updates Time: {movie_update_time:.2f}s")
                        logger.info(f" - Total Epoch Time: {epoch_time:.2f}s\n")

    def _update_users(self, users_train, users_train_idxes):
        start_time = time.time()
        for current_user_idx, (movie_indices, rating) in zip(users_train_idxes, users_train):
            Update_user_biases(current_user_idx, movie_indices, rating, self.item_bias, self.user_bias, self.lambda_reg, self.gamma, self.users_factors, self.movies_factors)
            Update_user_factors(current_user_idx, movie_indices, rating, self.movies_factors, self.users_factors, self.user_bias, self.item_bias, self.lambda_reg, self.taw, self.K_factors)
        return time.time() - start_time

    def _update_movies(self, movies_train, movies_train_idxes, movies_genres_array):
        start_time = time.time()
        for current_movie_idx, (user_indices, rating) in zip(movies_train_idxes, movies_train):
            Update_movie_biases(current_movie_idx, user_indices, rating, self.user_bias, self.item_bias, self.lambda_reg, self.gamma, self.users_factors, self.movies_factors)
            Update_movie_factors( current_movie_idx,user_indices, rating , self.users_factors, self.movies_factors,self.user_bias, self.item_bias, self.lambda_reg, self.taw, self.K_factors, )
        return time.time() - start_time

    def _calculate_metrics(self, users, users_idxes):
        return calc_metrics(users, users_idxes, self.users_factors, self.movies_factors, self.user_bias, self.item_bias, self.lambda_reg, self.gamma, self.taw)

    def save_results(self, experiment_folder):
        save_model(experiment_folder, self.movies_factors, self.users_factors, self.user_bias, self.item_bias, self.genre_factors)

    def plot_results(self, n_epochs, experiment_folder):
        plot_likelihood(x_axis=n_epochs, y_axis=[self.train_loss, self.valid_loss], experiment_folder=experiment_folder)
        plot_rmse(x_axis=n_epochs, y_axis=[self.train_rmse, self.valid_rmse], experiment_folder=experiment_folder)

# === Main Script === 
if __name__ == "__main__":
    Datasets = ["ml-100k","ml-25m","ml-32m"]
    dataset = Datasets[0]
    experiment_name = "B_U_V"  # Customize this name
    experiment_folder = setup_experiment_folder(dataset,experiment_name)
    logger = setup_logging(experiment_folder)

    source_data_folder = f"Training_data/{dataset}"
    
    users_train, movies_train, movies_train_idxes, users_train_idxes, movies_genres_array = Load_training_data(source_data_folder)
    users_test, movies_test, users_test_idxes, movies_test_idxes = Load_test_data(source_data_folder)
    user_idx_map, movie_idx_map, idx_to_user, idx_to_movie, genre_to_idx, specific_indices = Load_idx_maps(source_data_folder)

    n_users = len(user_idx_map)
    n_movies = len(movie_idx_map)
    num_features = movies_genres_array.shape[-1]

    # Best Parameters after grid Search->
    # K_factors = 32 ;    EPOCHS = 30 ;  lambda_reg = 0.5 ; gamma = 0.5 ; taw =  10 ; std = np.sqrt(K_factors)
    K_factors = 16 ;    EPOCHS = 30 ;  lambda_reg = 0.1 ; gamma = 0.05 ; taw =  100 ; std = np.sqrt(K_factors)
    # Initialize and train the model
    model = RecommendationModel(n_users = n_users, n_movies = n_movies, num_features = num_features,K_factors=K_factors,lambda_reg= lambda_reg,gamma=gamma,taw=taw,n_epochs=EPOCHS,jitter=1e-8)
    model.train(users_train, movies_train, users_train_idxes, movies_train_idxes, movies_genres_array, users_test, movies_test, users_test_idxes, movies_test_idxes, logger)

    # Save and plot results
    model.save_results(experiment_folder)
    model.plot_results(model.n_epochs, experiment_folder)
