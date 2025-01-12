import matplotlib.pylab as plt
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
from Main import RecommendationModel

# === Main Script === 
if __name__ == "__main__":
    Dataset = "ml-25m"
    experiment_name = "PARAM_K"  # Customize this name
    experiment_folder = setup_experiment_folder(Dataset,experiment_name)
    logger = setup_logging(experiment_folder)

    # source_data_folder = f"Training_data/{Dataset}"
    source_data_folder = f"Training_data/{Dataset}"
    
    users_train, movies_train, movies_train_idxes, users_train_idxes, movies_genres_array = Load_training_data(source_data_folder)
    users_test, movies_test, users_test_idxes, movies_test_idxes = Load_test_data(source_data_folder)

    user_idx_map, movie_idx_map, idx_to_user, idx_to_movie, genre_to_idx, specific_indices = Load_idx_maps(source_data_folder)
    EPOCHS = 10
    n_users = len(user_idx_map)
    n_movies = len(movie_idx_map)
    num_features = movies_genres_array.shape[-1]
    k_values = [8, 16,25, 32, 40]
    final_valid_rmses = []

    for k in k_values:
        logger.info(f"Training model with K_factors={k}")
        model = RecommendationModel(
            n_users=n_users,
            n_movies=n_movies,
            num_features=num_features,
            K_factors=k,
            lambda_reg=0.5,   # Example from your “best” hyperparams
            gamma=0.5,
            taw=10,
            n_epochs=20,      # or however many epochs you want
            jitter=1e-8
        )
        
        # Train the model
        model.train(
            users_train, movies_train,
            users_train_idxes, movies_train_idxes,
            movies_genres_array,
            users_test, movies_test,
            users_test_idxes, movies_test_idxes,
            logger
        )
        
        # Store final validation RMSE (last epoch) for this K_factors
        final_rmse = model.valid_rmse[-1]
        final_valid_rmses.append(final_rmse)
        logger.info(f"Final Validation RMSE for K={k}: {final_rmse:.4f}")

    # Plot Final Validation RMSE vs. K_factors
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, final_valid_rmses, marker='o')
    plt.title("Validation RMSE vs. Number of Latent Factors")
    plt.xlabel("Latent Factors (K)")
    plt.ylabel("Final Validation RMSE")
    plt.grid(True)

    # Save the plot
    plot_filename = f"{experiment_folder}/final_rmse_vs_k.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Plot saved to {plot_filename}")
