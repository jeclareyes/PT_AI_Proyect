# src/model_training.py
#AI tools were used to develop and enhance this code

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
import time
import optuna
from tqdm import tqdm
import torch  # For CUDA verification
from sklearn.metrics import mean_absolute_error
import numpy as np

warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

def load_and_preprocess_data(filepath, scenario_id, sample_type, sample_subdir, additional_columns_to_drop=None,
                             max_rows=None, overwrite=False):
    """
    Loads and preprocesses data for a specific combination of scenario and sample type.

    Args:
        filepath (str): Path to the CSV file.
        scenario_id (str): Scenario identifier ('s1', 's2', 's3').
        sample_type (str): Sample type ('no_sample', 'stratified', 'kMeans').
        sample_subdir (str): Subdirectory where the files will be saved.
        additional_columns_to_drop (list, optional): Additional columns to drop.
        max_rows (int, optional): Maximum number of rows to load. If None, load the entire dataset.
        overwrite (bool): If True, overwrite existing preprocessed files.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    prefix = scenario_id
    sample_suffix = sample_type
    sample_subdir = os.path.join(sample_subdir, sample_type)
    os.makedirs(sample_subdir, exist_ok=True)

    # Define paths for preprocessed datasets
    X_train_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_X_train.joblib')
    X_test_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_X_test.joblib')
    y_train_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_y_train.joblib')
    y_test_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_y_test.joblib')
    scaler_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_scaler.joblib')
    categorical_vars_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_categorical_vars.joblib')

    # Check if preprocessed data exists and if overwriting is disabled
    if (all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path, scaler_path,
                                        categorical_vars_path])) and not overwrite:
        print(f"Loading preprocessed data from existing files for {scenario_id} - {sample_type}...")
        X_train = joblib.load(X_train_path)
        X_test = joblib.load(X_test_path)
        y_train = joblib.load(y_train_path)
        y_test = joblib.load(y_test_path)
        categorical_vars_original = joblib.load(categorical_vars_path)
        return X_train, X_test, y_train, y_test, categorical_vars_original
    else:
        print(f"Preprocessing data from CSV for {scenario_id} - {sample_type}...")

        # Load the dataset
        df = pd.read_csv(filepath, low_memory=False)
        if max_rows:
            df = df.iloc[0:max_rows]

        # Convert 'Calendar_date' to datetime if it exists
        if 'Calendar_date' in df.columns:
            df['Calendar_date'] = pd.to_datetime(df['Calendar_date'], format='%Y%m%d')

        # Columns to always drop
        columns_to_drop = ['Calendar_date', 'route_id', 'bus_id', 'weather', 'temperature',
                           'day_of_week_num', 'day_of_week', 'time_of_day']

        # Add additional columns to drop if provided
        if additional_columns_to_drop:
            columns_present_to_drop = [col for col in additional_columns_to_drop if col in df.columns]
            columns_to_drop.extend(columns_present_to_drop)

        # Drop the specified columns
        df = df.drop(columns=columns_to_drop, axis=1)

        # List of original categorical variables
        categorical_vars_original = []

        # Treat 'stop_sequence' as categorical and perform One-Hot Encoding if it exists
        if 'stop_sequence' in df.columns:
            df['stop_sequence'] = df['stop_sequence'].astype(str)  # Convert to string for One-Hot Encoding
            categorical_vars_original.append('stop_sequence')
            df = pd.get_dummies(df, columns=['stop_sequence'], prefix='stop_seq')
        else:
            print("'stop_sequence' not present in data. Skipping One-Hot Encoding.")

        # Separate features and target variable
        if 'arrival_delay' not in df.columns:
            raise KeyError("The 'arrival_delay' column is not present in the dataset.")

        X = df.drop(['arrival_delay'], axis=1)
        y = df['arrival_delay']

        # Split the data into training and test sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define continuous features dynamically
        all_continuous_features = ['dwell_time', 'travel_time_for_previous_section', 'scheduled_travel_time',
                                   'upstream_stop_delay', 'origin_delay', 'previous_bus_delay',
                                   'previous_trip_travel_time', 'traffic_condition', 'recurrent_delay']
        present_continuous = [feat for feat in all_continuous_features if feat in X_train.columns]

        if present_continuous:
            scaler = StandardScaler()
            X_train[present_continuous] = scaler.fit_transform(X_train[present_continuous])
            X_test[present_continuous] = scaler.transform(X_test[present_continuous])
            # Save the scaler for future use
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved at {scaler_path}")
        else:
            print("No continuous features found to scale.")

        # Save the original categorical variables
        joblib.dump(categorical_vars_original, categorical_vars_path)
        print(f"Original categorical variables saved at {categorical_vars_path}")

        # Save the preprocessed datasets
        joblib.dump(X_train, X_train_path)
        joblib.dump(X_test, X_test_path)
        joblib.dump(y_train, y_train_path)
        joblib.dump(y_test, y_test_path)
        print(f"Datasets saved at '{X_train_path}', '{X_test_path}', '{y_train_path}', '{y_test_path}'")

        return X_train, X_test, y_train, y_test, categorical_vars_original


def define_models():
    """
    Define the models and their hyperparameter spaces.

    Returns:
        dict: Dictionary of models and their hyperparameters.
    """
    # Check if CUDA is available for XGBoost
    cuda_available = torch.cuda.is_available()
    xgb_tree_method = 'gpu_hist' if cuda_available else 'hist'

    models = {
        'MLPRegressor': {
            'model': MLPRegressor(max_iter=500, early_stopping=True,
                                  validation_fraction=0.1, random_state=42,
                                  learning_rate='adaptive', tol=1e-3),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100),
                                       (150, 100), (150, 150)],
                'activation': ['tanh'],  # relu was too computationally demanding
                'solver': ['adam'],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'batch_size': ['auto', 32, 64, 128]
            }
        },
        'KNeighbors': {
            'model': KNeighborsRegressor(n_jobs=-1, p=2),
            'params': {
                'n_neighbors': [5, 10, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto'],
                'leaf_size': [20, 30, 40, 50]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(
                device='gpu' if cuda_available else 'cpu',
                n_jobs=-1,
                random_state=42,
                objective='reg:squarederror',
                eval_metric='mae',
                tree_method=xgb_tree_method,
                subsample=0.1,  # recommended when using gradient-based sampling method
                sampling_method='gradient_based' if cuda_available else 'uniform'
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1,
                                           max_samples=0.8, bootstrap=True),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 15, 20, 25],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_leaf_nodes': [None, 10, 20, 50, 100],
                'max_features': ['sqrt', 'log2', 0.5, 0.7]
            }
        },
        'SVR': {
            'model': SVR(shrinking=True, tol=0.0001),
            'params': {
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.5],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1.0, 10]
            }
        }

    }

    return models


def objective(trial, model, params, X_train, y_train, cv, scoring, scenario_id, sample_type):
    """
    Objective function for Optuna with manual cross-validation.

    Args:
        trial: Optuna trial.
        model: Model to train.
        params (dict): Hyperparameter space.
        X_train: Training data.
        y_train: Training targets.
        cv (int): Number of cross-validation folds.
        scoring (str): Evaluation metric.
        scenario_id (str): Scenario identifier ('s1', 's2', 's3').
        sample_type (str): Sample type ('no_sample', 'stratified', 'kMeans').

    Returns:
        float: Mean performance metric.
    """
    # Suggest hyperparameters
    param_suggestions = {}
    for param, values in params.items():
        if isinstance(values, list):
            param_suggestions[param] = trial.suggest_categorical(param, values)
        elif isinstance(values, tuple):
            param_suggestions[param] = trial.suggest_float(param, *values)
        else:
            param_suggestions[param] = trial.suggest_categorical(param, values)

    # Configure the model with the suggested hyperparameters
    model.set_params(**param_suggestions)

    # Manual K-Fold implementation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mae_scores.append(mae)

        # Report the MAE for this fold to Optuna
        trial.report(mae, step=fold)

        # Decide whether to prune the trial
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at fold {fold} for {scenario_id} - {sample_type} with MAE={mae}")
            raise optuna.exceptions.TrialPruned()

    return np.mean(mae_scores)


def train_with_optuna(X_train, y_train, model, params, n_trials=50, cv=3, scoring='neg_mean_absolute_error',
                      scenario_id='s1', sample_type='no_sample'):
    """
    Trains a model using Optuna for hyperparameter optimization.

    Args:
        X_train: Training data.
        y_train: Training targets.
        model: Model to train.
        params (dict): Hyperparameter space.
        n_trials (int): Number of Optuna trials.
        cv (int): Number of cross-validation folds.
        scoring (str): Evaluation metric.
        scenario_id (str): Scenario identifier ('s1', 's2', 's3').
        sample_type (str): Sample type ('no_sample', 'stratified', 'kMeans').

    Returns:
        tuple: Trained model and the best hyperparameters.
    """
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(direction='minimize', pruner=pruner)  # 'minimize' for MAE
    study.optimize(
        lambda trial: objective(trial, model, params, X_train, y_train, cv, scoring, scenario_id, sample_type),
        n_trials=n_trials)

    best_params = study.best_params
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    return model, best_params


def train_with_random_search(X_train, y_train, model, params, n_iter=100, cv=5, scoring='neg_mean_absolute_error',
                             n_jobs=-1, scenario_id='s1', sample_type='no_sample'):
    """
    Trains a model using RandomizedSearchCV for hyperparameter optimization.

    Args:
        X_train: Training data.
        y_train: Training targets.
        model: Model to train.
        params (dict): Hyperparameter space.
        n_iter (int): Number of iterations for RandomizedSearchCV.
        cv (int): Number of cross-validation folds.
        scoring (str): Evaluation metric.
        n_jobs (int): Number of processes for parallelization.
        scenario_id (str): Scenario identifier ('s1', 's2', 's3').
        sample_type (str): Sample type ('no_sample', 'stratified', 'kMeans').

    Returns:
        RandomizedSearchCV: Trained RandomizedSearchCV object.
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=2,
        random_state=42,
        n_jobs=n_jobs
    )
    random_search.fit(X_train, y_train)
    return random_search


def train_models(X_train, y_train, models, selected_models=None, search_method='random', n_iter=100, cv=5,
                 scoring='neg_mean_absolute_error', scenario_id='s1', sample_type='no_sample'):
    """
    Trains selected models using the specified search method.

    Args:
        X_train: Training data.
        y_train: Training targets.
        models (dict): Dictionary of models and their hyperparameters.
        selected_models (list, optional): List of model names to train. If None, train all.
        search_method (str): Hyperparameter search method: 'random' or 'optuna'.
        n_iter (int): Number of iterations for Random Search or number of trials for Optuna.
        cv (int): Number of cross-validation folds.
        scoring (str): Evaluation metric.
        scenario_id (str): Scenario identifier ('s1', 's2', 's3').
        sample_type (str): Sample type ('no_sample', 'stratified', 'kMeans').

    Returns:
        dict: Trained models.
        dict: Training times by model.
    """
    trained_models = {}
    timings = {}

    # If no specific models are selected, train all models
    if selected_models is None:
        selected_models = list(models.keys())

    for name in tqdm(selected_models, desc=f"Training models for {scenario_id} - {sample_type}"):
        if name not in models:
            print(f"Model '{name}' not found in model definition. Skipping...")
            continue

        config = models[name]
        print(f"\nTraining {name} with method '{search_method}' for {scenario_id} - {sample_type}...")
        start_time = time.time()

        if search_method == 'random':
            search = train_with_random_search(
                X_train, y_train,
                config['model'],
                config['params'],
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                scenario_id=scenario_id,
                sample_type=sample_type
            )
            best_model = search.best_estimator_
            best_params = search.best_params_
        elif search_method == 'optuna':
            best_model, best_params = train_with_optuna(
                X_train, y_train,
                config['model'],
                config['params'],
                n_trials=n_iter,
                cv=cv,
                scoring=scoring,
                scenario_id=scenario_id,
                sample_type=sample_type
            )
        else:
            raise ValueError("Unsupported search method. Use 'random' or 'optuna'.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        timings[name] = elapsed_time
        trained_models[name] = best_model
        print(f"Best hyperparameters for {name} in {scenario_id} - {sample_type}: {best_params}")
        print(f"Training time for {name} in {scenario_id} - {sample_type}: {elapsed_time:.2f} seconds")

    return trained_models, timings


def save_trained_models(trained_models, sample_subdir, prefix='s1_no_sample'):
    """
    Save trained models in separate files.

    Args:
        trained_models (dict): Dictionary of trained models.
        sample_subdir (str): Subdirectory to save the models.
        prefix (str): Prefix for file names (e.g., 's1_no_sample').
    """
    for name, model in trained_models.items():
        filename = os.path.join(sample_subdir, f'{prefix}_{name}_best_model.joblib')
        joblib.dump(model, filename)
        print(f"Model {name} saved at {filename}")


def save_training_times(timings, sample_subdir, prefix='s1_no_sample'):
    """
    Save training times in separate files by model.

    Args:
        timings (dict): Dictionary of training times.
        sample_subdir (str): Subdirectory to save the times.
        prefix (str): Prefix for file names (e.g., 's1_no_sample').
    """
    for name, elapsed_time in timings.items():
        filename = os.path.join(sample_subdir, f'{prefix}_{name}_training_time.joblib')
        joblib.dump(elapsed_time, filename)
        print(f"Training time for {name} saved at {filename}")


def main():
    """
    Main function orchestrating the workflow.
    """
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    data_dir = os.path.join(base_dir, '..', 'data')
    models_base_dir = os.path.join(base_dir, '..', 'models_optuna_1')

    # Create the model directory if it doesn't exist
    os.makedirs(models_base_dir, exist_ok=True)

    # Global training parameters
    config = {
        'n_iter': 100,  # Number of iterations for Random Search and n_trials for Optuna
        'cv': 5,  # Number of cross-validation folds
        'scoring': 'neg_mean_absolute_error',  # Evaluation metric
        'search_method': 'optuna',  # Search method: 'random' or 'optuna'
        'random_state': 42,  # Random seed for reproducibility
        'n_jobs': -1,  # Number of processes for parallelization
        # Other global parameters if needed
    }

    # Option to overwrite existing preprocessed data
    overwrite_preprocessed = True  # If it is desired to overwrite existing files

    # Creating scenarios by discarding features
    scenario_columns_to_drop = {
        's1': [],  # Scenario 1: Considers all features (continuous and categorical)
        's2': ['travel_time_for_previous_section', 'recurrent_delay',
               'previous_trip_travel_time', 'scheduled_travel_time',
               'trafic_condition'],  # Scenario 2: dropped some continuous features
        's3': ['travel_time_for_previous_section', 'recurrent_delay',
               'previous_trip_travel_time', 'scheduled_travel_time',
               'traffic_condition', 'factor(weather)Light_Rain',
               'factor(weather)Light_Snow', 'factor(weather)Normal', 'factor(weather)Rain',
               'factor(weather)Snow', 'factor(temperature)Cold', 'factor(temperature)Extra_cold',
               'factor(temperature)Normal']  # Scenario 3: dropped both continuous and categorical features
    }
    # which scenarios and samples to train
    # selected_scenarios = ['s1', 's2', 's3']
    selected_scenarios = ['s1']
    # selected_samples = ['no_sample', 'stratified', 'kMeans']
    selected_samples = ['no_sample']

    # Specify which models to train
    #  'MLPRegressor' 'KNeighbors' 'XGBoost' 'RandomForest' 'SVR'
    selected_models = ['MLPRegressor']

    # Map scenarios to CSV files by sample
    scenario_map = {
        's1': {
            'no_sample': 'no_sample.csv',
            'stratified': 'stratified.csv',
            'kMeans': 'kMeans.csv'
        },
        's2': {
            'no_sample': 'no_sample.csv',
            'stratified': 'stratified.csv',
            'kMeans': 'kMeans.csv'
        },
        's3': {
            'no_sample': 'no_sample.csv',
            'stratified': 'stratified.csv',
            'kMeans': 'kMeans.csv'
        }
    }

    # Define models and their hyperparameters
    models = define_models()

    # Iterate over each selected scenario
    for scenario_id in selected_scenarios:
        print(f"\n===== Processing {scenario_id} =====")
        samples_map = scenario_map.get(scenario_id, None)

        if not samples_map:
            print(f"Sample configuration for {scenario_id} not found in the map. Skipping this scenario.")
            continue

        # Iterate over each selected sample
        for sample_type in selected_samples:
            print(f"\n--- Sample: {sample_type} ---")
            csv_filename = samples_map.get(sample_type, None)

            if not csv_filename:
                print(
                    f"CSV file for {scenario_id} - {sample_type} not found in the map. Skipping this sample.")
                continue

            csv_path = os.path.join(data_dir, csv_filename)
            print(f"reading: ", csv_filename, " from: ", csv_path)

            if not os.path.exists(csv_path):
                print(f"CSV file '{csv_path}' does not exist. Skipping this sample.")
                continue

            # Define subdirectory for the scenario and sample
            scenario_subdir = os.path.join(models_base_dir, f's{scenario_id[-1]}')
            sample_subdir = os.path.join(scenario_subdir, sample_type)
            os.makedirs(sample_subdir, exist_ok=True)

            # Define path for sample-specific scaler
            scaler_path = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_scaler.joblib')

            # Load and preprocess the data
            try:
                X_train, X_test, y_train, y_test, categorical_vars_original = load_and_preprocess_data(
                    filepath=csv_path,
                    scenario_id=scenario_id,
                    sample_type=sample_type,
                    sample_subdir=scenario_subdir,
                    additional_columns_to_drop=scenario_columns_to_drop[scenario_id],
                    max_rows=None,  # Takes just the specified number of rows. It was used only when coding for
                    # the first time to check if the code was functioning properly
                    overwrite=overwrite_preprocessed
                )
            except FileNotFoundError as e:
                print(e)
                print(
                    f"Error processing {scenario_id} - {sample_type}. Make sure the data is correctly formatted.")
                continue
            except KeyError as e:
                print(e)
                print(f"Preprocessing error for {scenario_id} - {sample_type}.")
                continue

            # Save the test set for future evaluation
            if sample_type == 'no_sample':
                X_test_path_save = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_X_test.joblib')
                y_test_path_save = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_y_test.joblib')

                if overwrite_preprocessed or not (
                        os.path.exists(X_test_path_save) and os.path.exists(y_test_path_save)):
                    joblib.dump(X_test, X_test_path_save)
                    joblib.dump(y_test, y_test_path_save)
                    print(f"Test set saved at '{X_test_path_save}' and '{y_test_path_save}'")
                else:
                    print("Test set already exists. Skipping save.")
            else:
                # Optionally save the test sets for subsampled data if necessary
                X_test_path_save = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_X_test.joblib')
                y_test_path_save = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_y_test.joblib')

                if not (os.path.exists(X_test_path_save) and os.path.exists(y_test_path_save)):
                    joblib.dump(X_test, X_test_path_save)
                    joblib.dump(y_test, y_test_path_save)
                    print(f"Test set saved at '{X_test_path_save}' and '{y_test_path_save}'")
                else:
                    print("Test set already exists. Skipping save.")

            # Train selected models using the specified search method
            trained_models, timings = train_models(
                X_train, y_train, models,
                selected_models=selected_models,
                search_method=config['search_method'],
                n_iter=config['n_iter'],
                cv=config['cv'],
                scoring=config['scoring'],
                scenario_id=scenario_id,
                sample_type=sample_type
            )

            # Define prefix for file names ex. 's1_no_sample'
            prefix = f"{scenario_id}_{sample_type}"

            # Save the trained models and training times
            save_trained_models(trained_models, sample_subdir, prefix=prefix)
            save_training_times(timings, sample_subdir, prefix=prefix)

            print(f"===== Training completed for {scenario_id} - {sample_type} =====\n")

    print("Training for all scenarios and samples completed.")


if __name__ == "__main__":
    main()
