# evaluation.py
#AI tools were used to develop and enhance this code


import joblib
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score,
                             root_mean_squared_error)
import matplotlib.pyplot as plt
import os
import re
import json
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_artifacts(paths, scenario):
    """
    Loads the necessary artifacts for evaluation from the 'no_sample' folder for each scenario.
    """
    artifacts = {}

    # Path to 'no_sample' for this scenario
    no_sample_dir = os.path.join(paths['models_base_dir'], scenario, 'no_sample')

    # Define paths for X_test and y_test
    X_test_path = os.path.join(no_sample_dir, f"{scenario}_no_sample_X_test.joblib")
    y_test_path = os.path.join(no_sample_dir, f"{scenario}_no_sample_y_test.joblib")

    # Check if X_test and y_test exist
    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        try:
            artifacts['X_test'] = joblib.load(X_test_path)
            artifacts['y_test'] = joblib.load(y_test_path)
            logging.info(f"Loaded X_test and y_test from '{no_sample_dir}'")
        except Exception as e:
            logging.error(f"Error loading X_test or y_test from '{no_sample_dir}': {e}")
            artifacts['X_test'] = None
            artifacts['y_test'] = None
    else:
        logging.error(f"Error: X_test or y_test not found in '{no_sample_dir}'.")
        artifacts['X_test'] = None
        artifacts['y_test'] = None

    # Load original categorical variables
    categorical_vars_path = os.path.join(no_sample_dir, f"{scenario}_no_sample_categorical_vars.joblib")
    if os.path.exists(categorical_vars_path):
        try:
            artifacts['categorical_vars_original'] = joblib.load(categorical_vars_path)
            logging.info(f"Original categorical variables loaded from '{categorical_vars_path}'")
        except Exception as e:
            logging.error(f"Error loading original categorical variables from '{categorical_vars_path}': {e}")
            artifacts['categorical_vars_original'] = []
    else:
        logging.error(f"Error: Original categorical variables not found in '{categorical_vars_path}'.")
        artifacts['categorical_vars_original'] = []

    return artifacts

def load_trained_model(model_path):
    """
    Loads a trained model from the specified path.
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from '{model_path}'")
        return model
    except FileNotFoundError:
        logging.error(f"Model not found in '{model_path}'.")
    except Exception as e:
        logging.error(f"Error loading model from '{model_path}': {e}")
    return None

def evaluate_model(model, X_test, y_test):
    """
    Makes predictions and calculates evaluation metrics.
    """
    if not hasattr(model, 'predict'):
        raise AttributeError("The model does not have the 'predict' method.")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }, y_pred

def evaluate_conditional_metrics(model, X_test, y_test, conditioning_variable, prefix_dict):
    """
    Evaluates MAE, MSE, RMSE, MAPE, and R2 conditioned by a categorical variable.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features DataFrame.
        y_test (pd.Series): Series with true values.
        conditioning_variable (str): Name of the conditioning categorical variable.
        prefix_dict (dict): Dictionary mapping categorical variables to their One-Hot Encoding prefixes.

    Returns:
        Dict with conditional metrics.
    """
    # Reconstruct the original categorical variable using regular expressions
    prefix = prefix_dict.get(conditioning_variable)
    if prefix is None:
        logging.error(f"Prefix for conditioning variable '{conditioning_variable}' not found.")
        return None

    # Identify columns corresponding to the One-Hot Encoding of the variable
    cat_cols = [col for col in X_test.columns if re.match(rf'^{re.escape(prefix)}\w+', col)]
    if not cat_cols:
        logging.error(f"No columns found in X_test corresponding to prefix '{prefix}'.")
        return None

    # Reconstruct the original categorical variable
    try:
        reconstructed_var = X_test[cat_cols].idxmax(axis=1).apply(lambda x: re.sub(rf'^{re.escape(prefix)}', '', x))
        reconstructed_var.name = conditioning_variable
    except Exception as e:
        logging.error(f"Error reconstructing the categorical variable '{conditioning_variable}': {e}")
        return None

    # Add predictions and true values to the DataFrame
    conditional_df = pd.DataFrame({
        'y_true': y_test.reset_index(drop=True),
        'y_pred': model.predict(X_test),
        conditioning_variable: reconstructed_var.reset_index(drop=True)
    })

    # Group by the conditioning variable and calculate metrics
    grouped = conditional_df.groupby(conditioning_variable)

    def calculate_metrics(df):
        try:
            mae = mean_absolute_error(df['y_true'], df['y_pred'])
            mse = mean_squared_error(df['y_true'], df['y_pred'])
            rmse = root_mean_squared_error(df['y_true'], df['y_pred'])
            mape = mean_absolute_percentage_error(df['y_true'], df['y_pred'])
            r2 = r2_score(df['y_true'], df['y_pred'])
            return pd.Series({'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2})
        except Exception as e:
            logging.error(f"Error calculating metrics for the group: {e}")
            return pd.Series({'MAE': None, 'MSE': None, 'RMSE': None, 'MAPE': None, 'R2': None})

    metrics_by_condition = grouped[['y_true', 'y_pred']].apply(calculate_metrics).reset_index()

    # Initialize a dictionary to store the conditional metrics
    conditional_metrics = {
        'MAE': metrics_by_condition['MAE'].tolist(),
        'MSE': metrics_by_condition['MSE'].tolist(),
        'RMSE': metrics_by_condition['RMSE'].tolist(),
        'MAPE': metrics_by_condition['MAPE'].tolist(),
        'R2': metrics_by_condition['R2'].tolist()
    }

    logging.info(f"Conditional metrics calculated for '{conditioning_variable}'")

    return conditional_metrics

def save_metrics(metrics_df, model_name, sample_type, scenario, paths, continuous_vars, categorical_vars, conditional_metrics=None):
    """
    Saves the metrics DataFrame to a CSV file by model.
    If an entry already exists for the same model, sample type, and scenario, it is not overwritten.

    Additionally, it adds columns for Continuous_Variables and Categorical_Variables,
    as well as conditional metrics if provided.

    :param metrics_df: DataFrame with metrics to save.
    :param model_name: Model name for naming the file.
    :param sample_type: Sample type.
    :param scenario: Scenario name.
    :param paths: Dictionary with file paths.
    :param continuous_vars: List of continuous variables used.
    :param categorical_vars: List of categorical variables used.
    :param conditional_metrics: Dict with metrics conditioned by variables.
    """
    # Define the specific CSV file path for the metrics
    output_path = os.path.join(paths['metrics_dir'], "evaluation_metrics.csv")

    # Add scenario, continuous, and categorical variables columns
    metrics_to_save = metrics_df.copy()
    metrics_to_save['Scenario'] = scenario
    metrics_to_save['Continuous_Variables'] = ', '.join(continuous_vars)
    metrics_to_save['Categorical_Variables'] = ', '.join(categorical_vars)

    # Add conditional metrics if they exist
    if conditional_metrics:
        for var, metrics in conditional_metrics.items():
            for metric_name, values in metrics.items():
                column_name = f"{metric_name}_{var}"
                # Serialize the list of metrics as JSON to store them in the CSV
                metrics_to_save[column_name] = json.dumps(values)

    # Reorder columns for clarity
    base_cols = ['Scenario', 'Sample_type', 'Model', 'Continuous_Variables', 'Categorical_Variables',
                 'Hyperparameters', 'Training_Time_Seconds', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
    conditional_cols = [col for col in metrics_to_save.columns if re.match(r'^(MAE|MSE|RMSE|MAPE|R2)_\w+$', col)]
    other_cols = [col for col in metrics_to_save.columns if col not in base_cols + conditional_cols]
    cols = base_cols + other_cols + conditional_cols
    metrics_to_save = metrics_to_save[cols]

    # Check if the file already exists
    if os.path.isfile(output_path):
        try:
            existing_df = pd.read_csv(output_path, sep=';')
            # Check if a row with the same Model, Sample_type, and Scenario already exists
            exists = ((existing_df['Model'] == model_name) &
                      (existing_df['Sample_type'] == sample_type) &
                      (existing_df['Scenario'] == scenario)).any()
            if exists:
                logging.info(f"Metrics for model '{model_name}', sample '{sample_type}', scenario '{scenario}' already exist in '{output_path}'. Skipping.")
                return
        except Exception as e:
            logging.error(f"Error reading '{output_path}': {e}")
            # If there's an error reading the existing file, proceed to write to avoid data loss
            pass

    # Save the DataFrame to the CSV file, appending if it already exists
    try:
        metrics_to_save.to_csv(output_path, sep=';', mode='a', header=not os.path.isfile(output_path), index=False)
        logging.info(f"Metrics saved to '{output_path}'")
    except Exception as e:
        logging.error(f"Error saving metrics to '{output_path}': {e}")

def generate_plots(y_test, y_pred, model_name, plots_dir, scenario, sample_type):
    """
    Generates and saves a Predictions vs. Actual values plot.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        plt.title(f'{scenario} - {sample_type} - Predicted vs Actual - {model_name}')
        plt.tight_layout()
        plot_filename = f"pred_vs_real_{model_name}_{scenario}_{sample_type}.png"
        plot_path = os.path.join(plots_dir, plot_filename)

        # Avoid overwriting existing files by adding a timestamp if the file already exists
        if os.path.exists(plot_path):
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(plot_path)
            plot_path = f"{base}_{timestamp}{ext}"

        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Predicted vs Actual plot saved at '{plot_path}'")
    except Exception as e:
        logging.error(f"Error generating plot for {model_name}: {e}")

def main():
    # Define internal lists for selection
    selected_scenarios = ['s1', 's2', 's3']  # List of scenarios to evaluate
    # 'no_sample', 'stratified', 'kMeans'
    selected_samples = ['no_sample', 'stratified', 'kMeans']  # List of sample types to evaluate
    selected_models = ['MLPRegressor', 'KNeighbors', 'XGBoost', 'RandomForest']  # List of models to evaluate
    # 'stop_sequence', 'day_of_week_num', 'Calendar_date', 'weather', 'temperature', 'day_of_week', 'time_of_day'
    selected_conditioning_variables = ['stop_sequence', 'day_of_week_num', 'Calendar_date',
                                       'weather', 'temperature', 'day_of_week', 'time_of_day']  # List of categorical variables for evaluating conditional metrics

    # Define base paths
    paths = {
        'models_base_dir': '../models_completo/',  # Base directory for models
        'metrics_dir': '../metrics_1/',  # Directory for saving metrics
        'plots_dir': '../plots_1/'  # Directory for saving plots
    }

    # Create directories if they don't exist
    os.makedirs(paths['metrics_dir'], exist_ok=True)
    os.makedirs(paths['plots_dir'], exist_ok=True)

    # Define the prefix dictionary for categorical variables
    prefix_dict = {
        'stop_sequence': 'stop_seq_',
        'weather': 'factor(weather)',
        'temperature': 'factor(temperature)',
        'day_of_week': 'factor(day_of_week)',
        'time_of_day': 'factor(time_of_day)'
        #add more categorical variables if necessary
    }

    # Iterate over each selected scenario
    for scenario in selected_scenarios:
        logging.info(f"\n===== Processing {scenario} =====")
        scenario_dir = os.path.join(paths['models_base_dir'], scenario)

        # Load X_test and y_test using load_artifacts
        artifacts = load_artifacts(paths, scenario)
        X_test = artifacts.get('X_test')
        y_test = artifacts.get('y_test')

        if X_test is None or y_test is None:
            logging.error(f"Failed to load X_test and/or y_test for '{scenario}'. Skipping this scenario.")
            continue

        # Get the original categorical variables
        categorical_vars_original = artifacts.get('categorical_vars_original', [])
        all_features = X_test.columns.tolist()

        # Identify categorical and continuous variables
        categorical_prefixes = list(prefix_dict.values())
        # Extract original categorical variable names
        categorical_vars = [var for var in categorical_vars_original if var in prefix_dict]
        # Add the One-Hot Encoded columns corresponding to categorical variables
        categorical_vars_encoded = [col for col in all_features if any(col.startswith(prefix) for prefix in categorical_prefixes)]
        continuous_vars = [col for col in all_features if not any(col.startswith(prefix) for prefix in categorical_prefixes)]

        logging.info(f"\nContinuous Variables: {continuous_vars}")
        logging.info(f"Categorical Variables: {categorical_vars_encoded}")

        # Iterate over each selected sample type
        for sample_type in selected_samples:
            logging.info(f"\n--- Sample: {sample_type} ---")
            sample_dir = os.path.join(scenario_dir, sample_type)

            # Check if the sample folder exists
            if not os.path.exists(sample_dir):
                logging.error(f"Sample folder '{sample_dir}' does not exist. Skipping this sample.")
                continue

            # Iterate over each selected model
            for model_name in selected_models:
                logging.info(f"\nEvaluating model '{model_name}' for {scenario} - {sample_type}")

                # Define the model path
                model_filename = f"{scenario}_{sample_type}_{model_name}_best_model.joblib"
                model_path = os.path.join(sample_dir, model_filename)

                # Check if the model exists
                if not os.path.exists(model_path):
                    logging.error(f"Model '{model_path}' not found. Skipping this model.")
                    continue

                # Load the model
                model = load_trained_model(model_path)
                if model is None:
                    logging.error(f"Error loading model '{model_name}'. Skipping this model.")
                    continue

                # Evaluate the model
                try:
                    metrics, y_pred = evaluate_model(model, X_test, y_test)
                except Exception as e:
                    logging.error(f"Error evaluating model '{model_name}': {e}")
                    continue

                # Get the model's hyperparameters
                try:
                    hyperparams = model.get_params()
                    hyperparams_str = json.dumps(hyperparams)
                except Exception as e:
                    logging.error(f"Error getting hyperparameters for model '{model_name}': {e}")
                    hyperparams_str = "N/A"

                # Get the training time
                timing_filename = f"{scenario}_{sample_type}_{model_name}_training_time.joblib"
                timing_path = os.path.join(sample_dir, timing_filename)

                if os.path.exists(timing_path):
                    try:
                        training_time = joblib.load(timing_path)
                    except Exception as e:
                        logging.error(f"Error loading training time from '{timing_path}': {e}")
                        training_time = 'N/A'
                else:
                    training_time = 'N/A'

                # Create a single-row DataFrame for the current metrics
                result_df = pd.DataFrame([{
                    'Model': model_name,
                    'Sample_type': sample_type,
                    'Hyperparameters': hyperparams_str,
                    'Training_Time_Seconds': training_time,
                    'MAE': metrics['MAE'],
                    'MSE': metrics['MSE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'R2': metrics['R2']
                }])

                # Evaluate conditional metrics by selected categorical variables
                conditional_metrics_all = {}
                for conditioning_var in selected_conditioning_variables:
                    logging.info(f"\nEvaluating conditional metrics by '{conditioning_var}' for model '{model_name}'")
                    conditional_metrics = evaluate_conditional_metrics(
                        model,
                        X_test,
                        y_test,
                        conditioning_var,
                        prefix_dict
                    )
                    if conditional_metrics:
                        conditional_metrics_all[conditioning_var] = conditional_metrics

                # Save the metrics to the CSV file, including conditional metrics
                save_metrics(
                    result_df,
                    model_name,
                    sample_type,
                    scenario,
                    paths,
                    continuous_vars,
                    categorical_vars_encoded,
                    conditional_metrics=conditional_metrics_all if conditional_metrics_all else None
                )

                # Generate and save the Predictions vs. Actual values plot
                generate_plots(y_test, y_pred, model_name, paths['plots_dir'], scenario, sample_type)

if __name__ == "__main__":
    main()
