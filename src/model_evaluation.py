# evaluation.py

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import os


def load_artifacts(paths):
    """
    Load the necessary artifacts for evaluation.
    Checks the existence of each file before loading.
    """
    artifacts = {}

    # Check and load the scaler
    if os.path.exists(paths['scaler_path']):
        try:
            artifacts['scaler'] = joblib.load(paths['scaler_path'])
            print(f"Scaler loaded from '{paths['scaler_path']}'")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            artifacts['scaler'] = None
    else:
        print(f"Error: Scaler file '{paths['scaler_path']}' does not exist.")
        artifacts['scaler'] = None

    # Check and load the test set
    if os.path.exists(paths['X_test_path']) and os.path.exists(paths['y_test_path']):
        try:
            artifacts['X_test'] = joblib.load(paths['X_test_path'])
            artifacts['y_test'] = joblib.load(paths['y_test_path'])
            print(f"Test set loaded from '{paths['X_test_path']}' and '{paths['y_test_path']}'")
        except Exception as e:
            print(f"Error loading test set: {e}")
            artifacts['X_test'] = None
            artifacts['y_test'] = None
    else:
        print(
            f"Error: Test set files '{paths['X_test_path']}' or '{paths['y_test_path']}' do not exist.")
        artifacts['X_test'] = None
        artifacts['y_test'] = None

    # Check and load training timings
    if os.path.exists(paths['timings_path']):
        try:
            artifacts['timings'] = joblib.load(paths['timings_path'])
            print(f"Training timings loaded from '{paths['timings_path']}'")
        except Exception as e:
            print(f"Error loading training timings: {e}")
            artifacts['timings'] = None
    else:
        print(f"Error: Training timings file '{paths['timings_path']}' does not exist.")
        artifacts['timings'] = None

    return artifacts


def load_trained_models(models_dir, selected_models=None):
    """
    Load trained models from the specified directory.
    If selected_models is provided, only load those models.
    """
    models = {}
    for filename in os.listdir(models_dir):
        if filename.endswith('_best_model.joblib'):
            model_name = filename.replace('_best_model.joblib', '')
            if (selected_models is None) or (model_name in selected_models):
                model_path = os.path.join(models_dir, filename)
                if os.path.exists(model_path):
                    try:
                        model = joblib.load(model_path)
                        models[model_name] = model
                        print(f"Model '{model_name}' loaded from '{model_path}'")
                    except Exception as e:
                        print(f"Error loading model '{model_name}': {e}")
                else:
                    print(f"Error: Model file '{model_path}' does not exist.")
    return models


def evaluate_model(model, X_test, y_test):
    """
    Make predictions and calculate evaluation metrics.
    """
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


def save_metrics(metrics_df, model_name, paths):
    """
    Save the metrics DataFrame to a CSV file per model.
    If the file exists, add a new row without the header.
    If it doesn't exist, create it with a header.

    :param metrics_df: DataFrame with metrics to save.
    :param model_name: Name of the model for naming the file.
    :param paths: Dictionary with file paths.
    """
    # Define the CSV file path specific to the model
    output_path = os.path.join(paths['metrics_output_csv'], f"evaluation_metrics_{model_name}.csv")

    # Check if the file already exists
    file_exists = os.path.isfile(output_path)

    # Save the DataFrame to the CSV file, appending if it already exists
    metrics_df.to_csv(output_path, mode='a', header=not file_exists, index=False)

    print(f"Metrics saved to '{output_path}'")


def save_metrics_text(metrics_df, model_name, paths):
    """
    Save the metrics DataFrame to a text file per model.
    If the file exists, append the new result at the end.
    If it doesn't exist, create it with headers.

    :param metrics_df: DataFrame with metrics to save.
    :param model_name: Name of the model for naming the file.
    :param paths: Dictionary with file paths.
    """
    # Define the text file path specific to the model
    output_path = os.path.join(paths['metrics_output_txt'], f"evaluation_metrics_{model_name}.txt")

    # Check if the file already exists and has content
    file_exists = os.path.isfile(output_path) and os.path.getsize(output_path) > 0

    # Open the file in append mode
    with open(output_path, 'a') as f:
        if not file_exists:
            # Write the header if the file is new
            f.write(metrics_df.to_string(index=False))
        else:
            # Add a blank line before the new data
            f.write("\n" + metrics_df.to_string(index=False))

    print(f"Metrics saved to '{output_path}'")


def generate_plots(y_test, y_pred, model_name, plots_dir):
    """
    Generate and save a plot of Predictions vs. Real values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Actual for {model_name}')
    plt.tight_layout()
    plot_filename = f"pred_vs_real_{model_name}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Predictions vs Actual plot saved to '{plot_path}'")


def main():
    # Define a list of models to evaluate. If the list is empty, all models will be evaluated.
    # 'RandomForest' 'XGBoost' 'MLPRegressor'
    models_to_evaluate = ['SVR']  # Change or add the names of the models you want to evaluate

    # Define paths
    paths = {
        'models_dir': '../models/',  # Directory where the models are saved
        'scaler_path': '../models/scaler.joblib',
        'X_test_path': '../models/X_test.joblib',
        'y_test_path': '../models/y_test.joblib',
        'timings_path': '../models/training_timings.joblib',
        'metrics_output_csv': '../evaluations/',
        'metrics_output_txt': '../evaluations/',
        'plots_dir': '../models/plots/'  # Directory to save the plots
    }

    # Create the plots directory if it doesn't exist
    os.makedirs(paths['plots_dir'], exist_ok=True)

    # Load necessary artifacts
    artifacts = load_artifacts(paths)

    scaler = artifacts.get('scaler')
    X_test = artifacts.get('X_test')
    y_test = artifacts.get('y_test')
    timings = artifacts.get('timings')

    # Verify that the artifacts were loaded correctly
    if scaler is None or X_test is None or y_test is None or timings is None:
        print("Error: One or more necessary artifacts could not be loaded. Aborting evaluation.")
        return

    # Load trained models
    if models_to_evaluate:
        print(f"Selected models to evaluate: {models_to_evaluate}")
    else:
        print("No models specified. All available models will be evaluated.")
    trained_models = load_trained_models(paths['models_dir'], models_to_evaluate if models_to_evaluate else None)

    if not trained_models:
        print("No models found for evaluation. Aborting.")
        return

    # Evaluate each model and collect metrics
    evaluation_results = []
    for model_name, model in trained_models.items():
        print(f"\nEvaluating model '{model_name}'...")
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        # Get model hyperparameters
        hyperparams = model.get_params()
        hyperparams_str = "; ".join([f"{key}={value}" for key, value in hyperparams.items()])

        # Get training time
        training_time = timings.get(model_name, 'N/A')

        # Create a DataFrame with a single row for the current metrics
        result_df = pd.DataFrame([{
            'Model': model_name,
            'Hyperparameters': hyperparams_str,
            'Training_Time_Seconds': training_time,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2']
        }])

        # Save metrics in separate files per model
        save_metrics(result_df, model_name, paths)
        save_metrics_text(result_df, model_name, paths)

        # Add the result to the list (optional, if you still want to maintain a collection)
        evaluation_results.append(result_df)

        # Generate and save the Predictions vs. Actual plot
        generate_plots(y_test, y_pred, model_name, paths['plots_dir'])


if __name__ == "__main__":
    main()
