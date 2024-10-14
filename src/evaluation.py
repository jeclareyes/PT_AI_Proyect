# evaluation.py

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_artifacts(paths, scenario):
    """
    Carga los artefactos necesarios para la evaluación desde la carpeta 'no_sample' de cada escenario.
    """
    artifacts = {}

    # Ruta a 'no_sample' para este escenario
    no_sample_dir = os.path.join(paths['models_base_dir'], scenario, 'no_sample')

    # Definir rutas para X_test y y_test
    X_test_path = os.path.join(no_sample_dir, f"{scenario}_no_sample_X_test.joblib")
    y_test_path = os.path.join(no_sample_dir, f"{scenario}_no_sample_y_test.joblib")

    # Verificar existencia de X_test y y_test
    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        try:
            artifacts['X_test'] = joblib.load(X_test_path)
            artifacts['y_test'] = joblib.load(y_test_path)
            logging.info(f"Cargados X_test y y_test desde '{no_sample_dir}'")
        except Exception as e:
            logging.error(f"Error al cargar X_test o y_test desde '{no_sample_dir}': {e}")
            artifacts['X_test'] = None
            artifacts['y_test'] = None
    else:
        logging.error(f"Error: X_test o y_test no encontrados en '{no_sample_dir}'.")
        artifacts['X_test'] = None
        artifacts['y_test'] = None

    # Cargar variables categóricas originales
    categorical_vars_path = os.path.join(no_sample_dir, f"{scenario}_no_sample_categorical_vars.joblib")
    if os.path.exists(categorical_vars_path):
        try:
            artifacts['categorical_vars_original'] = joblib.load(categorical_vars_path)
            logging.info(f"Variables categóricas originales cargadas desde '{categorical_vars_path}'")
        except Exception as e:
            logging.error(f"Error al cargar variables categóricas originales desde '{categorical_vars_path}': {e}")
            artifacts['categorical_vars_original'] = []
    else:
        logging.error(f"Error: Variables categóricas originales no encontradas en '{categorical_vars_path}'.")
        artifacts['categorical_vars_original'] = []

    return artifacts

def load_trained_model(model_path):
    """
    Carga un modelo entrenado desde la ruta especificada.
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Modelo cargado desde '{model_path}'")
        return model
    except FileNotFoundError:
        logging.error(f"Modelo no encontrado en '{model_path}'.")
    except Exception as e:
        logging.error(f"Error al cargar el modelo desde '{model_path}': {e}")
    return None

def evaluate_model(model, X_test, y_test):
    """
    Realiza predicciones y calcula las métricas de evaluación.
    """
    if not hasattr(model, 'predict'):
        raise AttributeError("El modelo no tiene el método 'predict'.")

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
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
    Evalúa MAE, MSE, RMSE, MAPE y R2 condicionados por una variable categórica.

    Args:
        model: Modelo entrenado.
        X_test (pd.DataFrame): DataFrame de características de prueba.
        y_test (pd.Series): Serie con los valores reales.
        conditioning_variable (str): Nombre de la variable categórica condicionante.
        prefix_dict (dict): Diccionario que mapea variables categóricas a sus prefijos de One-Hot Encoding.

    Returns:
        Dict con métricas condicionadas.
    """
    # Reconstruir la variable categórica original usando expresiones regulares
    prefix = prefix_dict.get(conditioning_variable)
    if prefix is None:
        logging.error(f"No se encontró el prefijo para la variable condicionante '{conditioning_variable}'.")
        return None

    # Identificar columnas que corresponden al One-Hot Encoding de la variable
    cat_cols = [col for col in X_test.columns if re.match(rf'^{re.escape(prefix)}\w+', col)]
    if not cat_cols:
        logging.error(f"No se encontraron columnas en X_test que correspondan al prefijo '{prefix}'.")
        return None

    # Reconstruir la variable categórica original
    # Usamos expresiones regulares para eliminar el prefijo de manera precisa
    try:
        reconstructed_var = X_test[cat_cols].idxmax(axis=1).apply(lambda x: re.sub(rf'^{re.escape(prefix)}', '', x))
        reconstructed_var.name = conditioning_variable
    except Exception as e:
        logging.error(f"Error al reconstruir la variable categórica '{conditioning_variable}': {e}")
        return None

    # Añadir las predicciones y el objetivo al DataFrame
    conditional_df = pd.DataFrame()
    conditional_df['y_true'] = y_test.reset_index(drop=True)
    conditional_df['y_pred'] = model.predict(X_test)
    conditional_df[conditioning_variable] = reconstructed_var.reset_index(drop=True)

    # Agrupar por la variable condicionante y calcular métricas
    grouped = conditional_df.groupby(conditioning_variable)

    def calculate_metrics(df):
        try:
            mae = mean_absolute_error(df['y_true'], df['y_pred'])
            mse = mean_squared_error(df['y_true'], df['y_pred'])
            rmse = mean_squared_error(df['y_true'], df['y_pred'], squared=False)
            mape = mean_absolute_percentage_error(df['y_true'], df['y_pred'])
            r2 = r2_score(df['y_true'], df['y_pred'])
            return pd.Series({'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2})
        except Exception as e:
            logging.error(f"Error al calcular métricas para el grupo: {e}")
            return pd.Series({'MAE': None, 'MSE': None, 'RMSE': None, 'MAPE': None, 'R2': None})

    metrics_by_condition = grouped.apply(calculate_metrics).reset_index()

    # Inicializar diccionario para almacenar las métricas condicionadas
    conditional_metrics = {
        'MAE': metrics_by_condition['MAE'].tolist(),
        'MSE': metrics_by_condition['MSE'].tolist(),
        'RMSE': metrics_by_condition['RMSE'].tolist(),
        'MAPE': metrics_by_condition['MAPE'].tolist(),
        'R2': metrics_by_condition['R2'].tolist()
    }

    logging.info(f"Métricas condicionadas calculadas para '{conditioning_variable}'")

    return conditional_metrics

def save_metrics(metrics_df, model_name, scenario, paths, continuous_vars, categorical_vars, conditional_metrics=None):
    """
    Guarda el DataFrame de métricas en un archivo CSV por modelo.
    Si el archivo existe, se agrega una nueva fila sin el encabezado.
    Si no existe, se crea con encabezado.

    Además, añade columnas para Continuous_Variables y Categorical_Variables,
    así como para las métricas condicionadas si se proporcionan.

    :param metrics_df: DataFrame con las métricas a guardar.
    :param model_name: Nombre del modelo para nombrar el archivo.
    :param scenario: Nombre del escenario.
    :param paths: Diccionario con las rutas de archivos.
    :param continuous_vars: Lista de variables continuas utilizadas.
    :param categorical_vars: Lista de variables categóricas utilizadas.
    :param conditional_metrics: Dict con métricas condicionadas por variables.
    """
    # Definir la ruta del archivo CSV específico para el modelo
    output_path = os.path.join(paths['metrics_dir'], f"evaluation_metrics.csv")

    # Añadir columnas de escenario, variables continuas y categóricas
    metrics_to_save = metrics_df.copy()
    metrics_to_save['Scenario'] = scenario
    metrics_to_save['Continuous_Variables'] = ', '.join(continuous_vars)
    metrics_to_save['Categorical_Variables'] = ', '.join(categorical_vars)

    # Añadir métricas condicionadas si existen
    if conditional_metrics:
        for var, metrics in conditional_metrics.items():
            for metric_name, values in metrics.items():
                column_name = f"{metric_name}_{var}"
                # Serializar la lista de métricas como JSON para almacenarlas en el CSV
                metrics_to_save[column_name] = json.dumps(values)

    # Reordenar columnas para mayor claridad
    base_cols = ['Scenario', 'Continuous_Variables', 'Categorical_Variables',
                 'Modelo', 'Hiperparámetros', 'Tiempo_Entrenamiento_Segundos',
                 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
    conditional_cols = [col for col in metrics_to_save.columns if re.match(r'^(MAE|MSE|RMSE|MAPE|R2)_\w+$', col)]
    other_cols = [col for col in metrics_to_save.columns if col not in base_cols + conditional_cols]
    cols = base_cols + other_cols + conditional_cols
    metrics_to_save = metrics_to_save[cols]

    # Verificar si el archivo ya existe
    file_exists = os.path.isfile(output_path)

    # Guardar el DataFrame en el archivo CSV, añadiendo si ya existe
    try:
        metrics_to_save.to_csv(output_path, sep=';', mode='a', header=not file_exists, index=False)
        logging.info(f"Métricas guardadas en '{output_path}'")
    except Exception as e:
        logging.error(f"Error al guardar métricas en '{output_path}': {e}")

#%%
#Plots

def generate_plots(y_test, y_pred, model_name, plots_dir, scenario, sample_type):
    """
    Genera y guarda un gráfico de Predicciones vs. Reales.
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

        # Evitar sobrescribir archivos existentes añadiendo un timestamp si ya existe
        if os.path.exists(plot_path):
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(plot_path)
            plot_path = f"{base}_{timestamp}{ext}"

        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Gráfico de Predicciones vs Reales guardado en '{plot_path}'")
    except Exception as e:
        logging.error(f"Error al generar el gráfico para {model_name}: {e}")

#Box plots por categoría
def generate_error_distribution_plots(y_test, y_pred, conditioning_variable, conditioning_var, plots_dir):
    """
    Genera y guarda box plots de errores por categoría.

    Args:
        y_test (pd.Series): Valores reales.
        y_pred (np.array): Valores predichos.
        conditioning_variable (pd.Series): Variable categórica condicionante.
        plots_dir (str): Directorio para guardar los gráficos.
    """
    try:
        errors = y_pred - y_test
        df = pd.DataFrame({
            'Error': errors,
            'Categoría': conditioning_variable
        })

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Categoría', y='Error', data=df)
        plt.xlabel(conditioning_var)
        plt.ylabel('Error de Predicción (y_pred - y_true)')
        plt.title(f'Distribución de Errores por {conditioning_var}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_filename = f"error_distribution_{conditioning_var}.png"
        plot_path = os.path.join(plots_dir, plot_filename)

        if os.path.exists(plot_path):
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(plot_path)
            plot_path = f"{base}_{timestamp}{ext}"

        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Gráfico de Distribución de Errores por '{conditioning_var}' guardado en '{plot_path}'")
    except Exception as e:
        logging.error(f"Error al generar el gráfico de distribución de errores para '{conditioning_var}': {e}")

#Bar charts por categoría
def generate_metrics_bar_charts(metrics_by_condition, conditioning_variable, plots_dir):
    """
    Genera y guarda gráficos de barras para cada métrica por categoría.

    Args:
        metrics_by_condition (dict): Diccionario con métricas condicionadas.
        conditioning_variable (str): Nombre de la variable categórica condicionante.
        plots_dir (str): Directorio para guardar los gráficos.
    """
    try:
        metrics_df = pd.DataFrame(metrics_by_condition)
        metrics_df['Categoría'] = range(1, len(metrics_df['MAE']) + 1)  # Asumiendo categorías numeradas
        #  'MAE', 'MSE', 'RMSE', 'MAPE', 'R2'
        metrics = ['MAE']
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Categoría', y=metric, data=metrics_df)
            plt.xlabel(conditioning_variable)
            plt.ylabel(metric)
            plt.title(f'{metric} por {conditioning_variable}')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_filename = f"{metric}_by_{conditioning_variable}.png"
            plot_path = os.path.join(plots_dir, plot_filename)

            if os.path.exists(plot_path):
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(plot_path)
                plot_path = f"{base}_{timestamp}{ext}"

            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Gráfico de {metric} por '{conditioning_variable}' guardado en '{plot_path}'")
    except Exception as e:
        logging.error(f"Error al generar gráficos de barras para métricas condicionadas por '{conditioning_variable}': {e}")

#Scatter plots por categoría
def generate_scatter_by_category(y_test, y_pred, conditioning_variable, conditioning_var, model_name, plots_dir):
    """
    Genera y guarda scatter plots de Predicciones vs Reales color-coded por categoría.

    Args:
        y_test (pd.Series): Valores reales.
        y_pred (np.array): Valores predichos.
        conditioning_variable (pd.Series): Variable categórica condicionante.
        model_name (str): Nombre del modelo.
        plots_dir (str): Directorio para guardar los gráficos.
    """
    try:
        df = pd.DataFrame({
            'Valores Reales': y_test,
            'Predicciones': y_pred,
            'Categoría': conditioning_variable,
            'Variable': conditioning_var
        }

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='Valores Reales', y='Predicciones', hue='Categoría', palette='viridis', alpha=0.6)
        plt.plot([df['Valores Reales'].min(), df['Valores Reales'].max()],
                 [df['Valores Reales'].min(), df['Valores Reales'].max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted')
        plt.title(f'Predicted vs Actual by {conditioning_var} - {model_name}')
        plt.legend(title=conditioning_var, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plot_filename = f"pred_vs_real_by_{conditioning_var}_{model_name}.png"
        plot_path = os.path.join(plots_dir, plot_filename)

        if os.path.exists(plot_path):
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(plot_path)
            plot_path = f"{base}_{timestamp}{ext}"

        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Scatter plot Predicted vs Actual by '{conditioning_var}' guardado en '{plot_path}'")
    except Exception as e:
        logging.error(f"Error al generar scatter plot por '{conditioning_var}': {e}")


#%%
def main():
    # Definir listas internas para la selección
    selected_scenarios = ['s1', 's2', 's3']  # Lista de escenarios a evaluar
    # 'no_sample', 'stratified', 'kMeans'
    selected_samples = ['no_sample','stratified','kMeans']  # Lista de tipos de muestras a evaluar
    selected_models = ['MLPRegressor', 'KNeighbors', 'XGBoost', 'RandomForest']  # Lista de modelos a evaluar
    #  'stop_sequence', 'day_of_week_num', 'Calendar_date', 'weather', 'temperature', 'day_of_week', 'time_of_day'
    selected_conditioning_variables = ['stop_sequence', 'day_of_week_num', 'Calendar_date',
                                       'weather', 'temperature', 'day_of_week', 'time_of_day']  # Lista de variables categóricas para evaluar métricas condicionadas

    # Definir paths base
    paths = {
        'models_base_dir': '../models_completo/',  # Directorio base de modelos
        'metrics_dir': '../metrics_completo/',  # Directorio para guardar métricas
        'plots_dir': '../plots_completo/'  # Directorio para guardar gráficos
    }

    # Crear directorios si no existen
    os.makedirs(paths['metrics_dir'], exist_ok=True)
    os.makedirs(paths['plots_dir'], exist_ok=True)

    # Definir el diccionario de prefijos para variables categóricas
    prefix_dict = {
        'stop_sequence': 'stop_seq_',
        'weather': 'factor(weather)',
        'temperature': 'factor(temperature)',
        'day_of_week': 'factor(day_of_week)',
        'time_of_day': 'factor(time_of_day)'
        # Añade aquí otros pares variable: prefijo si tienes más variables categóricas
    }

    # Iterar sobre cada escenario seleccionado
    for scenario in selected_scenarios:
        logging.info(f"\n===== Procesando {scenario} =====")
        scenario_dir = os.path.join(paths['models_base_dir'], scenario)

        # Cargar X_test y y_test usando load_artifacts
        artifacts = load_artifacts(paths, scenario)
        X_test = artifacts.get('X_test')
        y_test = artifacts.get('y_test')

        if X_test is None or y_test is None:
            logging.error(f"No se pudieron cargar X_test y/o y_test para '{scenario}'. Saltando este escenario.")
            continue

        # Obtener las variables categóricas originales
        categorical_vars_original = artifacts.get('categorical_vars_original', [])
        all_features = X_test.columns.tolist()

        # Identificar variables categóricas y continuas
        categorical_prefixes = list(prefix_dict.values())
        categorical_vars = [col for col in all_features if any(col.startswith(prefix) for prefix in categorical_prefixes)]
        continuous_vars = [col for col in all_features if not any(col.startswith(prefix) for prefix in categorical_prefixes)]

        logging.info(f"\nVariables Continuas: {continuous_vars}")
        logging.info(f"Variables Categóricas: {categorical_vars}")

        # Iterar sobre cada tipo de muestra seleccionado
        for sample_type in selected_samples:
            logging.info(f"\n--- Muestra: {sample_type} ---")
            sample_dir = os.path.join(scenario_dir, sample_type)

            # Verificar existencia de la carpeta de la muestra
            if not os.path.exists(sample_dir):
                logging.error(f"Carpeta de muestra '{sample_dir}' no existe. Saltando esta muestra.")
                continue

            # Iterar sobre cada modelo seleccionado
            for model_name in selected_models:
                logging.info(f"\nEvaluando modelo '{model_name}' para {scenario} - {sample_type}")

                # Definir la ruta del modelo
                model_filename = f"{scenario}_{sample_type}_{model_name}_best_model.joblib"
                model_path = os.path.join(sample_dir, model_filename)

                # Verificar existencia del modelo
                if not os.path.exists(model_path):
                    logging.error(f"Modelo '{model_path}' no encontrado. Saltando este modelo.")
                    continue

                # Cargar el modelo
                model = load_trained_model(model_path)
                if model is None:
                    logging.error(f"Error al cargar el modelo '{model_name}'. Saltando este modelo.")
                    continue

                # Evaluar el modelo
                try:
                    metrics, y_pred = evaluate_model(model, X_test, y_test)
                except Exception as e:
                    logging.error(f"Error al evaluar el modelo '{model_name}': {e}")
                    continue

                # Obtener los hiperparámetros del modelo
                try:
                    hyperparams = model.get_params()
                    hyperparams_str = json.dumps(hyperparams)
                except Exception as e:
                    logging.error(f"Error al obtener hiperparámetros del modelo '{model_name}': {e}")
                    hyperparams_str = "N/A"

                # Obtener el tiempo de entrenamiento
                timing_filename = f"{scenario}_{sample_type}_{model_name}_training_time.joblib"
                timing_path = os.path.join(sample_dir, timing_filename)

                if os.path.exists(timing_path):
                    try:
                        training_time = joblib.load(timing_path)
                    except Exception as e:
                        logging.error(f"Error al cargar el tiempo de entrenamiento desde '{timing_path}': {e}")
                        training_time = 'N/A'
                else:
                    training_time = 'N/A'

                # Crear un DataFrame con una sola fila para las métricas actuales
                result_df = pd.DataFrame([{
                    'Modelo': model_name,
                    'Hiperparámetros': hyperparams_str,
                    'Tiempo_Entrenamiento_Segundos': training_time,
                    'MAE': metrics['MAE'],
                    'MSE': metrics['MSE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'R2': metrics['R2']
                }])

                # Evaluar métricas condicionadas por variables categóricas seleccionadas
                conditional_metrics_all = {}
                for conditioning_var in selected_conditioning_variables:
                    logging.info(f"\nEvaluando métricas condicionadas por '{conditioning_var}' para el modelo '{model_name}'")
                    conditional_metrics = evaluate_conditional_metrics(
                        model,
                        X_test,
                        y_test,
                        conditioning_var,
                        prefix_dict
                    )
                    if conditional_metrics:
                        conditional_metrics_all[conditioning_var] = conditional_metrics

                        # Obtener la variable categórica reconstruida
                        prefix = prefix_dict.get(conditioning_var)
                        cat_cols = [col for col in X_test.columns if re.match(rf'^{re.escape(prefix)}\w+', col)]
                        reconstructed_var = X_test[cat_cols].idxmax(axis=1).apply(
                            lambda x: re.sub(rf'^{re.escape(prefix)}', '', x))

                        # Generar gráficas adicionales
                        #generate_error_distribution_plots(y_test, y_pred, reconstructed_var, conditioning_var, paths['plots_dir'])
                        #generate_metrics_bar_charts(conditional_metrics, conditioning_var, paths['plots_dir'])
                        #generate_scatter_by_category(y_test, y_pred, reconstructed_var, conditioning_var, model_name, paths['plots_dir'])

                # Guardar las métricas en el archivo CSV, incluyendo métricas condicionadas
                save_metrics(
                    result_df,
                    model_name,
                    scenario,
                    paths,
                    continuous_vars,
                    categorical_vars,
                    conditional_metrics=conditional_metrics_all if conditional_metrics_all else None
                )

                # Generar y guardar el gráfico de Predicciones vs. Reales
                generate_plots(y_test, y_pred, model_name, paths['plots_dir'], scenario, sample_type)


if __name__ == "__main__":
    main()
