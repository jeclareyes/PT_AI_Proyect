# src/model_training.py

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
import torch  # Para verificar CUDA
from sklearn.metrics import mean_absolute_error
import numpy as np


warnings.filterwarnings('ignore')  # Para evitar warnings innecesarios


def load_and_preprocess_data(filepath, scenario_id, sample_type, sample_subdir, max_rows=None):
    """
    Carga y preprocesa los datos para una combinación específica de escenario y muestra.

    Args:
        filepath (str): Ruta al archivo CSV.
        scenario_id (str): Identificador del escenario ('s1', 's2', 's3').
        sample_type (str): Tipo de muestra ('no_sample', 'stratified', 'kMeans').
        sample_subdir (str): Subcarpeta del escenario donde se guardarán los archivos.
        max_rows (int, optional): Número máximo de filas a cargar. Si es None, carga to.do el dataset.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    prefix = scenario_id
    sample_suffix = sample_type
    sample_subdir = os.path.join(sample_subdir, sample_type)
    os.makedirs(sample_subdir, exist_ok=True)

    # Definir rutas para los conjuntos de datos preprocesados
    X_train_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_X_train.joblib')
    X_test_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_X_test.joblib')
    y_train_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_y_train.joblib')
    y_test_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_y_test.joblib')
    scaler_path = os.path.join(sample_subdir, f'{prefix}_{sample_suffix}_scaler.joblib')

    # Verificar si los datos preprocesados ya existen
    if all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path, scaler_path]):
        print(f"Cargando datos preprocesados desde archivos existentes para {scenario_id} - {sample_type}...")
        X_train = joblib.load(X_train_path)
        X_test = joblib.load(X_test_path)
        y_train = joblib.load(y_train_path)
        y_test = joblib.load(y_test_path)
        return X_train, X_test, y_train, y_test
    else:
        print(f"Preprocesando datos desde el archivo CSV para {scenario_id} - {sample_type}...")
        # Cargar el dataset
        df = pd.read_csv(filepath, low_memory=False)
        if max_rows:
            df = df.iloc[0:max_rows]

        # Convertir 'Calendar_date' a datetime si existe
        if 'Calendar_date' in df.columns:
            df['Calendar_date'] = pd.to_datetime(df['Calendar_date'], format='%Y%m%d')

        # Eliminar columnas irrelevantes si existen
        columns_to_drop = ['Calendar_date', 'route_id', 'bus_id', 'weather', 'temperature', 'day_of_week', 'time_of_day']
        columns_present_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_present_to_drop, axis=1)

        # Tratar 'stop_sequence' como categórica y realizar One-Hot Encoding si existe
        if 'stop_sequence' in df.columns:
            df['stop_sequence'] = df['stop_sequence'].astype(str)  # Convertir a string para One-Hot Encoding
            df = pd.get_dummies(df, columns=['stop_sequence'], prefix='stop_seq')
        else:
            print("'stop_sequence' no presente en los datos. Skipping One-Hot Encoding.")

        # Separar características y variable objetivo
        if 'arrival_delay' not in df.columns:
            raise KeyError("La columna 'arrival_delay' no está presente en el dataset.")

        X = df.drop(['arrival_delay'], axis=1)
        y = df['arrival_delay']

        # Dividir los datos en entrenamiento y prueba (80/20) para cada muestra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Definir características continuas dinámicamente
        all_continuous_features = ['dwell_time', 'travel_time_for_previous_section', 'scheduled_travel_time',
                                   'upstream_stop_delay', 'origin_delay', 'previous_bus_delay',
                                   'previous_trip_travel_time', 'traffic_condition', 'recurrent_delay']
        present_continuous = [feat for feat in all_continuous_features if feat in X_train.columns]

        if present_continuous:
            scaler = StandardScaler()
            X_train[present_continuous] = scaler.fit_transform(X_train[present_continuous])
            X_test[present_continuous] = scaler.transform(X_test[present_continuous])
            # Guardar el scaler para uso futuro
            joblib.dump(scaler, scaler_path)
            print(f"Scaler guardado en {scaler_path}")
        else:
            print("No continuous features found to scale.")

        # Guardar los conjuntos de datos preprocesados
        joblib.dump(X_train, X_train_path)
        joblib.dump(X_test, X_test_path)
        joblib.dump(y_train, y_train_path)
        joblib.dump(y_test, y_test_path)
        print(f"Conjuntos de datos guardados en '{X_train_path}', '{X_test_path}', '{y_train_path}', '{y_test_path}'")

        return X_train, X_test, y_train, y_test


def define_models():
    """
    Define los modelos y sus espacios de hiperparámetros.

    Returns:
        dict: Diccionario de modelos con sus hiperparámetros.
    """
    # Detectar si CUDA está disponible para XGBoost
    cuda_available = torch.cuda.is_available()
    xgb_tree_method = 'gpu_hist' if cuda_available else 'hist'

    models = {
        'MLPRegressor': {
            'model': MLPRegressor(max_iter=1000, early_stopping=True,
                                  validation_fraction=0.1, random_state=42,
                                  learning_rate='adaptive'),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100),
                                       (150, 100), (150, 150), (150, 100, 50)],
                'activation': ['relu', 'tanh'],  # relu tanh
                'solver': ['adam'],  # adam lbfgs
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01, 0.1],  # Opcional
                'batch_size': ['auto', 32, 64, 128]
            }
        },
        'KNeighbors': {
            'model': KNeighborsRegressor(n_jobs=-1),
            'params': {
                'n_neighbors': [5, 10, 15],
                'weights': ['uniform', 'distance'],
                'leaf_size': [20, 30, 40, 50],
                'p': [1, 2]
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
                subsample=0.1,  # Aumenté subsample de 0.1 a 0.8 para mayor estabilidad
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
                'kernel': ['linear', 'rbf'],  # Cambiado de 'precomputed' a 'linear' y 'rbf'
                'gamma': ['scale', 'auto', 0.01, 0.1, 1.0, 10]
            }
        }

    }

    return models


def objective(trial, model, params, X_train, y_train, cv, scoring, scenario_id, sample_type):
    """
    Función objetivo para Optuna con validación cruzada manual.

    Args:
        trial: Trial de Optuna.
        model: Modelo a entrenar.
        params (dict): Espacio de hiperparámetros.
        X_train: Datos de entrenamiento.
        y_train: Objetivos de entrenamiento.
        cv (int): Número de pliegues para la validación cruzada.
        scoring (str): Métrica de evaluación.
        scenario_id (str): Identificador del escenario ('s1', 's2', 's3').
        sample_type (str): Tipo de muestra ('no_sample', 'stratified', 'kMeans').

    Returns:
        float: Métrica de rendimiento media.
    """
    # Sugerir hiperparámetros
    param_suggestions = {}
    for param, values in params.items():
        if isinstance(values, list):
            param_suggestions[param] = trial.suggest_categorical(param, values)
        elif isinstance(values, tuple):
            param_suggestions[param] = trial.suggest_float(param, *values)
        else:
            param_suggestions[param] = trial.suggest_categorical(param, values)

    # Configurar el modelo con los hiperparámetros sugeridos
    model.set_params(**param_suggestions)

    # Implementar K-Fold manual
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mae_scores.append(mae)

        # Reportar el MAE de este pliegue a Optuna
        trial.report(mae, step=fold)

        # Decidir si podar el trial
        if trial.should_prune():
            print(f"Trial {trial.number} podado en pliegue {fold} para {scenario_id} - {sample_type} con MAE={mae}")
            raise optuna.exceptions.TrialPruned()

    return np.mean(mae_scores)

def train_with_optuna(X_train, y_train, model, params, n_trials=50, cv=3, scoring='neg_mean_absolute_error', scenario_id='s1', sample_type='no_sample'):
    """
    Entrena un modelo usando Optuna para la optimización de hiperparámetros.

    Args:
        X_train: Datos de entrenamiento.
        y_train: Objetivos de entrenamiento.
        model: Modelo a entrenar.
        params (dict): Espacio de hiperparámetros.
        n_trials (int): Número de trials de Optuna.
        cv (int): Número de pliegues para la validación cruzada.
        scoring (str): Métrica de evaluación.
        scenario_id (str): Identificador del escenario ('s1', 's2', 's3').
        sample_type (str): Tipo de muestra ('no_sample', 'stratified', 'kMeans').

    Returns:
        tuple: Modelo entrenado y los mejores hiperparámetros.
    """
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(direction='minimize', pruner=pruner)  # 'minimize' para MAE
    study.optimize(lambda trial: objective(trial, model, params, X_train, y_train, cv, scoring, scenario_id, sample_type), n_trials=n_trials)

    best_params = study.best_params
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    return model, best_params


def train_with_random_search(X_train, y_train, model, params, n_iter=10, cv=5, scoring='neg_mean_absolute_error',
                             n_jobs=-1, scenario_id='s1', sample_type='no_sample'):
    """
    Entrena un modelo usando RandomizedSearchCV para la optimización de hiperparámetros.

    Args:
        X_train: Datos de entrenamiento.
        y_train: Objetivos de entrenamiento.
        model: Modelo a entrenar.
        params (dict): Espacio de hiperparámetros.
        n_iter (int): Número de iteraciones para RandomizedSearchCV.
        cv (int): Número de pliegues para la validación cruzada.
        scoring (str): Métrica de evaluación.
        n_jobs (int): Número de procesos para paralelización.
        scenario_id (str): Identificador del escenario ('s1', 's2', 's3').
        sample_type (str): Tipo de muestra ('no_sample', 'stratified', 'kMeans').

    Returns:
        RandomizedSearchCV: Objeto de RandomizedSearchCV entrenado.
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=1,
        random_state=42,
        n_jobs=n_jobs
    )
    random_search.fit(X_train, y_train)
    return random_search


def train_models(X_train, y_train, models, selected_models=None, search_method='random', n_iter=10, cv=5, scoring='neg_mean_absolute_error', scenario_id='s1', sample_type='no_sample'):
    """
    Entrena los modelos seleccionados usando el método de búsqueda especificado.

    Args:
        X_train: Datos de entrenamiento.
        y_train: Objetivos de entrenamiento.
        models (dict): Diccionario de modelos y sus hiperparámetros.
        selected_models (list, optional): Lista de nombres de modelos a entrenar. Si es None, entrena todos.
        search_method (str): Método de búsqueda de hiperparámetros: 'random' o 'optuna'.
        n_iter (int): Número de iteraciones para Random Search o número de trials para Optuna.
        cv (int): Número de pliegues para la validación cruzada.
        scoring (str): Métrica de evaluación.
        scenario_id (str): Identificador del escenario ('s1', 's2', 's3').
        sample_type (str): Tipo de muestra ('no_sample', 'stratified', 'kMeans').

    Returns:
        dict: Modelos entrenados.
        dict: Tiempos de entrenamiento por modelo.
    """
    trained_models = {}
    timings = {}

    # Si no se especifica, entrenar todos los modelos
    if selected_models is None:
        selected_models = list(models.keys())

    for name in tqdm(selected_models, desc=f"Entrenando modelos para {scenario_id} - {sample_type}"):
        if name not in models:
            print(f"Modelo '{name}' no encontrado en la definición de modelos. Saltando...")
            continue

        config = models[name]
        print(f"\nEntrenando {name} con método '{search_method}' para {scenario_id} - {sample_type}...")
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
                n_trials=n_iter,  # Usar n_iter directamente como n_trials
                cv=cv,
                scoring=scoring,
                scenario_id=scenario_id,
                sample_type=sample_type
            )
        else:
            raise ValueError("Método de búsqueda no soportado. Usa 'random' o 'optuna'.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        timings[name] = elapsed_time
        trained_models[name] = best_model
        print(f"Mejores hiperparámetros para {name} en {scenario_id} - {sample_type}: {best_params}")
        print(f"Tiempo de entrenamiento para {name} en {scenario_id} - {sample_type}: {elapsed_time:.2f} segundos")

    return trained_models, timings


def save_trained_models(trained_models, sample_subdir, prefix='s1_no_sample'):
    """
    Guarda los modelos entrenados en archivos separados.

    Args:
        trained_models (dict): Diccionario de modelos entrenados.
        sample_subdir (str): Subcarpeta donde guardar los modelos.
        prefix (str): Prefijo para los nombres de los archivos (e.g., 's1_no_sample').
    """
    for name, model in trained_models.items():
        filename = os.path.join(sample_subdir, f'{prefix}_{name}_best_model.joblib')
        joblib.dump(model, filename)
        print(f"Modelo {name} guardado en {filename}")


def save_training_times(timings, sample_subdir, prefix='s1_no_sample'):
    """
    Guarda los tiempos de entrenamiento en archivos separados por modelo.

    Args:
        timings (dict): Diccionario de tiempos de entrenamiento.
        sample_subdir (str): Subcarpeta donde guardar los tiempos.
        prefix (str): Prefijo para los nombres de los archivos (e.g., 's1_no_sample').
    """
    for name, elapsed_time in timings.items():
        filename = os.path.join(sample_subdir, f'{prefix}_{name}_training_time.joblib')
        joblib.dump(elapsed_time, filename)
        print(f"Tiempo de entrenamiento para {name} guardado en {filename}")


def main():
    """
    Función principal que orquesta el flujo de trabajo.
    """
    # Definir paths
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio actual del script
    data_dir = os.path.join(base_dir, '..', 'data')
    models_base_dir = os.path.join(base_dir, '..', 'models')

    # Crear el directorio de modelos si no existe
    os.makedirs(models_base_dir, exist_ok=True)

    # Parámetros globales de entrenamiento
    config = {
        'n_iter': 50,  # Número de iteraciones para Random Search y n_trials para Optuna
        'cv': 5,  # Número de pliegues para la validación cruzada
        'scoring': 'neg_mean_absolute_error',  # Métrica de evaluación
        'search_method': 'optuna',  # Método de búsqueda: 'random' o 'optuna'
        'random_state': 42,  # Semilla aleatoria para reproducibilidad
        'n_jobs': -1,  # Número de procesos para paralelización
        # Otros parámetros globales que consideres necesarios
    }

    # Especificar qué escenarios y muestras entrenar
    # selected_scenarios = ['s1', 's2', 's3']
    selected_scenarios = ['s1', 's2', 's3']  # Modifica esta lista según tus necesidades, e.g., ['s1', 's2']
    # selected_samples = ['no_sample', 'stratified', 'kMeans']
    selected_samples = ['stratified', 'kMeans', 'no_sample']  # Modifica esta lista según tus necesidades, e.g., ['no_sample', 'stratified']

    # Especificar qué modelos entrenar
    #  'MLPRegressor' 'KNeighbors' 'XGBoost' 'RandomForest' 'SVR'
    selected_models = ['MLPRegressor', 'KNeighbors', 'XGBoost', 'RandomForest']  # Modifica esta lista según tus necesidades, e.g., ['RandomForest', 'XGBoost']

    # Mapeo de escenarios a archivos CSV por muestra
    scenario_map = {
        's1': {
            'no_sample': 's1_no_sample.csv',
            'stratified': 's1_stratified.csv',
            'kMeans': 's1_kMeans.csv'
        },
        's2': {
            'no_sample': 's2_no_sample.csv',
            'stratified': 's2_stratified.csv',
            'kMeans': 's2_kMeans.csv'
        },
        's3': {
            'no_sample': 's3_no_sample.csv',
            'stratified': 's3_stratified.csv',
            'kMeans': 's3_kMeans.csv'
        }
    }

    # Definir los modelos y sus hiperparámetros
    models = define_models()

    # Iterar sobre cada escenario seleccionado
    for scenario_id in selected_scenarios:
        print(f"\n===== Procesando {scenario_id} =====")
        samples_map = scenario_map.get(scenario_id, None)

        if not samples_map:
            print(f"Configuración de muestras para {scenario_id} no encontrada en el mapeo. Saltando este escenario.")
            continue

        # Iterar sobre cada muestra seleccionada
        for sample_type in selected_samples:
            print(f"\n--- Muestra: {sample_type} ---")
            csv_filename = samples_map.get(sample_type, None)

            if not csv_filename:
                print(f"Archivo CSV para {scenario_id} - {sample_type} no encontrado en el mapeo. Saltando esta muestra.")
                continue

            csv_path = os.path.join(data_dir, csv_filename)

            if not os.path.exists(csv_path):
                print(f"Archivo CSV '{csv_path}' no existe. Saltando esta muestra.")
                continue

            # Definir la subcarpeta del escenario y muestra
            scenario_subdir = os.path.join(models_base_dir, f'scenario{scenario_id[-1]}')
            sample_subdir = os.path.join(scenario_subdir, sample_type)
            os.makedirs(sample_subdir, exist_ok=True)

            # Definir la ruta del scaler específico de la muestra
            scaler_path = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_scaler.joblib')

            # Cargar y preprocesar los datos
            try:
                X_train, X_test, y_train, y_test = load_and_preprocess_data(
                    filepath=csv_path,
                    scenario_id=scenario_id,
                    sample_type=sample_type,
                    sample_subdir=scenario_subdir,
                    max_rows=None  # Puedes ajustar esto si necesitas limitar las filas
                )
            except FileNotFoundError as e:
                print(e)
                print(f"Error al procesar {scenario_id} - {sample_type}. Asegúrate de que los datos están correctamente formateados.")
                continue
            except KeyError as e:
                print(e)
                print(f"Error en el preprocesamiento de {scenario_id} - {sample_type}.")
                continue

            # Guardar el conjunto de prueba para evaluación futura
            if sample_type == 'no_sample':
                X_test_path = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_X_test.joblib')
                y_test_path = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_y_test.joblib')

                if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
                    joblib.dump(X_test, X_test_path)
                    joblib.dump(y_test, y_test_path)
                    print(f"Conjunto de prueba guardado en '{X_test_path}' y '{y_test_path}'")
                else:
                    print("Conjunto de prueba ya existe. No se guarda nuevamente.")
            else:
                # Guardar también los conjuntos de prueba para muestras subsampled si es necesario
                X_test_path = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_X_test.joblib')
                y_test_path = os.path.join(sample_subdir, f'{scenario_id}_{sample_type}_y_test.joblib')

                if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
                    # Opcional: Si deseas crear un nuevo conjunto de prueba para las muestras subsampled
                    # Puedes decidir si es necesario o no
                    # Aquí, simplemente reutilizo el X_test y y_test de 'no_sample' para mantener consistencia
                    joblib.dump(X_test, X_test_path)
                    joblib.dump(y_test, y_test_path)
                    print(f"Conjunto de prueba guardado en '{X_test_path}' y '{y_test_path}'")
                else:
                    print("Conjunto de prueba ya existe. No se guarda nuevamente.")

            # Entrenar los modelos seleccionados con el método de búsqueda especificado
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

            # Definir el prefijo para los archivos (ejemplo: 's1_no_sample')
            prefix = f"{scenario_id}_{sample_type}"

            # Guardar los modelos entrenados y los tiempos de entrenamiento
            save_trained_models(trained_models, sample_subdir, prefix=prefix)
            save_training_times(timings, sample_subdir, prefix=prefix)

            print(f"===== Entrenamiento completado para {scenario_id} - {sample_type} =====\n")

    print("Entrenamiento de todos los escenarios y muestras completado.")


if __name__ == "__main__":
    main()
