# src/model_training.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
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

warnings.filterwarnings('ignore')  # Para evitar warnings innecesarios


def load_and_preprocess_data(filepath, scaler_path, max_rows=None):
    """
    Carga y preprocesa los datos.

    Args:
        filepath (str): Ruta al archivo CSV.
        scaler_path (str): Ruta donde guardar o cargar el scaler.
        max_rows (int, optional): Número máximo de filas a cargar. Si es None, carga tdo el dataset.

    Returns:
        X_train, X_test, y_train, y_test (DataFrame, DataFrame, Series, Series)
    """
    # Verificar si los datos preprocesados ya existen
    X_train_path = os.path.join(os.path.dirname(scaler_path), 'X_train.joblib')
    X_test_path = os.path.join(os.path.dirname(scaler_path), 'X_test.joblib')
    y_train_path = os.path.join(os.path.dirname(scaler_path), 'y_train.joblib')
    y_test_path = os.path.join(os.path.dirname(scaler_path), 'y_test.joblib')

    if all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path, scaler_path]):
        print("Cargando datos preprocesados desde archivos existentes...")
        X_train = joblib.load(X_train_path)
        X_test = joblib.load(X_test_path)
        y_train = joblib.load(y_train_path)
        y_test = joblib.load(y_test_path)
        return X_train, X_test, y_train, y_test
    else:
        print("Preprocesando datos desde el archivo CSV...")
        # Cargar el dataset
        df = pd.read_csv(filepath, low_memory=False)
        if max_rows:
            df = df.iloc[0:max_rows]

        # Convertir 'Calendar_date' a datetime
        df['Calendar_date'] = pd.to_datetime(df['Calendar_date'], format='%Y%m%d')

        # Eliminar columnas irrelevantes
        columns_to_drop = ['Calendar_date', 'route_id', 'bus_id', 'weather', 'temperature', 'day_of_week',
                           'time_of_day']
        df = df.drop(columns=columns_to_drop, axis=1)

        # Tratar 'stop_sequence' como categórica y realizar One-Hot Encoding
        df['stop_sequence'] = df['stop_sequence'].astype(str)  # Convertir a string para One-Hot Encoding
        df = pd.get_dummies(df, columns=['stop_sequence'], prefix='stop_seq')

        # Separar características y variable objetivo
        X = df.drop(['arrival_delay'], axis=1)
        y = df['arrival_delay']

        # Dividir los datos en entrenamiento y prueba (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalar las características continuas
        continuous_features = ['dwell_time', 'travel_time_for_previous_section', 'scheduled_travel_time',
                               'upstream_stop_delay', 'origin_delay', 'previous_bus_delay',
                               'previous_trip_travel_time', 'traffic_condition', 'recurrent_delay']

        scaler = StandardScaler()
        X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
        X_test[continuous_features] = scaler.transform(X_test[continuous_features])

        # Guardar el scaler para uso futuro
        joblib.dump(scaler, scaler_path)
        print(f"Scaler guardado en {scaler_path}")

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
            'model': XGBRegressor(device='gpu',
                                  n_jobs=-1,
                                  random_state=42,
                                  objective='reg:squarederror',
                                  eval_metric='mae',
                                  tree_method='hist',
                                  subsample=0.1,
                                  sampling_method='gradient_based'),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1,
                                           max_samples=0.1, bootstrap=True),
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
            'model': SVR(shrinking=True, tol=0.01),
            'params': {
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.5],
                'kernel': ['rbf', 'poly'], #Ver si realmente funciona
                'gamma': [0.01, 0.1, 1.0, 10]
            }
        }

    }

    return models


def objective(trial, model, params, X_train, y_train, cv, scoring):
    """
    Función objetivo para Optuna.

    Args:
        trial: Trial de Optuna.
        model: Modelo a entrenar.
        params (dict): Espacio de hiperparámetros.
        X_train: Datos de entrenamiento.
        y_train: Objetivos de entrenamiento.
        cv (int): Número de pliegues para la validación cruzada.
        scoring (str): Métrica de evaluación.

    Returns:
        float: Métrica de rendimiento media.
    """
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

    # Evaluar el modelo con validación cruzada
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    # Como 'neg_mean_absolute_error' ya es negativo, Optuna lo interpretará para maximizar
    return scores.mean()


def train_with_optuna(X_train, y_train, model, params, n_trials=50, cv=3, scoring='neg_mean_absolute_error'):
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

    Returns:
        tuple: Modelo entrenado y los mejores hiperparámetros.
    """
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(lambda trial: objective(trial, model, params, X_train, y_train, cv, scoring), n_trials=n_trials)
    best_params = study.best_params
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


def train_with_random_search(X_train, y_train, model, params, n_iter=10, cv=5, scoring='neg_mean_absolute_error',
                             n_jobs=-1):
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


def train_models(X_train, y_train, models, selected_models=None, search_method='random', n_iter=10, cv=5, scoring='neg_mean_absolute_error'):
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

    Returns:
        dict: Modelos entrenados.
        dict: Tiempos de entrenamiento por modelo.
    """
    trained_models = {}
    timings = {}

    # Si no se especifica, entrenar todos los modelos
    if selected_models is None:
        selected_models = list(models.keys())

    for name in tqdm(selected_models, desc="Entrenando modelos"):
        if name not in models:
            print(f"Modelo '{name}' no encontrado en la definición de modelos. Saltando...")
            continue

        config = models[name]
        print(f"\nEntrenando {name} con método '{search_method}'...")
        start_time = time.time()

        if search_method == 'random':
            search = train_with_random_search(
                X_train, y_train,
                config['model'],
                config['params'],
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
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
                scoring=scoring
            )
        else:
            raise ValueError("Método de búsqueda no soportado. Usa 'random' o 'optuna'.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        timings[name] = elapsed_time
        trained_models[name] = best_model
        print(f"Mejores hiperparámetros para {name}: {best_params}")
        print(f"Tiempo de entrenamiento para {name}: {elapsed_time:.2f} segundos")

    return trained_models, timings


def save_trained_models(trained_models, models_dir):
    """
    Guarda los modelos entrenados en archivos separados.

    Args:
        trained_models (dict): Diccionario de modelos entrenados.
        models_dir (str): Directorio donde guardar los modelos.
    """
    for name, model in trained_models.items():
        filename = os.path.join(models_dir, f'{name}_best_model.joblib')
        joblib.dump(model, filename)
        print(f"Modelo {name} guardado en {filename}")

def save_training_times(timings, models_dir):
    """
    Guarda los tiempos de entrenamiento en archivos separados por modelo.

    Args:
        timings (dict): Diccionario de tiempos de entrenamiento.
        models_dir (str): Directorio donde guardar los tiempos.
    """
    for name, elapsed_time in timings.items():
        filename = os.path.join(models_dir, f'{name}_training_time.joblib')
        joblib.dump(elapsed_time, filename)
        print(f"Tiempo de entrenamiento para {name} guardado en {filename}")

def main():
    """
    Función principal que orquesta el flujo de trabajo.
    """
    # Definir paths
    paths = {
        'data': os.path.join('..', 'data', 'Dataset-PT.csv'),
        'models_dir': os.path.join('..', 'models'),
        'scaler_path': os.path.join('..', 'models', 'scaler.joblib'),
        # 'timings_path': os.path.join('..', 'models', 'training_timings.joblib'),  # Ahora se guarda por modelo
        'trained_models_dir': os.path.join('..', 'models'),  # Puedes agregar más si es necesario
    }

    # Crear el directorio de modelos si no existe
    os.makedirs(paths['models_dir'], exist_ok=True)

    # Parámetros globales de entrenamiento
    config = {
        'n_iter': 50,  # Número de iteraciones para Random Search y n_trials para Optuna
        'cv': 3,  # Número de pliegues para la validación cruzada
        'scoring': 'neg_mean_absolute_error',  # Métrica de evaluación
        'search_method': 'optuna',  # Método de búsqueda: 'random' o 'optuna'
        'random_state': 42,  # Semilla aleatoria para reproducibilidad
        'n_jobs': -1,  # Número de procesos para paralelización
        # Otros parámetros globales que consideres necesarios
    }

    # Opcional: Especificar qué modelos entrenar
    # 'RandomForest', 'XGBoost', 'MLPRegressor', 'KNeighbors', 'SVR'
    selected_models = ['KNeighbors']

    # Cargar y preprocesar los datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(paths['data'], paths['scaler_path'], max_rows=1000)

    # Guardar el conjunto de prueba para evaluación futura
    X_test_path = os.path.join(paths['models_dir'], 'X_test.joblib')
    y_test_path = os.path.join(paths['models_dir'], 'y_test.joblib')
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        joblib.dump(X_test, X_test_path)
        joblib.dump(y_test, y_test_path)
        print(f"Conjunto de prueba guardado en '{X_test_path}' y '{y_test_path}'")
    else:
        print("Conjunto de prueba ya existe. No se guarda nuevamente.")

    # Definir los modelos y sus hiperparámetros
    models = define_models()

    # Entrenar los modelos con el método de búsqueda seleccionado
    trained_models, timings = train_models(
        X_train, y_train, models,
        selected_models=selected_models,
        search_method=config['search_method'],
        n_iter=config['n_iter'],
        cv=config['cv'],
        scoring=config['scoring']
    )

    # Guardar los modelos entrenados
    save_trained_models(trained_models, paths['trained_models_dir'])

    # Guardar los tiempos de entrenamiento por modelo
    save_training_times(timings, paths['trained_models_dir'])

    print("Entrenamiento de modelos completado.")


if __name__ == "__main__":
    main()
