# src/model_training.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
import time
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')  # Para evitar warnings innecesarios


def load_and_preprocess_data(filepath, scaler_path):
    # Cargar el dataset
    df = pd.read_csv(filepath, low_memory=False)
    # df = df.iloc[0:1000]

    # Convertir 'Calendar_date' a datetime
    df['Calendar_date'] = pd.to_datetime(df['Calendar_date'], format='%Y%m%d')

    # Eliminar columnas irrelevantes
    columns_to_drop = ['Calendar_date', 'route_id', 'bus_id', 'weather', 'temperature', 'day_of_week', 'time_of_day']
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

    return X_train, X_test, y_train, y_test


def define_models():
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
            'model': XGBRegressor(device='cuda', n_jobs=-1,
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
            'model': SVR(shrinking=True, tol=0.0001),
            'params': {
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.5],
                'kernel': ['linear', 'rbf'],
                'gamma': [0.01, 0.1, 1.0, 10]
            }
        }

    }

    return models


def objective(trial, model, params, X_train, y_train, cv, scoring):
    # Definir el espacio de hiperparámetros para Optuna
    param_suggestions = {}
    current_kernel = None
    for param, values in params.items():
        if isinstance(values, list):
            param_suggestions[param] = trial.suggest_categorical(param, values)
        else:
            param_suggestions[param] = trial.suggest_float(param, *values)

    # Configurar el modelo con los hiperparámetros sugeridos
    model.set_params(**param_suggestions)

    # Evaluar el modelo con validación cruzada
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    # Como 'neg_mean_absolute_error' ya es negativo, Optuna lo interpretará para maximizar
    return scores.mean()


def train_with_optuna(X_train, y_train, model, params, n_trials=50, cv=3, scoring='neg_mean_absolute_error'):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(lambda trial: objective(trial, model, params, X_train, y_train, cv, scoring), n_trials=n_trials)
    best_params = study.best_params
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


def train_with_random_search(X_train, y_train, model, params, n_iter=10, cv=5, scoring='neg_mean_absolute_error',
                             n_jobs=-1):
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


def train_models(X_train, y_train, models, search_method='random', n_iter=10, cv=5, scoring='neg_mean_absolute_error'):
    trained_models = {}
    timings = {}
    for name, config in tqdm(models.items(), desc="Entrenando modelos"):
        print(f"Entrenando {name} con método {search_method}...")
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
        print(f"Tiempo de entrenamiento para {name}: {elapsed_time:.2f} segundos\n")
    return trained_models, timings


def save_trained_models(trained_models, models_dir):
    for name, model in trained_models.items():
        filename = f'{models_dir}{name}_best_model.joblib'
        joblib.dump(model, filename)
        print(f"Modelo {name} guardado en {filename}")


def evaluate_model(model, X_test, y_test, paths):
    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Imprimir métricas
    print(f"Evaluación del Modelo: {model.__class__.__name__}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R²: {r2:.4f}\n")

    # Guardar métricas en un archivo
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    metrics_filepath = f"{paths['models_dir']}metrics_{model.__class__.__name__}.txt"
    with open(metrics_filepath, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Métricas guardadas en {metrics_filepath}")

    # (Opcional) Visualizar Predicciones vs. Reales
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicciones vs Reales para {model.__class__.__name__}')
    plot_filepath = f"{paths['models_dir']}pred_vs_real_{model.__class__.__name__}.png"
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Gráfico de Predicciones vs Reales guardado en {plot_filepath}\n")

    return metrics


def evaluate_models(trained_models, X_test, y_test, paths):
    all_metrics = {}
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, paths)
        all_metrics[name] = metrics
    return all_metrics


def main():
    # Definir paths
    paths = {
        'data': '../data/Dataset-PT.csv',
        'models_dir': '../models/',
        'scaler_path': '../models/scaler.joblib',
        'timings_path': '../models/training_timings.joblib',
        'profiling_results': 'profiling_results.txt',  # Ya no se usa
        'trained_models_dir': '../models/',  # Puedes agregar más si es necesario
    }

    # Parámetros globales de entrenamiento
    config = {
        'n_iter': 10,  # Número de iteraciones para Random Search y n_trials para Optuna
        'cv': 5,  # Número de pliegues para la validación cruzada
        'scoring': 'neg_mean_absolute_error',  # Métrica de evaluación
        'search_method': 'optuna',  # Método de búsqueda: 'random' o 'optuna'
        'random_state': 42,  # Semilla aleatoria para reproducibilidad
        'n_jobs': -1,  # Número de procesos para paralelización
        # Otros parámetros globales que consideres necesarios
    }

    # Cargar y preprocesar los datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(paths['data'], paths['scaler_path'])

    # Definir los modelos y sus hiperparámetros
    models = define_models()

    # Entrenar los modelos con el método de búsqueda seleccionado
    trained_models, timings = train_models(
        X_train, y_train, models,
        search_method=config['search_method'],
        n_iter=config['n_iter'],
        cv=config['cv'],
        scoring=config['scoring']
    )

    # Guardar los modelos entrenados
    save_trained_models(trained_models, paths['trained_models_dir'])

    print("Entrenamiento de modelos completado.")

    # Guardar los tiempos de entrenamiento
    joblib.dump(timings, paths['timings_path'])
    print(f"Tiempos de entrenamiento guardados en '{paths['timings_path']}'")

    # Etapa de Evaluación
    print("Iniciando evaluación de modelos...")
    evaluation_metrics = evaluate_models(trained_models, X_test, y_test, paths)
    print("Evaluación de modelos completada.")

    print("Entrenamiento y evaluación de modelos completado.")


if __name__ == "__main__":
    main()
