# src/model_training.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
import time
import optuna
import cProfile
import pstats
from io import StringIO
import tqdm

warnings.filterwarnings('ignore')  # Para evitar warnings innecesarios


def load_and_preprocess_data(filepath):
    # Cargar el dataset
    df = pd.read_csv(filepath, low_memory=False)
    #df = df.iloc[0:100] #Momentaneamente esta limitado para que cargue rapido

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
    joblib.dump(scaler, '../models/scaler.joblib')

    return X_train, X_test, y_train, y_test


def define_models():
    models = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(device='cpu', random_state=42, objective='reg:squarederror'),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
        },
        'KNeighbors': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [5, 10, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.5],
                'kernel': ['linear', 'rbf']
            }
        },
        'MLPRegressor': {
            'model': MLPRegressor(max_iter=1000, early_stopping=True,
                                  validation_fraction=0.1, random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu'],  # relu tanh
                'solver': ['adam'],  # adam lbfgs
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01, 0.1]  # Opcional
            }
        }
    }

    return models


def train_with_random_search(X_train, y_train, model, params, n_iter=5, cv=3, scoring='neg_mean_absolute_error',
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


def objective(trial, model, params, X_train, y_train, cv, scoring):
    # Definir el espacio de hiperparámetros para Optuna
    param_suggestions = {}
    for param, values in params.items():
        if isinstance(values, list):
            param_suggestions[param] = trial.suggest_categorical(param, values)
        else:
            # Para otros tipos de distribución, puedes agregar más condiciones
            param_suggestions[param] = trial.suggest_float(param, *values)

    model.set_params(**param_suggestions)

    # Evaluar el modelo con validación cruzada
    score = -1 * mean_absolute_error(y_train, model.fit(X_train, y_train).predict(X_train))
    return score


def train_with_optuna(X_train, y_train, model, params, n_trials=50, cv=3, scoring='neg_mean_absolute_error'):
    def objective_func(trial):
        param = {}
        for key, value in params.items():
            if isinstance(value, list):
                param[key] = trial.suggest_categorical(key, value)
            else:
                # Agregar más condiciones si es necesario
                param[key] = trial.suggest_float(key, *value)
        model.set_params(**param)
        # Validación cruzada
        mae = mean_absolute_error(y_train, model.fit(X_train, y_train).predict(X_train))
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_func, n_trials=n_trials)
    best_params = study.best_params
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


def train_models(X_train, y_train, models, search_method='random', n_iter=5, cv=3, scoring='neg_mean_absolute_error'):
    trained_models = {}
    timings = {}
    for name, config in tqdm.tqdm(models.items(), desc="Entrenando modelos"):
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
                n_trials=n_iter * 10,  # Ajustar según sea necesario
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


def save_trained_models(trained_models):
    for name, model in trained_models.items():
        filename = f'../models/{name}_best_model.joblib'
        joblib.dump(model, filename)
        print(f"Modelo {name} guardado en {filename}")


def main():
    # Ruta al dataset
    data_filepath = '../data/Dataset-PT.csv'

    # Cargar y preprocesar los datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_filepath)

    # Definir los modelos y sus hiperparámetros
    models = define_models()

    # Seleccionar el método de búsqueda ('random' o 'optuna')
    search_method = 'optuna'  # 'random' 'optuna'

    # Iniciar el perfilado
    profiler = cProfile.Profile()
    profiler.enable()

    # Entrenar los modelos con Random Search
    trained_models, timings = train_models(X_train, y_train, models, search_method=search_method)

    # Detener el perfilado
    profiler.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(10)  # Mostrar las 10 funciones más costosas
    with open("profiling_results.txt", "w") as f:
        f.write(s.getvalue())
    print("Perfilado guardado en 'profiling_results.txt'")

    # Guardar los modelos entrenados
    save_trained_models(trained_models)

    print("Entrenamiento de modelos completado.")

    # Guardar los tiempos de entrenamiento
    joblib.dump(timings, '../models/training_timings.joblib')
    print("Tiempos de entrenamiento guardados en 'models/training_timings.joblib'")

    print("Entrenamiento de modelos completado.")


if __name__ == "__main__":
    main()
