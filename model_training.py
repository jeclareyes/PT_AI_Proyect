# src/model_training.py

import pandas as pd
import numpy as np
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

warnings.filterwarnings('ignore')  # Para evitar warnings innecesarios


def load_and_preprocess_data(filepath):
    # Cargar el dataset
    df = pd.read_csv(filepath, low_memory=False)

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
    joblib.dump(scaler, 'models/scaler.joblib')

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
            'model': XGBRegressor(random_state=42, objective='reg:squarederror'),
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
            'model': MLPRegressor(max_iter=500, random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    }

    return models


def train_models(X_train, y_train, models, n_iter=50, cv=3, scoring='neg_mean_absolute_error'):
    trained_models = {}
    for name, config in models.items():
        print(f"Entrenando {name}...")
        random_search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        trained_models[name] = random_search.best_estimator_
        print(f"Mejores hiperparámetros para {name}: {random_search.best_params_}\n")
    return trained_models


def save_trained_models(trained_models):
    for name, model in trained_models.items():
        filename = f'models/{name}_best_model.joblib'
        joblib.dump(model, filename)
        print(f"Modelo {name} guardado en {filename}")


def main():
    # Ruta al dataset
    data_filepath = 'data/Dataset-PT.csv'

    # Cargar y preprocesar los datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_filepath)

    # Definir los modelos y sus hiperparámetros
    models = define_models()

    # Entrenar los modelos con Random Search
    trained_models = train_models(X_train, y_train, models)

    # Guardar los modelos entrenados
    save_trained_models(trained_models)

    print("Entrenamiento de modelos completado.")


if __name__ == "__main__":
    main()
