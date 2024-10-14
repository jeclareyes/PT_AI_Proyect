# plotting.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import re
import json

# Configuración de estilo para las gráficas
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})  # Evitar advertencias al abrir muchas figuras


def load_and_prepare_data(csv_path):
    """
    Carga el archivo CSV y prepara los datos para la visualización.

    Args:
        csv_path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame preparado.
    """
    # Primero, identificamos las columnas que contienen listas
    # Suponemos que las columnas condicionadas empiezan con 'MAE_', 'MSE_', etc.
    metric_prefixes = ['MAE_', 'MSE_', 'RMSE_', 'MAPE_', 'R2_']

    # Función para convertir cadenas de listas a listas reales
    def parse_list_column(cell):
        try:
            return ast.literal_eval(cell)
        except:
            return []

    # Cargar el CSV
    df = pd.read_csv(csv_path, sep=';')  # Ajusta el separador si es necesario

    # Identificar columnas que contienen listas
    list_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in metric_prefixes)]

    # Convertir las columnas de listas de cadenas a listas reales
    for col in list_columns:
        df[col] = df[col].apply(parse_list_column)

    return df


def plot_overall_metrics(df, metric='MAE', output_dir='plots'):
    """
    Genera una gráfica de barras para una métrica general (MAE o R2) en función del Modelo y el Scenario.

    Args:
        df (pd.DataFrame): DataFrame con las métricas.
        metric (str): Métrica a graficar ('MAE' o 'R2').
        output_dir (str): Directorio donde se guardarán las gráficas.
    """
    # Filtrar las columnas necesarias
    metric_cols = ['Modelo', 'Scenario', metric]

    # Verificar si la métrica existe en el DataFrame
    if metric not in df.columns:
        print(f"La métrica '{metric}' no existe en el DataFrame.")
        return

    data = df[metric_cols].dropna()

    # Crear la gráfica
    plt.figure(figsize=(12, 8))  # Aumentar el tamaño para acomodar más barras

    sns.barplot(x='Modelo', y=metric, hue='Scenario', data=data, palette='Set2', edgecolor=None)
    plt.title(f'{metric} by Model and Scenario')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.legend(title='Scenario')
    plt.tight_layout()

    # Guardar la gráfica
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{metric}_by_Model_and_Scenario.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)  # Aumentar la resolución
    plt.close()
    print(f"Gráfica '{plot_filename}' guardada en '{output_dir}'.")


def extract_categories(categorical_vars_str, conditioning_var):
    """
    Extrae las categorías de una variable condicionante desde la columna 'Categorical_Variables'.

    Args:
        categorical_vars_str (str): Cadena con las variables categóricas.
        conditioning_var (str): Nombre de la variable condicionante.

    Returns:
        List[str]: Lista de categorías.
    """
    # Las variables categóricas están en el formato 'prefixCategory', por ejemplo, 'stop_seq_1'
    # Queremos extraer las categorías correspondientes a 'conditioning_var'

    # Primero, separamos las variables categóricas por comas
    vars_list = [var.strip() for var in categorical_vars_str.split(',')]

    # Filtrar las variables que corresponden a 'conditioning_var'
    # Por ejemplo, si conditioning_var='stop_sequence', buscamos variables que empiecen con 'stop_seq_'
    # Se asume que el prefijo en 'Categorical_Variables' coincide con 'prefix_dict'

    # Define el prefijo correspondiente (ajusta esto según tu 'prefix_dict')
    prefix_mapping = {
        'stop_sequence': 'stop_seq_',
        'weather': 'factor(weather)',
        'temperature': 'factor(temperature)',
        'day_of_week': 'factor(day_of_week)',
        'time_of_day': 'factor(time_of_day)'
        # Añade más si es necesario
    }

    prefix = prefix_mapping.get(conditioning_var, '')

    # Filtrar las variables que empiezan con el prefijo
    categories = [var.replace(prefix, '') for var in vars_list if var.startswith(prefix)]

    return categories


def plot_conditioned_metric(df, metric='MAE', conditioning_var='stop_sequence', output_dir='plots'):
    """
    Genera una gráfica de barras para una métrica condicionada por una variable categórica,
    comparando diferentes escenarios.

    Args:
        df (pd.DataFrame): DataFrame con las métricas.
        metric (str): Métrica a graficar ('MAE', 'MSE', etc.).
        conditioning_var (str): Variable categórica condicionante.
        output_dir (str): Directorio donde se guardarán las gráficas.
    """
    # Identificar la columna condicionada, por ejemplo, 'MAE_stop_sequence'
    conditioned_col = f"{metric}_{conditioning_var}"

    if conditioned_col not in df.columns:
        print(f"La columna '{conditioned_col}' no existe en el DataFrame.")
        return

    # Extraer las filas que tienen datos para esta métrica condicionada
    conditioned_df = df[['Scenario', 'Categorical_Variables', conditioned_col]].dropna()

    if conditioned_df.empty:
        print(f"No hay datos para la métrica condicionada '{conditioned_col}'.")
        return

    # Extraer las categorías
    conditioned_df['Categories'] = conditioned_df['Categorical_Variables'].apply(
        lambda x: extract_categories(x, conditioning_var))

    # Explorar las listas y crear un nuevo DataFrame donde cada fila corresponde a una categoría
    data_records = []
    for _, row in conditioned_df.iterrows():
        scenario = row['Scenario']
        categories = row['Categories']
        metric_values = row[conditioned_col]

        # Verificar que la longitud de metric_values coincida con categories
        if len(metric_values) != len(categories):
            print(
                f"Advertencia: La longitud de '{conditioned_col}' no coincide con las categorías para el Scenario '{scenario}'.")
            continue

        for category, value in zip(categories, metric_values):
            data_records.append({
                'Scenario': scenario,
                'Category': category,
                'Metric_Value': value
            })

    # Crear un DataFrame a partir de los registros
    plot_df = pd.DataFrame(data_records)

    if plot_df.empty:
        print(f"No hay datos válidos para la métrica condicionada '{conditioned_col}'.")
        return

    # Ordenar las categorías numéricamente si conditioning_var es 'stop_sequence'
    if conditioning_var == 'stop_sequence':
        # Convertir las categorías a enteros para el ordenamiento
        try:
            plot_df['Category'] = plot_df['Category'].apply(lambda x: int(x))
            plot_df = plot_df.sort_values(by='Category')
        except ValueError:
            print(
                f"No se pudo convertir las categorías de '{conditioning_var}' a enteros. Asegúrate de que son numéricas.")

    # Crear la gráfica
    plt.figure(figsize=(20, 8))  # Aumentar el ancho para acomodar muchas categorías

    # Definir el orden de las categorías si es 'stop_sequence'
    if conditioning_var == 'stop_sequence':
        ordered_categories = sorted(plot_df['Category'].unique())
        sns.barplot(x='Category', y='Metric_Value', hue='Scenario', data=plot_df, palette='Set2', edgecolor=None,
                    order=ordered_categories, ci=None)
    else:
        sns.barplot(x='Category', y='Metric_Value', hue='Scenario', data=plot_df, palette='Set2', edgecolor=None, ci=None)

    plt.title(f'{metric} by {conditioning_var.capitalize()} and Scenario')
    plt.ylabel(metric)
    plt.xlabel(conditioning_var.capitalize())
    plt.legend(title='Scenario')
    plt.xticks(rotation=90)  # Rotar las etiquetas para mejor legibilidad
    plt.tight_layout()

    # Guardar la gráfica
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{metric}_by_{conditioning_var}_and_Scenario.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)  # Aumentar la resolución
    plt.close()
    print(f"Gráfica '{plot_filename}' guardada en '{output_dir}'.")


def main():
    # Ruta al archivo CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, '..', 'metrics_completo')
    csv_path = os.path.join(folder, 'evaluation_metrics.csv')  # Ajusta el nombre y ruta según corresponda

    # Verificar que el archivo existe
    if not os.path.exists(csv_path):
        print(f"El archivo '{csv_path}' no existe. Asegúrate de que la ruta es correcta.")
        return

    # Cargar y preparar los datos
    df = load_and_prepare_data(csv_path)
    print("Datos cargados y preparados correctamente.")

    # Crear directorio para las gráficas
    output_dir = '../plots_completo'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Gráficas de Barras para MAE y R2 en Función del Modelo y del Scenario
    for metric in ['MAE', 'R2']:
        plot_overall_metrics(df, metric=metric, output_dir=output_dir)

    # 2. Gráficas de Barras para MAE por Sufijo de Categoría Comparando Escenarios
    # Identificar todas las columnas condicionadas de MAE
    conditioned_mae_cols = [col for col in df.columns if col.startswith('MAE_')]

    # Extraer los sufijos de las columnas condicionadas
    conditioning_vars = [col.split('_', 1)[1] for col in conditioned_mae_cols]

    # Para cada variable condicionante, generar la gráfica
    for conditioning_var in conditioning_vars:
        plot_conditioned_metric(df, metric='MAE', conditioning_var=conditioning_var, output_dir=output_dir)


if __name__ == "__main__":
    main()
