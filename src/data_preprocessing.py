# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.power import TTestIndPower

# %%
dtype_dict = {
    'route_id': 'int16',
    'bus_id': 'int32',
    'stop_sequence': 'int16',
    'arrival_delay': 'int16',
    'dwell_time': 'uint16',
    'travel_time_for_previous_section': 'uint16',
    'scheduled_travel_time': 'uint16',
    'upstream_stop_delay': 'int16',
    'origin_delay': 'int16',
    'previous_bus_delay': 'int16',
    'previous_trip_travel_time': 'uint16',
    'traffic_condition': 'float32',
    'recurrent_delay': 'float32'
}

dummy_vars = [
    'factor(weather)Light_Rain', 'factor(weather)Light_Snow', 'factor(weather)Normal',
    'factor(weather)Rain', 'factor(weather)Snow', 'factor(temperature)Cold',
    'factor(temperature)Extra_cold', 'factor(temperature)Normal', 'factor(day_of_week)weekday',
    'factor(day_of_week)weekend', 'factor(time_of_day)Afternoon_peak',
    'factor(time_of_day)Morning_peak', 'factor(time_of_day)Off-peak'
]

for var in dummy_vars:
    dtype_dict[var] = 'uint8'
# %%
data_path = 'data/Dataset-PT.csv'

df = pd.read_csv(
    data_path,
    dtype=dtype_dict,
    parse_dates=['Calendar_date'],
    date_format='%Y%m%d'
)

numeric_cols = [
    'arrival_delay', 'dwell_time', 'travel_time_for_previous_section',
    'scheduled_travel_time', 'upstream_stop_delay', 'origin_delay',
    'previous_bus_delay', 'previous_trip_travel_time', 'traffic_condition',
    'recurrent_delay'
]

categorical_columns = ['weather', 'temperature', 'day_of_week', 'time_of_day']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Validación de las variables dummy
for var in dummy_vars:
    if not df[var].isin([0, 1]).all():
        raise ValueError(f"La variable {var} contiene valores distintos de 0 y 1")
# %%
df.insert(1, 'day_of_week_num', df['Calendar_date'].dt.dayofweek)


# %% md
# Calculo Estadísticas
# %% md
## Variables Continuas
# %%
def calcular_estadisticas_continuas(df, cols):
    # Crear un diccionario para almacenar resultados
    stats_dict = {
        'Variable': [],
        'Media': [],
        'Mediana': [],
        'Desviación Estándar': [],
        'Percentil 95%': [],
        'Máximo': []
    }

    # Calcular estadísticas para cada columna
    for col in cols:
        stats_dict['Variable'].append(col)
        stats_dict['Media'].append(df[col].mean())
        stats_dict['Mediana'].append(df[col].median())
        stats_dict['Desviación Estándar'].append(df[col].std())
        stats_dict['Percentil 95%'].append(df[col].quantile(0.95))
        stats_dict['Máximo'].append(df[col].max())

    # Convertir el diccionario a un DataFrame
    stats_df = pd.DataFrame(stats_dict)

    # Imprimir la tabla en formato tabular
    print("\n### Estadísticas Descriptivas para Variables Continuas ###\n")
    print(stats_df.to_markdown(index=False))

    return stats_df


# Calcular estadísticas descriptivas para las variables continuas
estadisticas_continuas = calcular_estadisticas_continuas(df, numeric_cols)


# %% md
## Variables categóricas
# %%
def calcular_frecuencias_categoricas(df, cols):
    print("\n### Frecuencias Absolutas y Relativas para Variables Categóricas ###\n")

    # Iterar sobre las columnas categóricas para calcular las frecuencias
    for col in cols:
        absolute_freq = df[col].value_counts()
        relative_freq = df[col].value_counts(normalize=True) * 100

        # Combinar frecuencias absolutas y relativas en un solo DataFrame
        freq_df = pd.DataFrame({
            'Frecuencia Absoluta': absolute_freq,
            'Frecuencia Relativa (%)': relative_freq
        }).reset_index().rename(columns={'index': col})

        # Imprimir la tabla en formato tabular
        print(f"\nFrecuencias para {col}:\n")
        print(freq_df.to_markdown(index=False))


# Aplicar la función a todas las variables categóricas
variables_categoricas = ['weather', 'temperature', 'day_of_week', 'time_of_day']
calcular_frecuencias_categoricas(df, variables_categoricas)


# %% md
## Variables Continuas por Día de la semana
# %%
def calcular_estadisticas_por_dia_semana(df, cols):
    # Agrupar por day_of_week_num (0=Lunes, ..., 6=Domingo) y calcular estadísticas
    grouped_stats = df.groupby('day_of_week_num')[cols].agg(['mean', 'std']).reset_index()

    # Retornar el DataFrame con estadísticas agrupadas
    return grouped_stats


# Calcular estadísticas agregadas por día de la semana excluyendo ciertas variables
estadisticas_por_dia_semana = calcular_estadisticas_por_dia_semana(df, numeric_cols_filtradas)
estadisticas_por_dia_semana


# %% md
## Variables Continuas por Stop Sequence
# %%
def calcular_estadisticas_por_grupo(df, group_by_col, cols_continuas):
    grouped_stats = df.groupby(group_by_col)[cols_continuas].agg(['mean', 'median', 'std'])
    print(f"\n### Estadísticas Agregadas por {group_by_col} ###\n")
    return grouped_stats


# Estadísticas agregadas por número de parada (stop_sequence)
estadisticas_por_parada = calcular_estadisticas_por_grupo(df, 'stop_sequence', numeric_cols)
estadisticas_por_parada


# %% md
## Variables Categóricas por Día de la semana
# %%
def calcular_frecuencias_categoricas_por_grupo(df, group_by_col, cols_categoricas):
    combined_freq_dict = {}

    # Calcular frecuencias para cada variable categórica por el grupo especificado
    for col in cols_categoricas:
        # Frecuencia absoluta por grupo
        abs_freq = df.groupby(group_by_col)[col].value_counts().unstack().fillna(0)

        # Frecuencia relativa por grupo (proporción)
        rel_freq = abs_freq.div(abs_freq.sum(axis=1), axis=0) * 100

        # Combinar frecuencias absolutas y relativas en un solo DataFrame
        combined_freq = pd.concat([rel_freq], axis=1, keys=['(%)'])

        # Reorganizar las columnas para que las frecuencias relativas estén al lado de las absolutas
        combined_freq.columns = [f'{lvl1}_{lvl2}' for lvl1, lvl2 in combined_freq.columns]

        combined_freq_dict[col] = combined_freq

    return combined_freq_dict


# Definir las variables categóricas para el análisis
variables_categoricas = ['weather', 'temperature', 'day_of_week', 'time_of_day']

# Calcular frecuencias para variables categóricas por día de la semana
frecuencias_por_dia_semana = calcular_frecuencias_categoricas_por_grupo(df, 'day_of_week_num', variables_categoricas)

# Mostrar resultados para una variable categórica como ejemplo
for col, freqs in frecuencias_por_dia_semana.items():
    print(f"\n### Frecuencias combinadas para {col} ###\n")
    print(freqs)


# %% md
## Variables Categóricas por Stop Sequence
# %%
def calcular_frecuencias_categoricas_por_stop_sequence(df, cols_categoricas):
    combined_freq_dict = {}

    # Calcular frecuencias para cada variable categórica por stop_sequence
    for col in cols_categoricas:
        # Frecuencia absoluta por grupo (stop_sequence)
        abs_freq = df.groupby('stop_sequence')[col].value_counts().unstack().fillna(0)

        # Frecuencia relativa por grupo (proporción)
        rel_freq = abs_freq.div(abs_freq.sum(axis=1), axis=0) * 100

        # Crear DataFrame solo con frecuencias relativas
        combined_freq = pd.concat([rel_freq], axis=1, keys=['(%)'])

        # Renombrar las columnas para que reflejen las categorías
        combined_freq.columns = [f'{lvl1}_{lvl2}' for lvl1, lvl2 in combined_freq.columns]

        combined_freq_dict[col] = combined_freq

    return combined_freq_dict


# Calcular frecuencias relativas para variables categóricas por número de parada
frecuencias_por_stop_sequence = calcular_frecuencias_categoricas_por_stop_sequence(df, variables_categoricas)

# Mostrar resultados para una variable categórica como ejemplo
for col, freqs in frecuencias_por_stop_sequence.items():
    print(f"\n### Frecuencias relativas para {col} por stop_sequence ###\n")
    print(freqs)

# %% md
# Plots Media y Desviación Std.
# %% md
## Variables Continuas Vs. Día de la semana
# %%
excluir_vars = ['travel_time_for_previous_section', 'scheduled_travel_time', 'origin_delay']

# Filtrar las variables numéricas excluyendo las especificadas
numeric_cols_filtradas = [col for col in numeric_cols if col not in excluir_vars]


def calcular_y_graficar_por_dia(df, cols, group_by_col='day_of_week_num', excluir=None):
    # Excluir variables si se especifica la lista excluir
    if excluir is not None:
        cols = [col for col in cols if col not in excluir]

    # Agrupar por day_of_week_num y calcular estadísticas
    grouped_stats = df.groupby(group_by_col)[cols].agg(['mean', 'std']).reset_index()

    # Calcular el número de filas necesario para la grilla (2 columnas)
    num_cols = 2
    num_rows = (len(cols) + 1) // num_cols

    # Configurar el tamaño de la figura
    plt.figure(figsize=(15, num_rows * 5))

    # Graficar cada variable continua
    for i, col in enumerate(cols, 1):
        plt.subplot(num_rows, num_cols, i)

        # Extraer media y desviación estándar
        x = grouped_stats[group_by_col]
        y_mean = grouped_stats[(col, 'mean')]
        y_std = grouped_stats[(col, 'std')]

        # Graficar media y franja de desviación estándar
        plt.plot(x, y_mean, label='Media', color='blue', marker='o')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2, label='Desviación Estándar')

        # Personalización del gráfico
        plt.title(f'Media y Desviación Estándar de {col} por {group_by_col}')
        plt.xlabel('Día de la semana (0=Lunes, 6=Domingo)')
        plt.ylabel(col.capitalize())
        plt.xticks(ticks=range(7), labels=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])  # Etiquetas de los días
        plt.legend()
        plt.grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

    # Retornar estadísticas agrupadas
    return grouped_stats


# Aplicar la función para calcular y graficar las estadísticas por día de la semana excluyendo ciertas variables
estadisticas_por_dia = calcular_y_graficar_por_dia(df, numeric_cols, excluir=excluir_vars)
# %% md
## Variables Continuas Vs. Stop Sequence
# %%
excluir_vars = ['travel_time_for_previous_section', 'scheduled_travel_time', 'origin_delay']

# Filtrar las variables numéricas excluyendo las especificadas
numeric_cols_filtradas = [col for col in numeric_cols if col not in excluir_vars]


def plot_media_desviacion_por_parada(df, cols, group_by_col='stop_sequence'):
    # Agrupar por parada y calcular estadísticas
    grouped_stats = df.groupby(group_by_col)[cols].agg(['mean', 'std']).reset_index()

    # Calcular el número de filas necesario para la grilla
    num_cols = 2
    num_rows = (len(cols) + 1) // num_cols

    # Configurar el tamaño de la figura
    plt.figure(figsize=(15, num_rows * 5))

    # Graficar cada variable continua
    for i, col in enumerate(cols, 1):
        plt.subplot(num_rows, num_cols, i)

        # Extraer media y desviación estándar
        x = grouped_stats[group_by_col]
        y_mean = grouped_stats[(col, 'mean')]
        y_std = grouped_stats[(col, 'std')]

        # Graficar media y franja de desviación estándar
        plt.plot(x, y_mean, label='Media', color='blue', marker='o')
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2, label='Desviación Estándar')

        # Personalización del gráfico
        plt.title(f'Media y Desviación Estándar de {col} por {group_by_col}')
        plt.xlabel(group_by_col.capitalize())
        plt.ylabel(col.capitalize())
        plt.legend()
        plt.grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()


# Graficar todas las variables continuas filtradas por parada
plot_media_desviacion_por_parada(df, numeric_cols_filtradas)
# %% md
## Variables Categóricas Vs. Día de la Semana
# %%
excluir_vars_categoricas = ['day_of_week']


def plot_frecuencias_categoricas_por_dia(df, cols_categoricas, group_by_col='day_of_week_num'):
    # Calcular frecuencias relativas por grupo (día de la semana)
    for col in cols_categoricas:
        # Calcular frecuencias absolutas y luego relativas
        abs_freq = df.groupby(group_by_col)[col].value_counts().unstack().fillna(0)
        rel_freq = abs_freq.div(abs_freq.sum(axis=1), axis=0) * 100

        # Configurar el tamaño de la figura
        plt.figure(figsize=(15, 7))

        # Crear gráfico para cada variable categórica
        for category in rel_freq.columns:
            plt.plot(rel_freq.index, rel_freq[category], marker='o', label=f'{col}: {category}')

        # Personalización del gráfico
        plt.title(f'Frecuencias Relativas de {col} por {group_by_col}')
        plt.xlabel('Día de la semana (0=Lunes, 6=Domingo)')
        plt.ylabel('Frecuencia Relativa (%)')
        plt.xticks(ticks=range(7), labels=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])  # Etiquetas de los días
        plt.legend()
        plt.grid(True)

        # Mostrar el gráfico
        plt.show()


# Graficar las frecuencias relativas de las variables categóricas por día de la semana
plot_frecuencias_categoricas_por_dia(df, variables_categoricas_filtradas)
# %% md
## Variables Categóricas Vs. Stop Sequence
# %%
excluir_vars_categoricas = ['day_of_week']

# Filtrar las variables categóricas excluyendo las especificadas
variables_categoricas_filtradas = [col for col in variables_categoricas if col not in excluir_vars_categoricas]


def plot_frecuencias_categoricas_por_parada(df, cols_categoricas, group_by_col='stop_sequence'):
    # Calcular frecuencias relativas por grupo (stop_sequence)
    for col in cols_categoricas:
        # Calcular frecuencias absolutas y luego relativas
        abs_freq = df.groupby(group_by_col)[col].value_counts().unstack().fillna(0)
        rel_freq = abs_freq.div(abs_freq.sum(axis=1), axis=0) * 100

        # Configurar el tamaño de la figura
        plt.figure(figsize=(15, 7))

        # Crear gráfico para cada variable categórica
        for category in rel_freq.columns:
            plt.plot(rel_freq.index, rel_freq[category], marker='o', label=f'{col}: {category}')

        # Personalización del gráfico
        plt.title(f'Frecuencias Relativas de {col} por {group_by_col}')
        plt.xlabel(group_by_col.capitalize())
        plt.ylabel('Frecuencia Relativa (%)')
        plt.legend()
        plt.grid(True)

        # Mostrar el gráfico
        plt.show()


# Graficar las frecuencias relativas de las variables categóricas por parada
plot_frecuencias_categoricas_por_parada(df, variables_categoricas_filtradas)


# %% md
# Plots Histogramas
# %% md
## Histogramas Variables Continuas
# %%
def plot_histogram(df, cols, bins=30):
    rows = (len(cols) + 1) // 2
    plt.figure(figsize=(15, rows * 4))
    for i, col in enumerate(cols, 1):
        plt.subplot(rows, 2, i)
        sns.histplot(df[col], kde=True, bins=bins)
        plt.title(f'Histograma de {col}')
    plt.tight_layout()
    plt.show()


numeric_cols = ['arrival_delay', 'dwell_time', 'travel_time_for_previous_section',
                'scheduled_travel_time', 'upstream_stop_delay', 'origin_delay',
                'previous_bus_delay', 'previous_trip_travel_time', 'traffic_condition',
                'recurrent_delay']
plot_histogram(df, numeric_cols)


# %% md
## Histogramas cruzados entre variables continuas
# %%
def plot_crossed_histograms_grid(df, pairs, bins=30):
    num_pairs = len(pairs)
    num_cols = 2
    num_rows = (num_pairs + 1) // num_cols

    plt.figure(figsize=(15, num_rows * 5))

    # Graficar cada par de variables continuas en una grilla
    for i, (col_x, col_y) in enumerate(pairs, 1):
        plt.subplot(num_rows, num_cols, i)

        # Crear gráfico de dispersión (histograma bidimensional)
        sns.histplot(data=df, x=col_x, y=col_y, bins=bins, pthresh=.1, cmap="viridis")

        # Personalización del gráfico
        plt.title(f'{col_y} vs {col_x}')
        plt.xlabel(col_x.capitalize())
        plt.ylabel(col_y.capitalize())

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()


# Sugerir algunas combinaciones interesantes de variables continuas
pairs_continuas = [
    ('arrival_delay', 'previous_bus_delay'),
    ('arrival_delay', 'upstream_stop_delay'),
    ('travel_time_for_previous_section', 'scheduled_travel_time'),
    ('previous_trip_travel_time', 'arrival_delay'),
    ('traffic_condition', 'recurrent_delay')
]

# Graficar combinaciones cruzadas de variables continuas en una grilla
plot_crossed_histograms_grid(df, pairs_continuas)


# %% md
## Histogramas cruzados entre Variables Continuas y Categóricas
# %%
def plot_continuous_vs_categorical_grid(df, continuous_vars, categorical_vars):
    # Calcular el número total de gráficos y organizar en 2 columnas
    num_plots = len(continuous_vars) * len(categorical_vars)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    plt.figure(figsize=(15, num_rows * 5))

    # Índice para la posición del gráfico en la grilla
    plot_idx = 1

    # Iterar por cada combinación de variable continua y categórica
    for col_cont in continuous_vars:
        for col_cat in categorical_vars:
            plt.subplot(num_rows, num_cols, plot_idx)
            plot_idx += 1

            # Crear boxplot de la variable continua por la variable categórica
            sns.boxplot(x=col_cat, y=col_cont, data=df)

            # Personalización del gráfico
            plt.title(f'{col_cont.capitalize()} por {col_cat.capitalize()}')
            plt.xlabel(col_cat.capitalize())
            plt.ylabel(col_cont.capitalize())
            plt.xticks(rotation=45)  # Rotar etiquetas si es necesario

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()


# Definir variables continuas y categóricas para cruzar
variables_continuas_interes = ['arrival_delay', 'travel_time_for_previous_section', 'scheduled_travel_time',
                               'traffic_condition']
variables_categoricas_interes = ['weather', 'temperature', 'time_of_day', 'day_of_week_num']

# Graficar combinaciones de variables continuas y categóricas en una grilla
plot_continuous_vs_categorical_grid(df, variables_continuas_interes, variables_categoricas_interes)


# %% md
# Bar Charts
# %% md
## Bar Chart Variables Categóricas
# %%
def plot_bar_categoricas(df, cols_categoricas):
    rows = (len(cols_categoricas) + 1) // 2
    plt.figure(figsize=(15, rows * 5))

    for i, col in enumerate(cols_categoricas, 1):
        plt.subplot(rows, 2, i)
        # Contar la frecuencia de cada categoría
        value_counts = df[col].value_counts()

        # Crear el gráfico de barras
        sns.barplot(x=value_counts.index, y=value_counts.values)

        # Personalización del gráfico
        plt.title(f'Distribución de {col}')
        plt.xlabel(col.capitalize())
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)  # Rotar las etiquetas si es necesario

    plt.tight_layout()
    plt.show()


# Definir variables categóricas a graficar
variables_categoricas = ['weather', 'temperature', 'time_of_day']

# Graficar distribución de frecuencias de variables categóricas
plot_bar_categoricas(df, variables_categoricas)


# %% md
# Box plots
# %% md
## Box plots Variables Continuas
# %%
def plot_boxplots_continuous(df, continuous_vars):
    # Calcular el número total de variables y organizar en 2 columnas
    num_vars = len(continuous_vars)
    num_cols = 2
    num_rows = (num_vars + 1) // num_cols

    plt.figure(figsize=(15, num_rows * 5))

    # Graficar cada variable continua
    for i, col in enumerate(continuous_vars, 1):
        plt.subplot(num_rows, num_cols, i)

        # Crear boxplot de la variable continua
        sns.boxplot(data=df, y=col)

        # Personalización del gráfico
        plt.title(f'Boxplot de {col}')
        plt.ylabel(col.capitalize())

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()


# Definir las variables continuas para graficar
variables_continuas = ['arrival_delay', 'travel_time_for_previous_section', 'scheduled_travel_time',
                       'traffic_condition', 'recurrent_delay', 'upstream_stop_delay', 'dwell_time']

# Graficar boxplots de las variables continuas
plot_boxplots_continuous(df, variables_continuas)
# %% md
# Outliers
# %%
from scipy import stats


def eliminar_outliers_multiple(df, cols, threshold=3):
    rows_initial = df.shape[0]  # Número de filas inicial
    eliminados_totales = pd.Series(0, index=cols)  # Serie para contar filas eliminadas por cada columna

    # Iterar sobre cada columna y eliminar outliers
    for col in cols:
        z_scores = stats.zscore(df[col])
        abs_z_scores = np.abs(z_scores)
        is_not_outlier = abs_z_scores <= threshold

        # Contar filas eliminadas
        eliminados_totales[col] = rows_initial - is_not_outlier.sum()

        # Filtrar el DataFrame para la columna actual
        df = df[is_not_outlier]

    # Calcular el total de filas después de eliminar outliers
    rows_final = df.shape[0]
    eliminados_totales_absolutos = rows_initial - rows_final

    # Calcular porcentajes de eliminación
    eliminados_porcentaje = (eliminados_totales / rows_initial) * 100
    eliminados_totales_porcentaje = (eliminados_totales_absolutos / rows_initial) * 100

    # Imprimir estadísticas de eliminación por columna
    print("\n### Estadísticas de Eliminación de Outliers ###\n")
    print(f"Total de filas iniciales: {rows_initial}")
    print(f"Total de filas finales: {rows_final}")
    print(f"Total de filas eliminadas: {eliminados_totales_absolutos} ({eliminados_totales_porcentaje:.2f}%)")
    print("\nFilas eliminadas por columna:")
    for col in cols:
        print(f" - {col}: {eliminados_totales[col]} eliminadas ({eliminados_porcentaje[col]:.2f}%)")

    return df


# Definir variables continuas para las cuales eliminar outliers
'''
variables_continuas = ['arrival_delay', 'travel_time_for_previous_section', 'scheduled_travel_time',
                       'traffic_condition', 'recurrent_delay', 'upstream_stop_delay', 'dwell_time']
'''
variables_continuas = ['arrival_delay', 'dwell_time']

# Eliminar outliers de múltiples columnas
df_cleaned_multiple = eliminar_outliers_multiple(df, variables_continuas)
# %%
# Exportar el DataFrame limpio como un archivo CSV
output_path = "data/Dataset-PT_no_sample.csv"
df_cleaned_multiple.to_csv(output_path, index=False)

print(f"El DataFrame ha sido exportado exitosamente a {output_path}")
# %% md
# Sub Sampling
# %% md
## Stratified sample
# %%
df_sorted = df_cleaned_multiple.sort_values('Calendar_date')
df_time_sampled = df_sorted.iloc[::10, :].reset_index(drop=True)

df_train, df_stratified = train_test_split(
    df_time_sampled,
    test_size=0.1,
    stratify=df_time_sampled['day_of_week'],
    random_state=42
)
# %%
# Exportar el DataFrame limpio como un archivo CSV
output_path = "data/Dataset-PT_stratified.csv"
df_stratified.to_csv(output_path, index=False)

print(f"El DataFrame ha sido exportado exitosamente a {output_path}")
# %% md
## KMeans Sample
# %%
from sklearn.cluster import KMeans


def submuestreo_kmeans(df, columns_to_exclude, n_clusters=10, sample_percentage=0.01):
    # Excluir columnas específicas
    df_filtered = df.drop(columns=columns_to_exclude)

    # Separar características (X) y variable objetivo (y)
    X = df_filtered.drop(['arrival_delay'], axis=1)
    y = df_filtered['arrival_delay']

    # Determinar el número total de filas para calcular el tamaño de muestra por cluster
    total_rows = X.shape[0]

    # Calcular el número de muestras por cluster como 1% del total de filas
    n_samples_per_cluster = int(sample_percentage * total_rows / n_clusters)

    # Aplicar KMeans clustering usando las columnas filtradas
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters

    # Muestrear un porcentaje de cada cluster
    # Muestrear un porcentaje de cada cluster, respetando el tamaño máximo de cada cluster
    df_kmeans_sampled = df.groupby('cluster', group_keys=False).apply(
        lambda x: x.sample(
            n=n_samples_per_cluster,  # Tomar siempre el tamaño calculado
            replace=True,  # Permitir reemplazo si hay menos filas en el cluster
            random_state=42
        )
    ).reset_index(drop=True)

    return df_kmeans_sampled


# Definir columnas a excluir para KMeans
columns_to_drop = ['Calendar_date', 'route_id', 'bus_id', 'weather', 'temperature', 'day_of_week', 'time_of_day']

# Aplicar KMeans para el muestreo basado en el 1% del total de filas
df_kmeans_sampled = submuestreo_kmeans(df_cleaned_multiple, columns_to_drop)
df_kmeans_sampled = df_kmeans_sampled.drop(columns=['cluster'])
# %%
# Exportar el DataFrame limpio como un archivo CSV
output_path = "data/Dataset-PT_KMeans.csv"
df_kmeans_sampled.to_csv(output_path, index=False)

print(f"El DataFrame ha sido exportado exitosamente a {output_path}")


# %% md
# Plots Original Vs. KMeans Distribution
# %%
def plot_distributions(df_original, df_sampled, cols):
    for col in cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_original[col], color='blue', label='Original', kde=True, stat="density")
        sns.histplot(df_sampled[col], color='orange', label='Submuestra', kde=True, stat="density")
        plt.title(f'Distribución de {col}')
        plt.legend()
        plt.show()


plot_distributions(df, df_kmeans_sampled, numeric_cols)


# %% md
# Comparaciones estadísticas
# %% md
## Comparación estadística
# %%
def comparar_estadisticas(df_original, df_sampled, numeric_cols, threshold=0.05):
    original_summary = df_original[numeric_cols].describe()
    sampled_summary = df_sampled[numeric_cols].describe()

    print("\n### Comparación de Estadísticas ###")
    for col in numeric_cols:
        mean_diff = np.abs(original_summary.loc['mean', col] - sampled_summary.loc['mean', col]) / original_summary.loc[
            'mean', col]
        print(f"{col} - Diferencia de Medias: {mean_diff:.2%} (Umbral: {threshold * 100}%)")

        # Otras comparaciones de medianas y desviaciones estándar pueden agregarse aquí


# %%
comparar_estadisticas(df, df_stratified, numeric_cols)
# %%
comparar_estadisticas(df, df_kmeans_sampled, numeric_cols)


# %% md
## T-test
# %%
def prueba_t_test(df_original, df_sampled, cols, alpha=0.05):
    print("\n### T-tests ###")
    for col in cols:
        t_stat, p_value = ttest_ind(df_original[col], df_sampled[col])
        resultado = "Significativa" if p_value < alpha else "No significativa"
        print(f"{col} - p-value: {p_value:.3f} ({resultado})")


# %%
prueba_t_test(df, df_stratified, numeric_cols)
# %%
prueba_t_test(df, df_kmeans_sampled, numeric_cols)


# %% md
## Chi-square
# %%
def prueba_chi_square(df_original, df_sampled, dummy_vars, alpha=0.05):
    print("\n### Chi-Square Tests ###")
    for var in dummy_vars:
        contingency_table = pd.crosstab(df_original[var], df_sampled[var])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        resultado = "Significativa" if p_value < alpha else "No significativa"
        print(f"{var} - p-value: {p_value:.3f} ({resultado})")


# %%
prueba_chi_square(df, df_stratified, dummy_vars)
# %%
prueba_chi_square(df, df_kmeans_sampled, dummy_vars)