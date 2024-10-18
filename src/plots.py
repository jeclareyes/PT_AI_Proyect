# plotting.py
#AI tools were used to develop and enhance this code


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os


# Global style and font configuration
sns.set(style="whitegrid")

# Global font configuration for matplotlib
plt.rcParams.update({
    'font.size': 16,              # Base font size
    'axes.titlesize': 16,         # Axis titles font size
    'axes.labelsize': 16,         # Axis labels font size
    'legend.fontsize': 16,        # Legend font size
    'xtick.labelsize': 16,        # X-axis tick labels font size
    'ytick.labelsize': 16,        # Y-axis tick labels font size
    'figure.titlesize': 16,       # Figure titles font size
    'figure.figsize': (8, 6),     # Default figure size (adjustable as needed)
    'savefig.dpi': 300,           # Figure save resolution
    'legend.title_fontsize': 16,   # Legend title font size
    'xtick.major.size': 5,         # X-axis major tick size
    'ytick.major.size': 5,         # Y-axis major tick size
    'lines.markersize': 6,         # Line markers size
    'axes.linewidth': 1.2,         # Axis lines width
    'legend.frameon': False        # No frame in the legend
})

plt.rcParams.update({'figure.max_open_warning': 0})  # Avoid warnings when opening too many figures

#%%

def load_and_prepare_data(csv_path):
    """
    Loads the CSV file and prepares the data for visualization.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Prepared DataFrame.
    """
    # First, identify columns that contain lists
    # Assume that metric columns start with 'MAE_', 'MSE_', etc.
    metric_prefixes = ['MAE_', 'MSE_', 'RMSE_', 'MAPE_', 'R2_']

    # Function to convert list strings to actual lists
    def parse_list_column(cell):
        try:
            return ast.literal_eval(cell)
        except:
            return []

    # Load the CSV
    df = pd.read_csv(csv_path, sep=';')

    # Identify columns that contain lists
    list_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in metric_prefixes)]

    # Convert list string columns to actual lists
    for col in list_columns:
        df[col] = df[col].apply(parse_list_column)

    return df

def extract_categories(categorical_vars_str, conditioning_var):
    """
    Extracts categories of a conditioning variable from the 'Categorical_Variables' column.

    Args:
        categorical_vars_str (str): String containing categorical variables.
        conditioning_var (str): Name of the conditioning variable.

    Returns:
        List[str]: List of categories.
    """
    # Categorical variables are in the format 'prefixCategory', e.g., 'stop_seq_1'
    # We want to extract categories corresponding to 'conditioning_var'

    # First, split the categorical variables by commas
    vars_list = [var.strip() for var in categorical_vars_str.split(',')]

    # Filter variables that correspond to 'conditioning_var'
    # For example, if conditioning_var='stop_sequence', look for variables that start with 'stop_seq_'
    # It is assumed that the prefix in 'Categorical_Variables' matches 'prefix_dict'

    # Define the corresponding prefix (adjust this according to your 'prefix_dict')
    prefix_mapping = {
        'stop_sequence': 'stop_seq_',
        'weather': 'factor(weather)',
        'temperature': 'factor(temperature)',
        'day_of_week': 'factor(day_of_week)',
        'time_of_day': 'factor(time_of_day)'
        # Add more if necessary
    }

    prefix = prefix_mapping.get(conditioning_var, '')

    # Filter variables that start with the prefix
    categories = [var.replace(prefix, '') for var in vars_list if var.startswith(prefix)]

    return categories

#%%
def plot_overall_metrics_by_sample_type(df, metric='MAE', output_dir='plots'):
    """
    Generates bar charts for overall metrics (MAE or R2) segmented by Sample_Type,
    comparing Models and Scenarios.

    Args:
        df (pd.DataFrame): DataFrame with the metrics.
        metric (str): Metric to plot ('MAE' or 'R2').
        output_dir (str): Directory where the plots will be saved.
    """
    # Check if the metric exists in the DataFrame
    if metric not in df.columns:
        print(f"The metric '{metric}' does not exist in the DataFrame.")
        return

    # Get unique Sample_Types
    sample_types = df['Sample_type'].unique()

    for sample_type in sample_types:
        sample_df = df[df['Sample_type'] == sample_type]

        if sample_df.empty:
            print(f"No data for Sample_Type '{sample_type}'.")
            continue

        plt.figure(figsize=(12, 8))  # Increase size to accommodate more bars

        sns.barplot(x='Model', y=metric, hue='Scenario', data=sample_df, palette='Set2', edgecolor=None)
        plt.title(f'{metric} by Model and Scenario - Sample_Type: {sample_type}')
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.legend(title='Scenario')
        plt.tight_layout()

        # Define the directory to save the plot
        plot_dir = os.path.join(output_dir, 'general_metrics', 'by_sample_type', sample_type)
        os.makedirs(plot_dir, exist_ok=True)

        # Define the filename
        plot_filename = f"{metric}_by_Model_and_Scenario.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        # Save the plot
        plt.savefig(plot_path, dpi=300)  # Increase resolution
        plt.close()
        print(f"Plot '{plot_filename}' saved in '{plot_dir}'.")

def plot_overall_metrics_by_model(df, metric='MAE', output_dir='plots'):
    """
    Generates bar charts for overall metrics (MAE or R2) segmented by Model,
    comparing Sample_Types and Scenarios.

    Args:
        df (pd.DataFrame): DataFrame with the metrics.
        metric (str): Metric to plot ('MAE' or 'R2').
        output_dir (str): Directory where the plots will be saved.
    """
    # Check if the metric exists in the DataFrame
    if metric not in df.columns:
        print(f"The metric '{metric}' does not exist in the DataFrame.")
        return

    # Get unique Models
    models = df['Model'].unique()

    for model in models:
        model_df = df[df['Model'] == model]

        if model_df.empty:
            print(f"No data for Model '{model}'.")
            continue

        plt.figure(figsize=(12, 8))

        sns.barplot(x='Sample_type', y=metric, hue='Scenario', data=model_df, palette='Set2', edgecolor=None)
        plt.title(f'{metric} by Sample_Type and Scenario - Model: {model}')
        plt.ylabel(metric)
        plt.xlabel('Sample_Type')
        plt.legend(title='Scenario')
        plt.tight_layout()

        # Define the directory to save the plot
        plot_dir = os.path.join(output_dir, 'general_metrics', 'by_model', model)
        os.makedirs(plot_dir, exist_ok=True)

        # Define the filename
        plot_filename = f"{metric}_by_Sample_Type_and_Scenario_Model_{model}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        # Save the plot
        plt.savefig(plot_path, dpi=300)  # Increase resolution
        plt.close()
        print(f"Plot '{plot_filename}' saved in '{plot_dir}'.")

def plot_overall_metrics_by_scenario(df, metric='MAE', output_dir='plots'):
    """
    Generates bar charts for overall metrics (MAE or R2) segmented by Scenario,
    comparing Models and Sample_Types.

    Args:
        df (pd.DataFrame): DataFrame with the metrics.
        metric (str): Metric to plot ('MAE' or 'R2').
        output_dir (str): Directory where the plots will be saved.
    """
    # Check if the metric exists in the DataFrame
    if metric not in df.columns:
        print(f"The metric '{metric}' does not exist in the DataFrame.")
        return

    # Get unique Scenarios
    scenarios = df['Scenario'].unique()

    for scenario in scenarios:
        scenario_df = df[df['Scenario'] == scenario]

        if scenario_df.empty:
            print(f"No data for Scenario '{scenario}'.")
            continue

        plt.figure(figsize=(12, 8))

        sns.barplot(x='Model', y=metric, hue='Sample_type', data=scenario_df, palette='Set2', edgecolor=None)
        plt.title(f'{metric} by Model and Sample_Type - Scenario: {scenario}')
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.legend(title='Sample_type')
        plt.tight_layout()

        # Define the directory to save the plot
        plot_dir = os.path.join(output_dir, 'general_metrics', 'by_scenario', scenario)
        os.makedirs(plot_dir, exist_ok=True)

        # Define the filename
        plot_filename = f"{metric}_by_Model_and_Sample_Type_Scenario_{scenario}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        # Save the plot
        plt.savefig(plot_path, dpi=300)  # Increase resolution
        plt.close()
        print(f"Plot '{plot_filename}' saved in '{plot_dir}'.")

#%%
def plot_conditioned_metric_hue_scenario(df, metric='MAE', conditioning_var='stop_sequence', output_dir='plots'):
    """
    Generates bar charts for metrics conditioned on a categorical variable,
    segmented by Sample_Type and Model, comparing Scenarios.

    Args:
        df (pd.DataFrame): DataFrame with the metrics.
        metric (str): Metric to plot ('MAE', 'MSE', etc.).
        conditioning_var (str): Conditioning categorical variable.
        output_dir (str): Directory where the plots will be saved.
    """
    # Identify the conditioned column, e.g., 'MAE_stop_sequence'
    conditioned_col = f"{metric}_{conditioning_var}"

    if conditioned_col not in df.columns:
        print(f"The column '{conditioned_col}' does not exist in the DataFrame.")
        return

    # Get unique Sample_Types
    sample_types = df['Sample_type'].unique()

    for sample_type in sample_types:
        sample_df = df[df['Sample_type'] == sample_type]

        if sample_df.empty:
            print(f"No data for Sample_Type '{sample_type}'.")
            continue

        # Filter rows that have data for this conditioned metric
        conditioned_df = sample_df[['Scenario', 'Categorical_Variables', conditioned_col]].dropna()

        if conditioned_df.empty:
            print(f"No data for the conditioned metric '{conditioned_col}' in Sample_Type '{sample_type}'.")
            continue

        # Extract categories
        conditioned_df['Categories'] = conditioned_df['Categorical_Variables'].apply(
            lambda x: extract_categories(x, conditioning_var))

        # Explore the lists and create a new DataFrame where each row corresponds to a category
        data_records = []
        for _, row in conditioned_df.iterrows():
            scenario = row['Scenario']
            categories = row['Categories']
            metric_values = row[conditioned_col]

            # Check if the length of metric_values matches the number of categories
            if len(metric_values) != len(categories):
                print(f"{metric_values} taken for scenario: '{scenario}' and '{sample_type}'.")
                # print(f"Warning: The length of '{conditioned_col}' does not match the categories for Scenario '{scenario}' and Sample_Type '{sample_type}'.")
                continue

            for category, value in zip(categories, metric_values):
                data_records.append({
                    'Scenario': scenario,
                    'Category': category,
                    'Metric_Value': value
                })

        # Create a DataFrame from the records
        plot_df = pd.DataFrame(data_records)

        if plot_df.empty:
            print(f"No valid data for the conditioned metric '{conditioned_col}' in Sample_Type '{sample_type}'.")
            continue

        # Order categories numerically if conditioning_var is 'stop_sequence'
        if conditioning_var == 'stop_sequence':
            # Convert categories to integers for sorting
            try:
                plot_df['Category'] = plot_df['Category'].astype(int)
                plot_df = plot_df.sort_values(by='Category')
                ordered_categories = sorted(plot_df['Category'].unique())
            except ValueError:
                print(
                    f"Could not convert categories of '{conditioning_var}' to integers in Sample_Type '{sample_type}'. Ensure they are numeric.")
                ordered_categories = None
        else:
            ordered_categories = None  # Do not apply specific order. I did this because I was getting a warning

        # Create the plot
        plt.figure(figsize=(20, 8))

        if conditioning_var == 'stop_sequence' and ordered_categories:
            sns.barplot(x='Category', y='Metric_Value', hue='Scenario', data=plot_df, palette='Set2', edgecolor=None,
                        order=ordered_categories, errorbar=None)
        else:
            sns.barplot(x='Category', y='Metric_Value', hue='Scenario', data=plot_df, palette='Set2', edgecolor=None, errorbar=None)

        plt.title(f'{metric} by Model and Scenario - Sample_Type: {sample_type}')
        plt.ylabel(metric)
        plt.xlabel(conditioning_var.capitalize())
        plt.legend(title='Scenario')
        plt.xticks(rotation=90)  # Rotate labels for better readability
        plt.tight_layout()

        # Define the directory to save the plot
        plot_dir = os.path.join(output_dir, 'conditioned_metrics', 'by_sample_type', sample_type)
        os.makedirs(plot_dir, exist_ok=True)

        # Define the filename
        plot_filename = f"{metric}_{conditioning_var}_by_Model_and_Scenario_Sample_Type_{sample_type}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        # Save the plot
        plt.savefig(plot_path, dpi=300)  # Increase resolution
        plt.close()
        print(f"Plot '{plot_filename}' saved in '{plot_dir}'.")

def plot_conditioned_metric_hue_model(df, metric='MAE', conditioning_var='stop_sequence', output_dir='plots'):
    """
    Generates bar charts for metrics conditioned on a categorical variable,
    segmented by Model, comparing Sample_Types and Scenarios.

    Args:
        df (pd.DataFrame): DataFrame with the metrics.
        metric (str): Metric to plot ('MAE', 'MSE', etc.).
        conditioning_var (str): Conditioning categorical variable.
        output_dir (str): Directory where the plots will be saved.
    """
    # Identify the conditioned column, e.g., 'MAE_stop_sequence'
    conditioned_col = f"{metric}_{conditioning_var}"

    if conditioned_col not in df.columns:
        print(f"The column '{conditioned_col}' does not exist in the DataFrame.")
        return

    # Get unique Models
    models = df['Model'].unique()

    for model in models:
        model_df = df[df['Model'] == model]

        if model_df.empty:
            print(f"No data for Model '{model}'.")
            continue

        # Get unique Sample_Types
        sample_types = model_df['Sample_type'].unique()

        for sample_type in sample_types:
            sample_df = model_df[model_df['Sample_type'] == sample_type]

            if sample_df.empty:
                print(f"No data for Model '{model}' and Sample_Type '{sample_type}'.")
                continue

            # Filter rows that have data for this conditioned metric
            conditioned_df = sample_df[['Scenario', 'Categorical_Variables', conditioned_col]].dropna()

            if conditioned_df.empty:
                print(f"No data for the conditioned metric '{conditioned_col}' in Model '{model}' and Sample_Type '{sample_type}'.")
                continue

            # Extract categories
            conditioned_df['Categories'] = conditioned_df['Categorical_Variables'].apply(
                lambda x: extract_categories(x, conditioning_var))

            # Explore the lists and create a new DataFrame where each row corresponds to a category
            data_records = []
            for _, row in conditioned_df.iterrows():
                scenario = row['Scenario']
                categories = row['Categories']
                metric_values = row[conditioned_col]

                # Check if the length of metric_values matches the number of categories
                if len(metric_values) != len(categories):
                    print(
                        f"Warning: The length of '{conditioned_col}' does not match the categories for Scenario '{scenario}', Model '{model}', Sample_Type '{sample_type}'.")
                    continue

                for category, value in zip(categories, metric_values):
                    data_records.append({
                        'Scenario': scenario,
                        'Category': category,
                        'Metric_Value': value
                    })

            # Create a DataFrame from the records
            plot_df = pd.DataFrame(data_records)

            if plot_df.empty:
                print(f"No valid data for the conditioned metric '{conditioned_col}' in Model '{model}' and Sample_Type '{sample_type}'.")
                continue

            # Order categories numerically if conditioning_var is 'stop_sequence'
            if conditioning_var == 'stop_sequence':
                # Convert categories to integers for sorting
                try:
                    plot_df['Category'] = plot_df['Category'].astype(int)
                    plot_df = plot_df.sort_values(by='Category')
                    ordered_categories = sorted(plot_df['Category'].unique())
                except ValueError:
                    print(
                        f"Could not convert categories of '{conditioning_var}' to integers in Model '{model}' and Sample_Type '{sample_type}'. Ensure they are numeric.")
                    ordered_categories = None
            else:
                ordered_categories = None  # Do not apply specific order. I did this because I was getting a warning

            # Create the plot
            plt.figure(figsize=(20, 8))

            if conditioning_var == 'stop_sequence' and ordered_categories:
                sns.barplot(x='Category', y='Metric_Value', hue='Scenario', data=plot_df, palette='Set2', edgecolor=None,
                            order=ordered_categories, errorbar=None)
            else:
                sns.barplot(x='Category', y='Metric_Value', hue='Scenario', data=plot_df, palette='Set2', edgecolor=None, errorbar=None)

            plt.title(f'{metric} by Sample_Type and Scenario - Model: {model}, Sample_Type: {sample_type}')
            plt.ylabel(metric)
            plt.xlabel(conditioning_var.capitalize())
            plt.legend(title='Scenario')
            plt.xticks(rotation=90)  # Rotate labels for better readability
            plt.tight_layout()

            # Define the directory to save the plot
            plot_dir = os.path.join(output_dir, 'conditioned_metrics', 'by_model', model)
            os.makedirs(plot_dir, exist_ok=True)

            # Define the filename
            plot_filename = f"{metric}_{conditioning_var}_by_Sample_Type_and_Scenario_Model_{model}_Sample_Type_{sample_type}.png"
            plot_path = os.path.join(plot_dir, plot_filename)

            # Save the plot
            plt.savefig(plot_path, dpi=300)  # Increase resolution
            plt.close()
            print(f"Plot '{plot_filename}' saved in '{plot_dir}'.")

#%%
def plot_conditioned_metric_by_scenario_sample_type(df, metric='MAE', conditioning_var='stop_sequence', output_dir='plots'):
    """
    Generates bar charts for metrics conditioned on a categorical variable,
    with Series as Sample_Type, for a fixed Scenario.

    Args:
        df (pd.DataFrame): DataFrame with the metrics.
        metric (str): Metric to plot ('MAE', 'MSE', etc.).
        conditioning_var (str): Conditioning categorical variable.
        output_dir (str): Directory where the plots will be saved.
    """
    # Identify the conditioned column, e.g., 'MAE_stop_sequence'
    conditioned_col = f"{metric}_{conditioning_var}"

    if conditioned_col not in df.columns:
        print(f"The column '{conditioned_col}' does not exist in the DataFrame.")
        return

    # Get unique Scenarios
    scenarios = df['Scenario'].unique()

    for scenario in scenarios:
        scenario_df = df[df['Scenario'] == scenario]

        if scenario_df.empty:
            print(f"No data for Scenario '{scenario}'.")
            continue

        # Filter rows that have data for this conditioned metric
        conditioned_df = scenario_df[['Sample_type', 'Categorical_Variables', conditioned_col]].dropna()

        if conditioned_df.empty:
            print(f"No data for the conditioned metric '{conditioned_col}' in Scenario '{scenario}'.")
            continue

        # Extract categories
        conditioned_df['Categories'] = conditioned_df['Categorical_Variables'].apply(
            lambda x: extract_categories(x, conditioning_var))

        # Explore the lists and create a new DataFrame where each row corresponds to a category
        data_records = []
        for _, row in conditioned_df.iterrows():
            sample_type = row['Sample_type']
            categories = row['Categories']
            metric_values = row[conditioned_col]

            # Check if the length of metric_values matches the number of categories
            if len(metric_values) != len(categories):
                print(
                    f"Warning: The length of '{conditioned_col}' does not match the categories for Scenario '{scenario}' and Sample_Type '{sample_type}'.")
                continue

            for category, value in zip(categories, metric_values):
                data_records.append({
                    'Sample_type': sample_type,
                    'Category': category,
                    'Metric_Value': value
                })

        # Create a DataFrame from the records
        plot_df = pd.DataFrame(data_records)

        if plot_df.empty:
            print(f"No valid data for the conditioned metric '{conditioned_col}' in Scenario '{scenario}'.")
            continue

        # Order categories numerically if conditioning_var is 'stop_sequence'
        if conditioning_var == 'stop_sequence':
            # Convert categories to integers for sorting
            try:
                plot_df['Category'] = plot_df['Category'].astype(int)
                plot_df = plot_df.sort_values(by='Category')
                ordered_categories = sorted(plot_df['Category'].unique())
            except ValueError:
                print(
                    f"Could not convert categories of '{conditioning_var}' to integers in Scenario '{scenario}'. Ensure they are numeric.")
                ordered_categories = None
        else:
            ordered_categories = None  # Do not apply specific order

        # Create the plot
        plt.figure(figsize=(20, 8))  # Increase width to accommodate many categories

        if conditioning_var == 'stop_sequence' and ordered_categories:
            sns.barplot(x='Category', y='Metric_Value', hue='Sample_type', data=plot_df, palette='Set2', edgecolor=None,
                        order=ordered_categories, errorbar=None)
        else:
            sns.barplot(x='Category', y='Metric_Value', hue='Sample_type', data=plot_df, palette='Set2', edgecolor=None, errorbar=None)

        plt.title(f'{metric} by Sample_Type - {conditioning_var.capitalize()} - Scenario: {scenario}')
        plt.ylabel(metric)
        plt.xlabel(conditioning_var.capitalize())
        plt.legend(title='Sample_Type')
        plt.xticks(rotation=90)  # Rotate labels for better readability
        plt.tight_layout()

        # Define the directory to save the plot
        plot_dir = os.path.join(output_dir, 'conditioned_metrics', 'by_scenario', scenario)
        os.makedirs(plot_dir, exist_ok=True)

        # Define the filename
        plot_filename = f"{metric}_{conditioning_var}_by_Sample_Type.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        # Save the plot
        plt.savefig(plot_path, dpi=300)  # Increase resolution
        plt.close()
        print(f"Plot '{plot_filename}' saved in '{plot_dir}'.")

def plot_conditioned_metric_by_scenario_model(df, metric='MAE', conditioning_var='stop_sequence', output_dir='plots'):
    """
    Generates bar charts for metrics conditioned on a categorical variable,
    with Series as Model, for a fixed Scenario.

    Args:
        df (pd.DataFrame): DataFrame with the metrics.
        metric (str): Metric to plot ('MAE', 'MSE', etc.).
        conditioning_var (str): Conditioning categorical variable.
        output_dir (str): Directory where the plots will be saved.
    """
    # Identify the conditioned column, e.g., 'MAE_stop_sequence'
    conditioned_col = f"{metric}_{conditioning_var}"

    if conditioned_col not in df.columns:
        print(f"The column '{conditioned_col}' does not exist in the DataFrame.")
        return

    # Get unique Scenarios
    scenarios = df['Scenario'].unique()

    for scenario in scenarios:
        scenario_df = df[df['Scenario'] == scenario]

        if scenario_df.empty:
            print(f"No data for Scenario '{scenario}'.")
            continue

        # Filter rows that have data for this conditioned metric
        conditioned_df = scenario_df[['Model', 'Categorical_Variables', conditioned_col]].dropna()

        if conditioned_df.empty:
            print(f"No data for the conditioned metric '{conditioned_col}' in Scenario '{scenario}'.")
            continue

        # Extract categories
        conditioned_df['Categories'] = conditioned_df['Categorical_Variables'].apply(
            lambda x: extract_categories(x, conditioning_var))

        # Explore the lists and create a new DataFrame where each row corresponds to a category
        data_records = []
        for _, row in conditioned_df.iterrows():
            model = row['Model']
            categories = row['Categories']
            metric_values = row[conditioned_col]

            # Check if the length of metric_values matches the number of categories
            if len(metric_values) != len(categories):
                print(
                    f"Warning: The length of '{conditioned_col}' does not match the categories for Scenario '{scenario}' and Model '{model}'.")
                continue

            for category, value in zip(categories, metric_values):
                data_records.append({
                    'Model': model,
                    'Category': category,
                    'Metric_Value': value
                })

        # Create a DataFrame from the records
        plot_df = pd.DataFrame(data_records)

        if plot_df.empty:
            print(f"No valid data for the conditioned metric '{conditioned_col}' in Scenario '{scenario}'.")
            continue

        # Order categories numerically if conditioning_var is 'stop_sequence'
        if conditioning_var == 'stop_sequence':
            # Convert categories to integers for sorting
            try:
                plot_df['Category'] = plot_df['Category'].astype(int)
                plot_df = plot_df.sort_values(by='Category')
                ordered_categories = sorted(plot_df['Category'].unique())
            except ValueError:
                print(
                    f"Could not convert categories of '{conditioning_var}' to integers in Scenario '{scenario}'. Ensure they are numeric.")
                ordered_categories = None
        else:
            ordered_categories = None  # Do not apply specific order

        # Create the plot
        plt.figure(figsize=(20, 8))  # Increase width to accommodate many categories

        if conditioning_var == 'stop_sequence' and ordered_categories:
            sns.barplot(x='Category', y='Metric_Value', hue='Model', data=plot_df, palette='Set2', edgecolor=None,
                        order=ordered_categories, errorbar=None)
        else:
            sns.barplot(x='Category', y='Metric_Value', hue='Model', data=plot_df, palette='Set2', edgecolor=None, errorbar=None)

        plt.title(f'{metric} by Model - {conditioning_var.capitalize()} - Scenario: {scenario}')
        plt.ylabel(metric)
        plt.xlabel(conditioning_var.capitalize())
        plt.legend(title='Model')
        plt.xticks(rotation=90)  # Rotate labels for better readability
        plt.tight_layout()

        # Define the directory to save the plot
        plot_dir = os.path.join(output_dir, 'conditioned_metrics', 'by_scenario', scenario)
        os.makedirs(plot_dir, exist_ok=True)

        # Define the filename
        plot_filename = f"{metric}_{conditioning_var}_by_Model.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        # Save the plot
        plt.savefig(plot_path, dpi=300)  # Increase resolution
        plt.close()
        print(f"Plot '{plot_filename}' saved in '{plot_dir}'.")

def plot_conditioned_metric(df, metric='MAE', conditioning_var='stop_sequence', output_dir='plots'):
    """
    Generates all necessary conditioned metric plots:
    - By Sample_Type with Series as Scenarios.
    - By Model with Series as Scenarios.
    - By Scenario with Series as Sample_Type.
    - By Scenario with Series as Model.

    Args:
        df (pd.DataFrame): DataFrame with the metrics.
        metric (str): Metric to plot ('MAE', 'MSE', etc.).
        conditioning_var (str): Conditioning categorical variable.
        output_dir (str): Directory where the plots will be saved.
    """
    # 1. Conditioned Plots with Series as Scenarios (by Sample_Type)
    plot_conditioned_metric_hue_scenario(df, metric=metric, conditioning_var=conditioning_var, output_dir=output_dir)

    # 2. Conditioned Plots with Series as Scenarios (by Model)
    plot_conditioned_metric_hue_model(df, metric=metric, conditioning_var=conditioning_var, output_dir=output_dir)

    # 3. Conditioned Plots with Series as Sample_Type (by Scenario)
    plot_conditioned_metric_by_scenario_sample_type(df, metric=metric, conditioning_var=conditioning_var, output_dir=output_dir)

    # 4. Conditioned Plots with Series as Model (by Scenario)
    # Already covered in plot_conditioned_metric_hue_model
    # Therefore, no need to duplicate it here

#%%
def main():
    # Path to the CSV file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, '..', 'metrics_1')
    csv_path = os.path.join(folder, 'evaluation_metrics.csv')  # Adjust the name and path as needed
    metrics_to_eval = ['MAE', 'R2']

    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"The file '{csv_path}' does not exist. Ensure the path is correct.")
        return

    # Load and prepare the data
    df = load_and_prepare_data(csv_path)
    print("Data loaded and prepared successfully.")

    # Create directory for plots
    output_dir = os.path.join(base_dir, '..', 'plots_3')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Bar Charts for MAE and R2 based on Model, Scenario, and Sample_Type
    for metric in metrics_to_eval:
        plot_overall_metrics_by_sample_type(df, metric=metric, output_dir=output_dir)
        plot_overall_metrics_by_model(df, metric=metric, output_dir=output_dir)
        plot_overall_metrics_by_scenario(df, metric=metric, output_dir=output_dir)  # Add this line

    # 2. Bar Charts for Conditioned Metrics
    # Identify all conditioned columns for MAE, MSE, RMSE, MAPE, R2
    conditional_prefixes = ['MAE_', 'MSE_', 'RMSE_', 'MAPE_', 'R2_']
    conditioned_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in conditional_prefixes)]

    # Extract suffixes of the conditioned columns
    conditioning_vars = [col.split('_', 1)[1] for col in conditioned_cols]
    conditioning_vars = list(set(conditioning_vars))  # Remove duplicates

    # For each conditioning variable, generate the plot
    for conditioning_var in conditioning_vars:
        for metric in metrics_to_eval:
            # Check if the conditioned metric exists
            if f"{metric}_{conditioning_var}" in df.columns:
                plot_conditioned_metric(df, metric=metric, conditioning_var=conditioning_var, output_dir=output_dir)


if __name__ == "__main__":
    main()
