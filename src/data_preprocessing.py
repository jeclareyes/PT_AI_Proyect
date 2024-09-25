## Load the Data
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset with low_memory set to False to avoid Dtype warnings
data_path = '../Dataset-PT.csv'
df = pd.read_csv(data_path, low_memory=False)

#%% md
# Basic Data Inspection
#%%
# Display the first few rows to understand the data
print(df.head())

# Basic statistics for numerical data
print(df.describe())

# Check for any missing values
print(df.isnull().sum())
#%% md
# Calculate Descriptive Statistics for Delays
#%%
# Focus on 'arrival_delay' column
arrival_delays = df['arrival_delay']

# Calculate mean, median, and standard deviation
print("Mean Delay:", arrival_delays.mean())
print("Median Delay:", arrival_delays.median())
print("Standard Deviation of Delays:", arrival_delays.std())

#%% md
# Visualization
#%% md
## a-Histogram of Arrival Delays
#%%
# Plot a histogram of arrival delays
plt.figure(figsize=(10, 6))
plt.hist(arrival_delays, bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Arrival Delays')
plt.xlabel('Delay (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
#%% md
## b. Delays Over Time
#%%
# Convert 'calendar_date' to datetime
df['Calendar_date'] = pd.to_datetime(df['Calendar_date'], format='%Y%m%d')

# Plot delays over time
plt.figure(figsize=(12, 6))
plt.plot(df['Calendar_date'], df['arrival_delay'], marker='o', linestyle='', markersize=2)
plt.title('Arrival Delays Over Time')
plt.xlabel('Date')
plt.ylabel('Delay (seconds)')
plt.grid(True)
plt.show()

#%% md
## c. Delays Across Different Stops
#%%

stop_delays = df.groupby('stop_sequence')['arrival_delay'].mean()

# Plot average delay for each stop
plt.figure(figsize=(14, 7))
stop_delays.plot(kind='bar', color='green')
plt.title('Average Arrival Delays by Stop')
plt.xlabel('Stop Sequence')  # Update this label accordingly
plt.ylabel('Average Delay (seconds)')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


#%% md
## d- Histogrtams
#%%
import seaborn as sns  # Import seaborn for better visualizations

# Simple box plot for arrival delays
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['arrival_delay'])
plt.title('Box Plot of Arrival Delays')
plt.xlabel('Arrival Delay (seconds)')
plt.show()

# Box plot of arrival delays grouped by day of the week
# Ensure 'day_of_week' is the correct column name for days
plt.figure(figsize=(12, 7))
sns.boxplot(x='day_of_week', y='arrival_delay', data=df)
plt.title('Arrival Delays by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Arrival Delay (seconds)')
plt.show()

# Box plot of arrival delays grouped by time of day
# Ensure 'time_of_day' is the correct column name for time categories
plt.figure(figsize=(12, 7))
sns.boxplot(x='time_of_day', y='arrival_delay', data=df)
plt.title('Arrival Delays by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Arrival Delay (seconds)')
plt.show()

#%% md
## Feature Engineering
#%%
df = df.drop(['weather','temperature','day_of_week','time_of_day'], axis=1)
#%%
corr_matrix = df.corr()
corr_matrix['arrival_delay'].sort_values(ascending=False)
#%%
columns = ['arrival_delay', 'dwell_time', 'travel_time_for_previous_section', 'scheduled_travel_time',
           'upstream_stop_delay', 'origin_delay', 'previous_bus_delay', 'previous_trip_travel_time',
           'traffic_condition', 'recurrent_delay']

# Compute the correlation matrix
corr_matrix = df[columns].corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Transportation Variables')
plt.show()

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# List of specific columns you want to include in the pair plot
selected_columns = [
    'arrival_delay', 'dwell_time', 'travel_time_for_previous_section', 'scheduled_travel_time',
    'upstream_stop_delay', 'origin_delay', 'previous_bus_delay', 'previous_trip_travel_time',
    'traffic_condition', 'recurrent_delay']


# Creating a new DataFrame with only the selected columns
x_selected = df[selected_columns]

# Setting up the visual style with Seaborn
sns.set(style="whitegrid")

# Creating the pair plot for the selected variables
pairplot = sns.pairplot(x_selected)

# Enhancing the plot with titles and labels
pairplot.fig.suptitle('Pair Plot of Selected Transportation Variables', y=1.02)  # Adjust the title and position

# Showing the plot
plt.show()

#%% md
# Point-Biserial Correlation for CatergoricaL Variables
#%%
import pandas as pd
from scipy.stats import pointbiserialr

# Assuming 'df' is your DataFrame

# List of categorical variables
categorical_vars = [
    'factor(weather)Light_Rain', 'factor(weather)Light_Snow', 'factor(weather)Normal',
    'factor(weather)Rain', 'factor(weather)Snow', 'factor(temperature)Cold',
    'factor(temperature)Extra_cold', 'factor(temperature)Normal', 'factor(day_of_week)weekday',
    'factor(day_of_week)weekend', 'factor(time_of_day)Afternoon_peak', 'factor(time_of_day)Morning_peak',
    'factor(time_of_day)Off-peak'
]

# Calculate Point-Biserial Correlation for each categorical variable with 'arrival_delay'
results = {}
for var in categorical_vars:
    correlation, p_value = pointbiserialr(df[var], df['arrival_delay'])
    results[var] = (correlation, p_value)

# Display the results
for key, value in results.items():
    print(f"{key}: Correlation = {value[0]}, P-value = {value[1]}")

#%% md
# Categorical Variables Correlation (Cramer's V)
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """Calculate Cramér's V statistic for two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))



# List of variables to analyze
variables = [
    'factor(weather)Light_Rain', 'factor(weather)Light_Snow', 'factor(weather)Normal',
    'factor(weather)Rain', 'factor(weather)Snow', 'factor(temperature)Cold',
    'factor(temperature)Extra_cold', 'factor(temperature)Normal', 'factor(day_of_week)weekday',
    'factor(day_of_week)weekend', 'factor(time_of_day)Afternoon_peak', 'factor(time_of_day)Morning_peak',
    'factor(time_of_day)Off-peak'
]

# Initialize an empty DataFrame to store Cramér's V values
cramers_v_matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)

# Calculate Cramér's V for each pair of variables
for var1 in variables:
    for var2 in variables:
        cramers_v_matrix.loc[var1, var2] = cramers_v(df[var1], df[var2])

#%%
plt.figure(figsize=(12, 10))
sns.heatmap(cramers_v_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Cramér\'s V Correlation Matrix')
plt.show()

#%%
#This is a trial to test merging to main branch