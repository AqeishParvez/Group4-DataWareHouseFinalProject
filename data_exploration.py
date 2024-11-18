import pandas as pd
from scipy.special import number

# Load the dataset
data = pd.read_csv("TOTAL_KSI_6133740419123635124.csv")

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Display the shape of the data
data.shape

# Describe numerical columns
print("\nSummary Statistics:")
print(data.describe())

print("\nFirst Five Rows:")
print(data.head(5))

# Display columns and data types
columns = list(data.columns)
print(columns)

dataTypes = data.dtypes
print(dataTypes)

# Display unique values for categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"Column: {col}")
    print(data[col].value_counts(), "\n")


# Calculate means, medians, and standard deviations for numerical data
means = data.mean(numeric_only=True)
medians = data.median(numeric_only=True)
stds = data.std(numeric_only=True)

# Combine into a DataFrame for review
summary_stats = pd.DataFrame({
    'Mean': means,
    'Median': medians,
    'Standard Deviation': stds
})

# Display summary statistics
print(summary_stats)

# Count and display missing values in each column
missing_values = data.isnull().sum()

# Display columns with missing values in descending order to identify columns with the most missing values
missing_values_descending = missing_values[missing_values > 0].sort_values(ascending=False)
print(missing_values_descending)

# Visualize missing data
import seaborn as sns
import matplotlib.pyplot as plt

# Plot missing values in yellow color
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data in Dataset')
plt.show()

# Calculate the percentage of missing values in each column
missing_percentage = (missing_values / len(data)) * 100

# Combine into a DataFrame for review
missing_data_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Missing %': missing_percentage
}).sort_values(by='Missing %', ascending=False)

# Display columns with missing values
print(missing_data_summary[missing_data_summary['Missing Values'] > 0])

# Testing plot distribution of a numerical column (Time)
plt.figure(figsize=(8, 4))
sns.histplot(data['TIME'], kde=True, bins=30)
plt.title("Distribution of TIME")
plt.show()

# Results showed most accidents occured between 1500 and 1600 hours and 1800 and 2000 hours
# This could indicate that accidents are more likely to occur during rush hours or late night due to reduced visibility
# further analysis could be done to determine the cause of accidents during these hours

# Create a new column that uses AC_CLASS categorical column to determine fatality
# 1 for Fatal and 0 for Non-Fatal
# new colum name: IS_FATAL
data['IS_FATAL'] = data['ACCLASS'].apply(lambda x: 1 if x == 'Fatal' else 0)

# Count only the 1s in the new column
fatal_count = data['IS_FATAL'].sum()
print(f"Number of Fatal Accidents: {fatal_count}")

# Compare ROAD_CLASS and A using a bar plot
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x='ROAD_CLASS', hue='IS_FATAL')
plt.title("Fatalities by Road Class")
plt.show()

# Results showed most fatalities occured on Major Arterial roads
# This could indicate that major arterial roads are more dangerous compared to other road classes
# further analysis could be done to determine the cause of fatalities on major arterial roads

# Box plot for Fata accidents by light condition
plt.figure(figsize=(8, 4))
sns.boxplot(data=data, x='LIGHT', y='TIME', hue='IS_FATAL')
plt.title("Fatal Accidents by Light Condition")
plt.show()

# Results showed that most fatal and non-fatal accidents occured when it was dark or dark artificial
# This confirms that accidents are more likely to occur during reduced visibility
# Further analysis could be done to determine the cause of accidents during these conditions

# Now lets check the ferequency of accidents by month
# Create a new column for month
data['MONTH'] = pd.to_datetime(data['DATE']).dt.month

# Count the number of accidents by month
monthly_accidents = data['MONTH'].value_counts().sort_index()

# Plot the number of accidents by month
plt.figure(figsize=(8, 4))
sns.barplot(x=monthly_accidents.index, y=monthly_accidents.values)
plt.title("Number of Accidents by Month")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.show()

# Results showed that most accidents occured in the month of August
# This could indicate that accidents are more likely to occur during the summer months
# Further analysis could be done to determine the cause of accidents during these months

# Now lets check the ferequency of accidents by day of the week
# Create a new column for day of the week
data['DAY_OF_WEEK'] = pd.to_datetime(data['DATE']).dt.day_name()

# Count the number of accidents by day of the week
daily_accidents = data['DAY_OF_WEEK'].value_counts()

# Plot the number of accidents by day of the week
plt.figure(figsize=(8, 4))
sns.barplot(x=daily_accidents.index, y=daily_accidents.values)
plt.title("Number of Accidents by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Accidents")
plt.show()

# Results showed that most accidents occured on Fridays
# This could indicate that accidents are more likely to occur during the weekends
# Further analysis could be done to determine the cause of accidents during these days

# Now lets check the ferequency of accidents by hour of the day
# our time is in the format of numbers such as 236, 1828, 1507 so we need to convert it to a datetime object
# Create a new column for hour of the day
data['HOUR'] = pd.to_datetime(data['TIME'].astype(str).str.zfill(4), format='%H%M').dt.hour

# Count the number of accidents by hour of the day
hourly_accidents = data['HOUR'].value_counts().sort_index()

# Plot the number of accidents by hour of the day
plt.figure(figsize=(8, 4))
sns.barplot(x=hourly_accidents.index, y=hourly_accidents.values)
plt.title("Number of Accidents by Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Accidents")
plt.show()

# Results showed that most accidents occured around 6pm with 5pm following closely
# This could indicate that accidents are more likely to occur during rush hours or late night due to reduced visibility

# Box plot for accidents by hour of the day
plt.figure(figsize=(8, 4))
sns.boxplot(data['HOUR'])
plt.title("Accidents by Hour of the Day")
plt.show()

# Create time buckets for better visualization
data['TIME_BUCKET'] = pd.cut(data['TIME'], bins=[0, 600, 1200, 1800, 2400],
                             labels=['Early Morning', 'Morning', 'Afternoon', 'Night'])
sns.countplot(data=data, x='TIME_BUCKET')
plt.title("Accidents by Time of Day")
plt.show()

# Results showed that most accidents occured in the afternoon with night following closely
# This could indicate that accidents are more likely to occur during rush hours or late night due to reduced visibility

# Now lets check fatal and non-fatal accidents by hour of the day
# Count the number of fatal and non-fatal accidents by hour of the day
hourly_fatalities = data[data['IS_FATAL'] == 1]['HOUR'].value_counts().sort_index()
hourly_non_fatalities = data[data['IS_FATAL'] == 0]['HOUR'].value_counts().sort_index()

# Plot the number of fatal accidents by hour of the day
plt.figure(figsize=(8, 4))
bar_width = 0.4
plt.bar(hourly_fatalities.index, hourly_fatalities.values, width=bar_width, color='red', label='Fatal')
plt.bar(hourly_non_fatalities.index + bar_width, hourly_non_fatalities.values, width=bar_width, color='blue', label='Non-Fatal')
for i in range(len(hourly_fatalities)):
    plt.text(hourly_fatalities.index[i], hourly_fatalities.values[i] + 10, str(hourly_fatalities.values[i]), ha='center')
    plt.text(hourly_non_fatalities.index[i] + bar_width, hourly_non_fatalities.values[i] + 10, str(hourly_non_fatalities.values[i]), ha='center')
plt.title("Fatal and Non-Fatal Accidents by Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Accidents")
plt.legend()
plt.show()

# These results painted an interesting picture of the data
# Most number of accidents occured around 5pm but most of them around 1127 were non-fatal whereas only 121 were fatal
# The second-highest number of accidents occured around 6pm. 1104 were non-fatal but fatal accidents were 173
# Interestingly the highest number of fatal accidents occured around 8 pm with 176 fatal accidents when non-fatal accidents were only 874
# This could indicate that accidents are more likely to occur during rush hours or late night due to reduced visibility
# More fatal accidents occured during the night compared to non-fatal accidents

# Now lets heck the frequency of values in key categorical columns using bar plots
# Columns for reference: ['OBJECTID', 'INDEX', 'ACCNUM', 'DATE', 'TIME', 'STREET1', 'STREET2', 'OFFSET', 'ROAD_CLASS', 'DISTRICT', 'LATITUDE', 'LONGITUDE', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY', 'FATAL_NO', 'INITDIR', 'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION', 'x', 'y', 'IS_FATAL', 'MONTH', 'DAY_OF_WEEK', 'HOUR']

# Bar plot for ROAD_CLASS
plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='ROAD_CLASS', order=data['ROAD_CLASS'].value_counts().index)
plt.title("Road Class Categories")
plt.show()

# Bar plot for TRAFFCTL
plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='TRAFFCTL', order=data['TRAFFCTL'].value_counts().index)
plt.title("Traffic Control Categories")
plt.show()

# Bar plot for VISIBILITY
plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='VISIBILITY', order=data['VISIBILITY'].value_counts().index)
plt.title("Visibility Categories")
plt.show()

# Bar plot for LIGHT
plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='LIGHT', order=data['LIGHT'].value_counts().index)
plt.title("Light Categories")
plt.show()

# Bar plot for vehicle type
plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='VEHTYPE', order=data['VEHTYPE'].value_counts().index)
plt.title("Vehicle Type Categories")
plt.show()

# Scatter plot for LATITUDE vs LONGITUDE
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='LONGITUDE', y='LATITUDE', hue='IS_FATAL')
plt.title("Accident Locations (Latitude vs Longitude)")
plt.show()

# Analyzing trends to see if fatalities are increasing or decreasing over time
data['YEAR'] = pd.to_datetime(data['DATE']).dt.year
yearly_accidents = data['YEAR'].value_counts().sort_index()

# Plot yearly trends
plt.figure(figsize=(8, 4))
sns.lineplot(x=yearly_accidents.index, y=yearly_accidents.values)
plt.title("Accident Trends by Year")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.show()

# Results showed that the number of accidents have been decreasing over the years
# This could indicate that road safety measures have been improving over time
# Further analysis could be done to determine the cause of the decrease in accidents

# Geospatial analysis of fatal and non-fatal accidents
import folium

# Create a map centered on Toronto
accident_map = folium.Map(location=[data['LATITUDE'].mean(), data['LONGITUDE'].mean()], zoom_start=12)

# Add accident locations as points
for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=3,
        color='red' if row['IS_FATAL'] == 1 else 'blue',
        fill=True
    ).add_to(accident_map)

# Display the map
accident_map.save('accident_map.html')


# Plot a histogram of a numerical column (replace 'ColumnName' with an actual column name from the dataset)
# if 'ColumnName' in data.columns:
#     data['ColumnName'].hist()
#     plt.title('Histogram of ColumnName')
#     plt.xlabel('ColumnName')
#     plt.ylabel('Frequency')
#     plt.show()
# else:
#     print("Column 'ColumnName' not found in dataset.")
#

# Create a correlation heatmap (for numerical columns only)
numerical_data = data.select_dtypes(include=['number'])
if not numerical_data.empty:
    sns.heatmap(numerical_data.corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.show()
else:
    print("No numerical columns available for correlation heatmap.")


# Finally let's check the correlation between categorical columns and fatal accidents
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
encoded_data = data.copy()

# Check for non-numeric columns in the dataset
non_numeric_columns = encoded_data.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_columns)

print(encoded_data.dtypes)

# Encode non-numeric columns using LabelEncoder
for col in non_numeric_columns:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))

categorical_columns = data.select_dtypes(include=['object']).columns

# Drop ACCLASS column as it was used to create the IS_FATAL column
categorical_columns = categorical_columns.drop('ACCLASS')

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le  # Save the encoder for inverse transformations

# Calculate the correlation between categorical columns and fatal accidents
correlations = encoded_data.corr()['IS_FATAL'].sort_values(ascending=False)

# Filter only categorical columns
categorical_correlations = correlations[categorical_columns]
print(categorical_correlations)

# Visualize correlations between categorical columns and fatal accidents in a bar plot and include count of fatal accidents on the plot
plt.figure(figsize=(10, 6))
categorical_correlations.plot(kind='bar')
plt.title("Correlation between Categorical Columns and Fatal Accidents")
plt.xlabel("Categorical Columns")
plt.ylabel("Correlation")
plt.show()


# Visualize the correlation between categorical columns and fatal accidents in a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(categorical_correlations.to_frame(), annot=True, cmap='coolwarm')
plt.title("Correlation between Categorical Columns and Fatal Accidents")
plt.show()

# Amongst the categorical columns vehicl type and cyclist have the highest correlation with fatal accidents
# This could indicate that the type of vehicle involved in an accident has a significant impact on the likelihood of fatalities




