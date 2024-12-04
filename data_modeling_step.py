import pandas as pd
#from scipy.special import number

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

# Amongst the categorical columns vehicle type and cyclist have the highest correlation with fatal accidents
# This could indicate that the type of vehicle involved in an accident has a significant impact on the likelihood of fatalities

# Analyze fatality rates by ROAD_CLASS
road_class_fatalities = data.groupby('ROAD_CLASS')['IS_FATAL'].mean().sort_values(ascending=False)
print(road_class_fatalities)

# Visualize fatality rates by ROAD_CLASS
plt.figure(figsize=(10, 6))
sns.barplot(x=road_class_fatalities.index, y=road_class_fatalities.values, palette='viridis')
plt.title("Fatality Rates by Road Class")
plt.xlabel("Road Class")
plt.ylabel("Fatality Rate")
plt.show()

# Compare ROAD_CLASS and LIGHT against fatalities
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='ROAD_CLASS', hue='LIGHT', palette='viridis')
plt.title("Accidents by Road Class and Light Condition")
plt.xlabel("Road Class")
plt.ylabel("Count")
plt.legend(title="Light Condition")
plt.show()

# Now we'll move towards data modeling to predict fatal accidents
# The first step is to prepare the data for modeling

# Calculate the percentage of missing values
missing_percentage = (data.isnull().sum() / len(data)) * 100
print("Missing Percentage for each column:\n")
print(missing_percentage.sort_values(ascending=False))

# Identify columns with more than 50% missing data
columns_to_drop = missing_percentage[missing_percentage > 50].index
print("\nColumns to Drop (More than 50% missing):\n", columns_to_drop)

# Drop these columns
data = data.drop(columns=columns_to_drop)

# Fill numerical columns with median
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    data[col].fillna(data[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

print("\nMissing values after handling:\n")
print(data.isnull().sum())

# There are some missing values in Time Bucket colum which is strange as we created it from TIME column

# Check where TIME_BUCKET is missing
missing_time_bucket = data[data['TIME_BUCKET'].isnull()]
print("\nRows with Missing TIME_BUCKET:\n", missing_time_bucket[['TIME', 'TIME_BUCKET']])

# Adjust bins to include 0 explicitly
data['TIME_BUCKET'] = pd.cut(
    data['TIME'],
    bins=[-1, 600, 1200, 1800, 2400],  # Start bins at -1 to include 0
    labels=['Early Morning', 'Morning', 'Afternoon', 'Night']
)

# Check if all missing values are resolved
print("\nMissing values after recalculating TIME_BUCKET:\n", data['TIME_BUCKET'].isnull().sum())

# The issue was resolved by adjusting the bins to include 0 explicitly

# Identify numerical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
print("Numerical columns to scale:\n", numerical_columns)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object', 'category']).columns
print("\nCategorical columns to encode:\n", categorical_columns)

# Drop remaining high-cardinality columns
high_cardinality_columns = ['HOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140']
data = data.drop(columns=high_cardinality_columns)

# Apply OneHotEncoding only to remaining categorical columns
data_encoded = pd.get_dummies(data, drop_first=True)

# Check the shape of the dataset
print("\nShape before encoding:", data.shape)
print("Shape after encoding:", data_encoded.shape)

# Display a sample of the encoded data
print("\nEncoded Data Sample:\n", data_encoded.head())

# Now we'll review the encoded data and prepare it for modeling

# Display basic information about the encoded dataset
print("Encoded Dataset Information:")
print(data_encoded.info())

# Check for any remaining missing values
print("\nMissing values in encoded dataset:")
print(data_encoded.isnull().sum().sort_values(ascending=False))

# Verify unique value counts to ensure one-hot encoding consistency
categorical_columns_encoded = [col for col in data_encoded.columns if "_" in col]
unique_value_counts = data_encoded[categorical_columns_encoded].nunique().sort_values()
print("\nUnique values per categorical column:")
print(unique_value_counts)

# Identify high-cardinality columns
high_cardinality_columns = [col for col in data_encoded.columns if data_encoded[col].nunique() > 50]

print(f"High Cardinality Columns ({len(high_cardinality_columns)}):")
for col in high_cardinality_columns[:10]:  # Display top 10 as a sample
    print(f"{col}: {data_encoded[col].nunique()} unique values")

# Check the impact of removing these columns
columns_to_drop = high_cardinality_columns  # Modify this list as needed
reduced_data = data_encoded.drop(columns=columns_to_drop)

print("\nShape after removing high-cardinality columns:")
print(reduced_data.shape)

# Columns with high null rates
null_percentage = data.isnull().mean() * 100
high_null_columns = null_percentage[null_percentage > 50]
print("\nColumns with >50% null values:")
print(high_null_columns)

# High cardinality columns
high_cardinality_columns = [col for col in data.columns if data[col].nunique() > 50]
print("\nHigh Cardinality Columns:")
print({col: data[col].nunique() for col in high_cardinality_columns})

# Step 1: Drop only non-essential columns for feature engineering
columns_to_drop = [
    'Index', 'ACCNUM', 'FID', 'X', 'Y', 'LATITUDE', 'LONGITUDE',  # Identifiers
    'ACCLASS', 'INJURY', 'FATAL_NO',  # Fatality indicators
    'OFFSET', 'LOCCOORD', 'ACCLOC'  # Redundant or less useful
]
data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

# Step 2: Feature engineering
# Extract temporal features
data_cleaned['DATE'] = pd.to_datetime(data_cleaned['DATE'])
data_cleaned['YEAR'] = data_cleaned['DATE'].dt.year
data_cleaned['MONTH'] = data_cleaned['DATE'].dt.month
data_cleaned['DAY_OF_WEEK'] = data_cleaned['DATE'].dt.day_name()
data_cleaned['HOUR'] = pd.to_datetime(data_cleaned['TIME'].astype(str).str.zfill(4), format='%H%M').dt.hour

# Drop DATE after extracting temporal features
data_cleaned = data_cleaned.drop(columns=['DATE', 'TIME'])

# Simplify STREET1 and STREET2
def categorize_street(street):
    if 'AVE' in street or 'ST' in street or 'RD' in street:
        return 'Major Road'
    elif 'DR' in street or 'CT' in street or 'PL' in street:
        return 'Residential'
    else:
        return 'Other'

data_cleaned['STREET1_CATEGORY'] = data_cleaned['STREET1'].apply(categorize_street)
data_cleaned['STREET2_CATEGORY'] = data_cleaned['STREET2'].apply(categorize_street)

# Drop original STREET1 and STREET2 columns
data_cleaned = data_cleaned.drop(columns=['STREET1', 'STREET2'])

# Step 3: Encode categorical variables
categorical_columns = data_cleaned.select_dtypes(include=['object', 'category']).columns
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

# Step 4: Display the cleaned dataset
print("\nCleaned and Encoded Dataset Info:")
print(data_encoded.info())

# Show a sample of the encoded dataset
print("\nSample of Encoded Dataset:")
print(data_encoded.head())

# Check the column names after encoding
print("\nColumn Names after Encoding:")
print(data_encoded.columns)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = data_encoded.drop(columns=['IS_FATAL'])
y = data_encoded['IS_FATAL']

# Balance the dataset
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Display the shape of the balanced dataset
print("\nBalanced Dataset Shape:")
print(f"X: {X_balanced.shape}, y: {y_balanced.shape}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_columns = X.select_dtypes(include=['float64', 'int32', 'int64']).columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Display the shapes of the resulting datasets
print("\nShapes of Datasets:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# Now we'll train a machine learning model to predict fatal accidents
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Adjust class weights to focus on fatal cases
model = LogisticRegression(max_iter=1000, random_state=42, class_weight={0: 1, 1: 2})
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print("\nClassification Report (with Class Weights):")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Overall Accuracy 90%: A strong performance, showing the model classifies most samples correctly.
# Class-Specific Metrics:
    # Class 0 (Non-Fatal):
    # Precision: 0.85 – Out of all predicted non-fatal accidents, 85% were correctly classified.
    # Recall: 0.98 – The model captured 98% of actual non-fatal cases.
    # F1-Score: 0.91 – A good balance between precision and recall.
# Class 1 (Fatal):
    # Precision: 0.98 – Out of all predicted fatal accidents, 98% were correctly classified.
    # Recall: 0.82 – The model captured 82% of actual fatal cases.
    # F1-Score: 0.89 – Strong performance, though recall can improve.
# Macro and Weighted Averages:
# Macro Average: Accounts for both classes equally; shows a balanced performance.
# Weighted Average: Accounts for class imbalance; aligns with overall accuracy.

# Confusion Matrix:
# True Positives (Fatal): 2,647 – Correctly predicted fatal accidents.
# True Negatives (Non-Fatal): 3,241 – Correctly predicted non-fatal accidents.
# False Positives (Fatal): 50 – Non-fatal cases incorrectly predicted as fatal.
# False Negatives (Non-Fatal): 577 – Fatal cases missed by the model.

# Let's do a feature importance analysis to understand the model better
import pandas as pd
import numpy as np

# Extract feature coefficients
coefficients = model.coef_[0]

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': coefficients,
    'Importance': np.abs(coefficients)
}).sort_values(by='Importance', ascending=False)

# Display top features
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

print("\nTop 10 Least Important Features:")
print(feature_importance.tail(10))

# Visualize feature importance (optional)
import matplotlib.pyplot as plt

top_features = feature_importance.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='blue')
plt.gca().invert_yaxis()
plt.title("Top 10 Most Important Features")
plt.xlabel("Coefficient Magnitude")
plt.show()

# Define features to drop based on importance
low_importance_features = [
    'MONTH', 'DIVISION_D12', 'VEHTYPE_Off Road - Other',
    'VEHTYPE_Other Emergency Vehicle', 'INVTYPE_Cyclist Passenger',
    'INVTYPE_Pedestrian - Not Hit', 'INVTYPE_Trailer Owner',
    'INVTYPE_Moped Passenger', 'LIGHT_Dawn, artificial',
    'DRIVACT_Wrong Way on One Way Road'
]

# Drop these features
X_reduced = X_train.drop(columns=low_importance_features, errors='ignore')
X_test_reduced = X_test.drop(columns=low_importance_features, errors='ignore')

# Retrain and evaluate the model
model_reduced = LogisticRegression(max_iter=1000, random_state=42)
model_reduced.fit(X_reduced, y_train)
y_pred_reduced = model_reduced.predict(X_test_reduced)

# Evaluate the reduced model
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report (Reduced Features):")
print(classification_report(y_test, y_pred_reduced))
print("\nConfusion Matrix (Reduced Features):")
print(confusion_matrix(y_test, y_pred_reduced))

# No Loss in Performance: Dropping the low-importance features did not affect the model's predictive ability.
# This indicates that the removed features were truly redundant and contributed little to the predictions.
# Simpler Model: By reducing the feature set, the model is now easier to interpret and train, and it may generalize better to new data.

# Let's see if using a decision tree model can improve the performance
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialize and train the Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, class_weight={0: 1, 1: 2})
dt_model.fit(X_reduced, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test_reduced)

# Evaluate the Decision Tree model
print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt))

print("\nConfusion Matrix (Decision Tree):")
print(confusion_matrix(y_test, y_pred_dt))

# Optional: Visualize the Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X_reduced.columns, class_names=['Non-Fatal', 'Fatal'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Analysis of Decision Tree Results:
# Classification Metrics:
    # Overall Accuracy: 70%
    # The Decision Tree performs worse than Logistic Regression, which had accuracies around 90%.
    # Class-Specific Metrics:
        # Class 0 (Non-Fatal):
        # Precision: 0.81 – When predicting non-fatal accidents, the model is fairly reliable.
        # Recall: 0.52 – The model misses a significant portion of actual non-fatal cases.
        # Class 1 (Fatal):
        # Precision: 0.64 – Less reliable when predicting fatal cases.
        # Recall: 0.87 – The model captures most fatal accidents, which is good.
        # Weighted Average:
        # The overall precision, recall, and F1-score are lower than other models, indicating a less effective prediction.
# Confusion Matrix:
    # True Positives (Fatal): 2,812 – The model correctly identifies most fatal accidents.
    # False Negatives (Fatal): 412 – Fewer fatal cases are missed compared to Logistic Regression.
    # True Negatives (Non-Fatal): 1,726 – The model correctly identifies fewer non-fatal accidents.
    # False Positives (Non-Fatal): 1,565 – High misclassification of non-fatal cases as fatal.

# Let's try a Random Forest model to see if it can improve the performance
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight={0: 1, 1: 2})
rf_model.fit(X_reduced, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_reduced)

# Evaluate the Random Forest model
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

# Analysis of Random Forest Results:
# Classification Metrics:
    # Overall Accuracy: 76%
    # The Random Forest performs better than the Decision Tree (70%) but is still lower than Logistic Regression (90%).
    # Class-Specific Metrics:
        # Class 0 (Non-Fatal):
        # Precision: 0.93 – The model is highly precise when predicting non-fatal accidents, though recall is low.
        # Recall: 0.56 – Many non-fatal cases are misclassified as fatal.
        # Class 1 (Fatal):
        # Precision: 0.68 – The model's predictions for fatal cases are less precise.
        # Recall: 0.96 – The model captures most fatal accidents, which is excellent.
        # F1-Score:
        # Non-Fatal: 0.70
        # Fatal: 0.80
        # Overall, the Random Forest strongly prioritizes recall for fatal accidents.
# Confusion Matrix:
    # True Positives (Fatal): 3,088 – The model correctly identifies most fatal accidents.
    # False Negatives (Fatal): 136 – Fewer fatal cases are missed compared to other models.
    # True Negatives (Non-Fatal): 1,843 – Non-fatal cases correctly classified.
    # False Positives (Non-Fatal): 1,448 – Many non-fatal cases are incorrectly predicted as fatal.

# Let's try grid search to find the best hyperparameters for the Random Forest model
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}]
}

# Perform Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1_weighted'
)
grid_search.fit(X_reduced, y_train)

# Best parameters and evaluation
print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_reduced)

# Extract the expected feature names
feature_names = X_reduced.columns
print("\nExpected Features for the Random Forest Model:")
print(feature_names.tolist())


print("\nClassification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_best_rf))
print("\nConfusion Matrix (Tuned Random Forest):")
print(confusion_matrix(y_test, y_pred_best_rf))

# The hyperparameter tuning has significantly improved the model's balance between precision and recall for both classes.

# Overall Accuracy: 89%
# This is comparable to the Logistic Regression model, showing that the tuning balanced precision and recall effectively.
# Class-Specific Metrics:
# Class 0 (Non-Fatal):
    # Precision: 0.93 – The model is highly reliable when predicting non-fatal accidents.
    # Recall: 0.84 – The model captures most non-fatal cases but misses some.
    # F1-Score: 0.88 – Balanced performance for non-fatal cases.
    # Class 1 (Fatal):
    # Precision: 0.85 – Slight improvement over previous models.
    # Recall: 0.94 – Excellent performance in identifying fatal accidents, minimizing false negatives.
    # F1-Score: 0.89 – Strong overall performance for fatal cases.
    # Macro and Weighted Averages:
    # Both are balanced at 0.89, indicating that the model performs equally well across both classes.
# Confusion Matrix:
    # True Positives (Fatal): 3,030 – The model correctly identifies most fatal accidents.
    # False Negatives (Fatal): 194 – A significant reduction in missed fatal cases compared to previous models.
    # True Negatives (Non-Fatal): 2,749 – High number of non-fatal cases correctly classified.
    # False Positives (Non-Fatal): 542 – A moderate improvement in reducing false alarms.

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Logistic Regression ROC and AUC
y_prob_lr = model.predict_proba(X_test)[:, 1]  # Predicted probabilities for the positive class
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Tuned Random Forest ROC and AUC
y_prob_rf = best_rf_model.predict_proba(X_test_reduced)[:, 1]  # Predicted probabilities for the positive class
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot the ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Chance Level (AUC = 0.50)')

plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Logistic Regression AUC = 0.95: Indicates strong performance but slightly lower than Random Forest.
# Random Forest AUC = 0.97: Slightly better at distinguishing between the two classes, showcasing improved model performance.

# ROC Curve Shape:
# Random Forest:
# The curve is closer to the top-left corner, meaning it achieves a better balance between True Positive Rate (TPR) and False Positive Rate (FPR).

# Logistic Regression:
# Still performs well but doesn’t achieve as steep an ascent as Random Forest in the early phase of the curve.
# Final Model: The Tuned Random Forest is the better choice due to:
# Higher AUC value (0.97 vs. 0.95).
# Superior ability to capture fatal accidents (class 1) with high recall (94%).


# The number of features is very high and it is difficult to interpret the model so we'll try to reduce the number of features
# Feature importance from Random Forest
import pandas as pd

importance = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_reduced.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Display the most important features
print("\nTop 20 Features by Importance:")
print(feature_importance_df.head(20))

# Optional: Visualize feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'].head(20), feature_importance_df['Importance'].head(20))
plt.gca().invert_yaxis()
plt.title('Top 20 Feature Importances')
plt.show()

# We'll drop features with low importance
threshold = 0.01
important_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature']

# Subset dataset to only include important features
X_reduced_important = X_reduced[important_features]
X_test_reduced_important = X_test_reduced[important_features]

# Print the number of features kept
print(f"Reduced to {len(important_features)} features from {X_reduced.shape[1]}")

# Retrain Random Forest with important features
best_rf_model.fit(X_reduced_important, y_train)
y_pred_rf_reduced = best_rf_model.predict(X_test_reduced_important)

# Evaluate
print("\nClassification Report (Reduced Features):")
print(classification_report(y_test, y_pred_rf_reduced))
print("\nConfusion Matrix (Reduced Features):")
print(confusion_matrix(y_test, y_pred_rf_reduced))

# Improved Generalization:
# Reducing irrelevant features minimizes noise and enhances the model’s ability to generalize to new data.
# Faster Inference:
# Fewer features mean faster predictions, which is beneficial in production environments.

# Recheck the features used in the model
revised_features = X_reduced_important.columns.tolist()

print("\nFinal Features Used in the Model:")
print(revised_features)
print(f"\nNumber of Features: {len(revised_features)}")

# Drop potential identifier features
identifier_features = ['INDEX', 'OBJECTID', 'x', 'y']
X_final = X_reduced_important.drop(columns=identifier_features, errors='ignore')
X_test_final = X_test_reduced_important.drop(columns=identifier_features, errors='ignore')

# Retrain and evaluate
best_rf_model.fit(X_final, y_train)
y_pred_final = best_rf_model.predict(X_test_final)

# Evaluate the model
print("\nClassification Report (Without Identifiers):")
print(classification_report(y_test, y_pred_final))
print("\nConfusion Matrix (Without Identifiers):")
print(confusion_matrix(y_test, y_pred_final))

# Recheck the features used in the model
final_features = X_final.columns.tolist()

print("\nFinal Features Used in the Model:")
print(final_features)
print(f"\nNumber of Features: {len(final_features)}")

# Recalculate feature importance for the reduced model
import pandas as pd
import matplotlib.pyplot as plt

importance = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_final.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Display the most important features
print("\nFeature Importance (Reduced Features):")
print(feature_importance_df)

# Visualize the top 10 features
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'].head(10), feature_importance_df['Importance'].head(10))
plt.gca().invert_yaxis()
plt.title('Top 10 Feature Importances (Reduced Features)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Define threshold for low importance
low_importance_threshold = 0.02
low_importance_features = feature_importance_df[feature_importance_df['Importance'] < low_importance_threshold]['Feature']

# Drop low-importance features
X_final_reduced = X_final.drop(columns=low_importance_features, errors='ignore')
X_test_final_reduced = X_test_final.drop(columns=low_importance_features, errors='ignore')

# Retrain the model
best_rf_model.fit(X_final_reduced, y_train)
y_pred_reduced = best_rf_model.predict(X_test_final_reduced)

# Evaluate the model
print("\nClassification Report (After Removing Low-Importance Features):")
print(classification_report(y_test, y_pred_reduced))
print("\nConfusion Matrix (After Removing Low-Importance Features):")
print(confusion_matrix(y_test, y_pred_reduced))

# Check the final features used in the model
final_reduced_features = X_final_reduced.columns.tolist()

print("\nFinal Features Used in the Model (Reduced):")
print(final_reduced_features)
print(f"\nNumber of Features: {len(final_reduced_features)}")

# Let's save the best model for future use
import joblib

# Save the tuned model
joblib.dump(best_rf_model, 'tuned_random_forest_model.pkl')
print("Model saved successfully!")
