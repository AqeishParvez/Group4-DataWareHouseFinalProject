import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv("TOTAL_KSI_6133740419123635124.csv")
original_data = pd.read_csv("TOTAL_KSI_6133740419123635124.csv")  # Save a copy of the original data



missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100 # Percentage of missing values
print(missing_percentage[missing_percentage > 0]) # Print columns with missing values

# Columns with Very High Missing Data (>50%): FATAL_NO (95.41%), PEDTYPE (82.97%), DISABILITY (97.40%), EMERG_VEH (99.74%).
# These columns are mostly unusable because they lack sufficient data.
# Columns with Moderate Missing Data (10-50%): VEHTYPE (18.39%), MANOEUVER (41.95%), DRIVACT (49.00%).
# These columns might be useful for modeling. We'll need to impute missing values based on the data type.
# Columns with Low Missing Data (<10%): STREET2 (8.99%), ROAD_CLASS (2.56%), TRAFFCTL (0.40%).
# These columns are likely important. Impute missing values with appropriate methods.

# Drop columns with more than 50% missing data
high_missing_columns = missing_percentage[missing_percentage > 50].index
data = data.drop(columns=high_missing_columns)

print(f"Dropped columns: {list(high_missing_columns)}")

list(data.columns) # Check the remaining columns
data.shape # Check the shape of the dataset

# Impute missing values for categorical columns with the mode
columns_to_impute = ['VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND',
                     'STREET2', 'ROAD_CLASS', 'TRAFFCTL', 'VISIBILITY',
                     'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'INVTYPE']

for col in columns_to_impute:
    mode_value = data[col].mode()[0]  # Find the most common value
    data[col].fillna(mode_value, inplace=True)

# Verify there are no more missing values
missing_values_after_imputation = data.isnull().sum()
print(missing_values_after_imputation[missing_values_after_imputation > 0])

# Critical Columns:
# DISTRICT: Location-related and has low missing values. This should be retained.
# AUTOMOBILE: Represents vehicle involvement and is relevant with manageable missing data.
# High-Missing Columns to Drop: ACCLOC and INITDIR: With nearly 30% missing values and less direct importance, we can consider dropping these.
# AG_DRIV: With nearly 50% missing values, this column likely doesn’t add much to the analysis, so I will drop this.
# Special Case - INJURY: This column has almost 47% missing values. For now we'll replace missing values with "Unknown" to retain the column. This column may be useful for analysis.

# Impute mode for DISTRICT and AUTOMOBILE
for col in ['DISTRICT', 'AUTOMOBILE']:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)

# Impute 'Unknown' for INJURY
data['INJURY'].fillna('Unknown', inplace=True)

# Drop columns with high missing values
columns_to_drop = ['ACCLOC', 'INITDIR', 'AG_DRIV']
data = data.drop(columns=columns_to_drop)

# Verify remaining missing values
missing_values_after_imputation = data.isnull().sum()
print(missing_values_after_imputation[missing_values_after_imputation > 0])

# Drop ACCNUM since it doesn't contribute to modeling
data = data.drop(columns=['ACCNUM'])

# Verify that no missing values remain
missing_values_after_imputation = data.isnull().sum()
print(missing_values_after_imputation[missing_values_after_imputation > 0])

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
print("Categorical Columns:", list(categorical_columns))

# Apply one-hot encoding to nominal columns
data = pd.get_dummies(data, columns=['TRAFFCTL', 'ROAD_CLASS'], drop_first=True)

from sklearn.preprocessing import LabelEncoder

# Apply label encoding to ordinal columns
label_encoders = {}
for col in ['INJURY']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoder for future use if needed

# Verify encoded dataset
print(data.head())
print(data.dtypes)

# Encoded Nominal Columns: Columns like TRAFFCTL and ROAD_CLASS are now represented as multiple binary columns.
# Encoded Ordinal Column (INJURY): Converted the INJURY column into numeric values (e.g., Unknown = 0, Fatal = 1, etc.).
# If we need to decode these values later, the label_encoders dictionary will allow us to reverse the transformation.
# Remaining Categorical Columns: Columns like DATE, STREET1, DISTRICT, and others are still categorical. Some, like DATE, may need special handling for time-based features but we'll address that later.

# Let's check the unique values in the remaining categorical columns
for col in categorical_columns:
    if col in data.columns:
        print(f"{col}: {data[col].nunique()} unique values")

# DATE: Contains 4128 unique values. We can extract useful features like Year, Month, or Day of Week and drop DATE after feature extraction.
# STREET1 and STREET2: Contain 1942 and 2822 unique values, respectively. These columns may be too granular for modeling. We're going to drop them.
# AUTOMOBILE: Contains only 1 unique value, so it doesn’t provide any variability or predictive value. We'll drop it.
# DISTRICT (4 unique values): Regional information could be predictive. Keep.
# VISIBILITY (8 unique values), LIGHT (9 unique values), RDSFCOND (9 unique values): These weather/road conditions are important for predicting accidents.
# ACCLASS (3 unique values): Accident classification is likely important. Retain.
# IMPACTYPE (10 unique values), DRIVCOND (10 unique values): These describe accident type and driver condition, which are crucial.
# INVAGE (21 unique values): Represents participant age, where numeric ordering is meaningful.
# VEHTYPE (32 unique values): Label encoding might be simpler.
# MANOEUVER (16 unique values), DRIVACT (13 unique values): Specific driver actions and maneuvers could hold some ordinal meaning.
# Neighborhood Columns
# HOOD_158 (159 unique values), NEIGHBOURHOOD_158 (159 unique values): Encoding 159 unique values can add too many features, which might not be helpful.
# We'll evaluate correlation with the target (IS_FATAL) before deciding to include or drop these.
# HOOD_140 (141 unique values), NEIGHBOURHOOD_140 (141 unique values): Same reasoning as above.
# DIVISION (17 unique values): Regional division likely correlates with accident patterns, making it important.

# Let's start with date
# Extract features from DATE
data['YEAR'] = pd.to_datetime(data['DATE']).dt.year
data['MONTH'] = pd.to_datetime(data['DATE']).dt.month
data['DAY_OF_WEEK'] = pd.to_datetime(data['DATE']).dt.day_name()

# Drop the DATE column
data = data.drop(columns=['DATE'])

# Now let's drop STREET1, STREET2, and AUTOMOBILE which are high cardinality and low variance
data = data.drop(columns=['STREET1', 'STREET2', 'AUTOMOBILE'])

# Now let's apply one-hot encoding to the remaining categorical columns
data = pd.get_dummies(data, columns=['DISTRICT', 'VISIBILITY', 'LIGHT', 'RDSFCOND',
                                     'ACCLASS', 'IMPACTYPE', 'DRIVCOND'], drop_first=True)

# Now for ordinal columns, we'll use label encoding
label_encoders = {}
for col in ['INVAGE', 'VEHTYPE', 'MANOEUVER', 'DRIVACT']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for potential decoding later

# Let's create the IS_FATAL Column just like we created it in the data_exploration.py file
# Recreate IS_FATAL from original data
data['IS_FATAL'] = original_data['ACCLASS'].apply(lambda x: 1 if x == 'Fatal' else 0)

print(data.columns) # Confirm that the Is_FATAL column is present

# We still had some columns that were not encoded. Let's encode them now
from sklearn.preprocessing import LabelEncoder

# Encode HOOD and NEIGHBOURHOOD columns
for col in ['HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140']:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))


# Now let's tackle the neighborhood columns
# Evaluate correlation with IS_FATAL
correlation = data[['HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'IS_FATAL']].corr()
print(correlation)

# The results show that none of the neighborhood columns have a strong correlation with IS_FATAL.
# We can drop these columns to reduce the feature space as they add little value to predictive modeling for fatal accidents.

# Drop neighborhood-related columns
data = data.drop(columns=['HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140'])

# Verify dataset shape
print(data.shape)

# Now that we have cleaned and preprocessed the data, we can consider saving it for future use.
data.to_csv("cleaned_data.csv", index=False)

# Now let's move to normalization and standardization

# Identify numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
print("Numeric Columns:", list(numeric_columns))

from sklearn.preprocessing import StandardScaler

# Standardize numeric columns
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Verify scaling
print(data[numeric_columns].mean())
print(data[numeric_columns].std())

# The means of all numeric columns are now close to 0, and the standard deviations are close to 1, indicating successful standardization.
# We can now split the data into training and testing sets for modeling.
# We'll use the IS_FATAL column as the target variable for classification.

# Separate features and target
X = data.drop(columns=['IS_FATAL'])
y = data['IS_FATAL']

from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify shapes of the splits
print(f"Training Features: {X_train.shape}, Training Labels: {y_train.shape}")
print(f"Testing Features: {X_test.shape}, Testing Labels: {y_test.shape}")


# Training Features: (15165, 82), Training Labels: (15165,)
# Testing Features: (3792, 82), Testing Labels: (3792,)
# The above output shows that we have successfully split the data into training and testing sets.

# At this point we are going to check for class imbalance
# Imbalance can lead to biased models, so we need to address it before training our model.
# We'll start with calculating the distribution of the target variable IS_FATAL in the training set.

# Check class distribution
class_distribution = y_train.value_counts(normalize=True)
print("Class Distribution in Training Set:")
print(class_distribution)

# Results showed a highly imbalanced class distribution
# Majority Class (-0.404888): ~86% of the training data.
# Minority Class (2.469818): ~14% of the training data.
# This imbalance is significant and could lead to a model that overly predicts the majority class, ignoring the minority class (IS_FATAL).

# Incorrect scaling of IS_Fatal colum could have lead to this imbalance. Let's restore the original binary labels

# Restore original binary labels in y_train and y_test
y_train = (y_train > 0).astype(int)
y_test = (y_test > 0).astype(int)

# Verify class distribution again
class_distribution = y_train.value_counts(normalize=True)
print("Corrected Class Distribution in Training Set:")
print(class_distribution)

from imblearn.over_sampling import SMOTE

# SMOTE is a popular technique for oversampling the minority class to balance class distribution.
# It generates synthetic samples for the minority class based on the existing samples.
# We'll apply SMOTE to the training data to balance the class distribution.

# SMOTE does require numerical features, so we need to ensure that all features are numeric before applying it.
# Check for non-numeric columns in X_train
non_numeric_columns = X_train.select_dtypes(include=['object']).columns
print("Non-Numeric Columns:", list(non_numeric_columns))

# Non-Numeric Columns: ['INVTYPE', 'DIVISION', 'DAY_OF_WEEK']

from sklearn.preprocessing import LabelEncoder

# Initialize a dictionary to store encoders for each column
label_encoders = {}

# Encode non-numeric columns in both X_train and X_test
for col in non_numeric_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))  # Apply the same mapping to X_test
    label_encoders[col] = le  # Save the encoder for future use

# Handle cases where X_test contains categories not seen during the training phase

for col in non_numeric_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))

    # Transform X_test safely
    unseen_label = 'Unknown'
    X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else unseen_label)
    le.classes_ = np.append(le.classes_, unseen_label)  # Add the placeholder to classes_
    X_test[col] = le.transform(X_test[col].astype(str))

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Verify class distribution after balancing
balanced_distribution = y_train_balanced.value_counts(normalize=True)
print("Balanced Class Distribution in Training Set:")
print(balanced_distribution)

#Balanced Class Distribution in Training Set:
# IS_FATAL
# 0    0.5
# 1    0.5
# Name: proportion, dtype: float64


