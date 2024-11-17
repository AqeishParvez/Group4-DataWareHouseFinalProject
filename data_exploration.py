import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("TOTAL_KSI_6133740419123635124.csv")

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

print("\nFirst Five Rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Describe numerical columns
print("\nSummary Statistics:")
print(data.describe())

# Plot a histogram of a numerical column (replace 'ColumnName' with an actual column name from the dataset)
if 'ColumnName' in data.columns:
    data['ColumnName'].hist()
    plt.title('Histogram of ColumnName')
    plt.xlabel('ColumnName')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Column 'ColumnName' not found in dataset.")

# Create a correlation heatmap (for numerical columns only)
numerical_data = data.select_dtypes(include=['number'])
if not numerical_data.empty:
    sns.heatmap(numerical_data.corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.show()
else:
    print("No numerical columns available for correlation heatmap.")