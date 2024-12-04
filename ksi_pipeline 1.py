from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import DecisionTreeClassifier

class KSIFatalityPipeline:
    
    def __init__(self, missing_threshold, columns_to_drop, model_name, selected_features, use_selected_features, feature_scope):
        self.data = None
        self.columns_to_drop = columns_to_drop
        self.missing_threshold = missing_threshold
        self.processor = None
        self.model = None
        self.loaded_model = None
        self.model_name = model_name
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "DTC": DecisionTreeClassifier(max_depth=10)
        }  
        self.scaler = StandardScaler()
        self.features = None
        self.SELECTED_FEATURES = selected_features
        self.feature_scope = feature_scope
        self.use_selected_features = use_selected_features
        
        
    def load_data(self, file_path):
        """
        Loads the dataset from a CSV file.

        Parameters:
        - file_path (str): Path to the CSV file.
        - use_selected_features (bool): If True, use only SELECTED_FEATURES. Otherwise, use all features.
        """
        self.data = pd.read_csv(file_path)  # Load the data
        if self.use_selected_features:
            missing_features = [feature for feature in self.SELECTED_FEATURES if feature not in self.data.columns]
            if missing_features:
                raise ValueError(f"Missing required features in the dataset: {missing_features}")
            self.data = self.data[self.SELECTED_FEATURES]  # Select only the desired features
            print(f"Data loaded with selected features: {self.SELECTED_FEATURES}.")
        else:
            print("Data loaded with all features.")
        print(f"Data shape: {self.data.shape}")
        print(self.data.head())
    

        
    def display_head(self):
        """Displays the first 5 rows of the dataset."""
        print("\nFirst 5 Rows of the Dataset:")
        print(self.data['ACCLASS'].head(20))
        #print(self.data.head())
      
    def describe(self):
        """Describes the statistics of the dataset."""
        print(self.data.describe())
        
        
        
    def handle_outliers(self):
        """
        Identifies and handles outliers in numerical columns (optional).
        Outliers can be removed or capped depending on the strategy chosen.
        """
        initial_acclass_count = self.data['ACCLASS'].value_counts()
        print("Initial 'ACCLASS' counts:")
        print(initial_acclass_count)
        
        for column in self.data.select_dtypes(include=['number']).columns:
            # Calculate the IQR and bounds for the current column
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter the rows for the current column based on the IQR outlier rule
            valid_rows = self.data[column].between(lower_bound, upper_bound)
            
            # Update the data to keep only valid rows for the current column
            self.data = self.data[valid_rows]
        
        print("Outliers handled.")
        
        # Check the counts after handling outliers
        final_acclass_count = self.data['ACCLASS'].value_counts()
        print("Updated 'ACCLASS' counts after handling outliers:")
        print(final_acclass_count)
    

        
    def drop_columns_with_missing_data(self):
        """
        Drops columns exceeding the missing threshold and updates the feature set.
        """
        initial_columns = self.data.shape[1]
        missing_percentage = self.data.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage > self.missing_threshold].index

        print(f"Columns to be dropped: {list(columns_to_drop)}")
        self.data.drop(columns=columns_to_drop, inplace=True)
        
        # Update features after dropping columns
        self.features = [col for col in self.data.columns if col != 'IS_FATAL']
        
        final_columns = self.data.shape[1]
        print(f"Columns remaining after drop: {final_columns}")

        
        
    def drop_columns_manually(self):
        """
        Drops specified columns from the dataset.
        
        Args:
        columns_to_drop (list): List of column names to be dropped.
        
        Returns:
        None: The function modifies the dataframe in-place.
        """
        # Drop the columns if they exist in the dataset
        missing_cols = [col for col in self.columns_to_drop if col not in self.data.columns]
        
        if missing_cols:
            print(f"Warning: The following columns were not found and were not dropped: {missing_cols}")
        
        self.data.drop(columns=[col for col in self.columns_to_drop if col in self.data.columns], inplace=True)
        
        print(f"Columns dropped: {self.columns_to_drop}")

        
    def fill_missing_values(self):
        """
        Fills missing values in the DataFrame.
        - For categorical columns, fills with the mode or 'Unknown'.
        - For numerical columns, fills with the median.
        """
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                mode_value = self.data[column].mode()[0] if not self.data[column].mode().empty else 'Unknown'
                self.data[column].fillna(mode_value, inplace=True)
            else:
                median_value = self.data[column].median()
                self.data[column].fillna(median_value, inplace=True)
        print("Missing values filled.")

        
    def split(self):
        """
        Splits the dataset into training and testing sets (80/20).
        Ensures the resulting splits are DataFrames.
        """
        X = self.data[self.features]
        y = self.data['IS_FATAL']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ensure X_train and X_test are DataFrames
        if not isinstance(self.X_train, pd.DataFrame):
            self.X_train = pd.DataFrame(self.X_train, columns=self.features)
        if not isinstance(self.X_test, pd.DataFrame):
            self.X_test = pd.DataFrame(self.X_test, columns=self.features)

        print(f"Training samples: {self.X_train.shape[0]}, Testing samples: {self.X_test.shape[0]}")
        print(f"Type of X_train: {type(self.X_train)}")  # Should output DataFrame

      
        
    def encode_categorical_features(self):
        categorical_cols = self.X_train.select_dtypes(include=['object']).columns.tolist()
        
        if 'IS_FATAL' in categorical_cols:
            categorical_cols.remove('IS_FATAL')  # Ensure 'IS_FATAL' is not treated as a categorical feature
        
        # Apply pd.get_dummies() to both training and test data
        self.X_train = pd.get_dummies(self.X_train, columns=categorical_cols, drop_first=True)
        self.X_test = pd.get_dummies(self.X_test, columns=categorical_cols, drop_first=True)
        self.features = self.X_train.columns.tolist()
        print(f"Categorical columns encoded: {categorical_cols}")


    def align_dummies(self):
        # Get all columns from both X_train and X_test
        all_columns = list(set(self.X_train.columns).union(set(self.X_test.columns)))
        
        # Reindex both X_train and X_test to include all columns (missing columns will be filled with NaN)
        self.X_train = self.X_train.reindex(columns=all_columns, fill_value=0)
        self.X_test = self.X_test.reindex(columns=all_columns, fill_value=0)
        self.features = self.X_train.columns.tolist()
        return self.X_train, self.X_test
 
    
    def handle_missing_values(self):
        """
        Impute missing values in X_train and X_test after encoding and alignment.
        Ensures the same imputation is applied to both datasets.
        """
        # Impute missing values for both training and test data
        imputer = SimpleImputer(strategy='most_frequent')  # Or 'mean' depending on the feature type
        self.X_train = imputer.fit_transform(self.X_train)  # Fit on X_train
        self.X_test = imputer.transform(self.X_test)  # Apply the same transformation to X_test       
        print("Missing values imputed.")
        
        
    def apply_smote(self):
        """
        Apply SMOTE to balance the class distribution in the training set.
        """
        # Check the class distribution in y_train
        class_counts = self.y_train.value_counts()
        print(f"Class distribution in y_train: {class_counts}")

        if class_counts.shape[0] <= 1:
            print("Error: Only one class present in the target variable. SMOTE cannot be applied.")
            return  # Skip SMOTE if only one class is present

        # Plot class distribution before SMOTE
        print("\n### Class Distribution Before SMOTE ###")
        self.plot_class_distribution(self.y_train, "Class Distribution Before SMOTE")

        # Apply SMOTE to the preprocessed training data
        smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        print("SMOTE applied. Class distribution in training set is now balanced.")

        # Plot class distribution after SMOTE
        print("\n### Class Distribution After SMOTE ###")
        self.plot_class_distribution(self.y_train_smote, "Class Distribution After SMOTE")

        
    def normalize(self):
        """
        Normalizes numeric columns to a 0-1 range for the entire dataset.
        This step is applied before target assignment and splitting.
        """
        # Convert to pandas DataFrame if necessary
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols] = (self.data[numeric_cols] - self.data[numeric_cols].min()) / (self.data[numeric_cols].max() - self.data[numeric_cols].min())

        print("Data normalized to 0-1 range for numeric columns.")
   
        
    def assign_target(self):
        """
        Creates the target column 'IS_FATAL' from the 'ACCLASS' column and drops 'ACCLASS'.
        This happens after normalization and before the data split.
        """
        if 'ACCLASS' in self.data.columns:
            # Create target variable 'IS_FATAL' and remap to binary (1 for FATAL, 0 for NON-FATAL)
            self.data['IS_FATAL'] = self.data['ACCLASS'].apply(lambda x: 1 if x == 'Fatal' else 0)
            print("Target column 'IS_FATAL' created and mapped to binary (1 for FATAL, 0 for NON-FATAL).")
            
            # Drop 'ACCLASS' column
            self.data.drop(columns=['ACCLASS'], inplace=True)
            print("No of classes in IS_FATAL: ", self.data['IS_FATAL'].value_counts())
            print("'ACCLASS' column dropped.")
            
            # Print the count of FATAL (1) and NON-FATAL (0)
            fatal_count = self.data['IS_FATAL'].value_counts().get(1, 0)
            non_fatal_count = self.data['IS_FATAL'].value_counts().get(0, 0)
            print(f"FATAL count: {fatal_count}")
            print(f"NON-FATAL count: {non_fatal_count}")
            
            # Update features to exclude 'IS_FATAL'
            self.features = [col for col in self.data.columns if col != 'IS_FATAL']
            print(f"Features updated: {self.features}")
        else:
            print("Error: 'ACCLASS' column not found.")

            
    def transform_columns(self):
        """
        Creates a column transformer for scaling numerical and encoding categorical columns.
        Applies the transformations to both the training and test data.
        Assumes no missing values in the dataset.
        """
        # Ensure X_train and X_test are DataFrames
        if not isinstance(self.X_train, pd.DataFrame):
            print("Converting X_train to DataFrame...")
            self.X_train = pd.DataFrame(self.X_train, columns=self.features)
        
        if not isinstance(self.X_test, pd.DataFrame):
            print("Converting X_test to DataFrame...")
            self.X_test = pd.DataFrame(self.X_test, columns=self.features)

        # Separate numerical and categorical columns
        numerical_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.X_train.select_dtypes(include=['object']).columns.tolist()

        # Remove the target column ('IS_FATAL') from the lists of numerical and categorical columns
        numerical_cols = [col for col in numerical_cols if col != 'IS_FATAL']
        categorical_cols = [col for col in categorical_cols if col != 'IS_FATAL']

        print(f"Numerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")

        # Create the ColumnTransformer with transformations for numerical and categorical columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ],
            remainder='passthrough'
        )

        # Fit and transform the training data
        X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        # Transform the test data using the already fitted preprocessor
        X_test_transformed = self.preprocessor.transform(self.X_test)

        # Generate column names after transformation using self.features
        num_col_names = numerical_cols  # Numerical columns remain the same
        cat_col_names = [f"{col}_{cat}" for col in categorical_cols for cat in self.X_train[col].unique()]
        transformed_col_names = list(num_col_names) + list(cat_col_names)

        # Convert the transformed arrays back to DataFrame with correct column names
        self.X_train = pd.DataFrame(X_train_transformed, columns=transformed_col_names, index=self.X_train.index)
        self.X_test = pd.DataFrame(X_test_transformed, columns=transformed_col_names, index=self.X_test.index)

        print("Column transformation applied to both training and test sets.")
        return self.preprocessor


    def train_model(self):
        """
        Train the Logistic Regression model using the processed and balanced data.
        """

        # Fit the model to the resampled training data
        self.model = self.models[self.model_name]
        self.model.fit(self.X_train_smote, self.y_train_smote)
        print(f"Number of features: {len(self.features)}")
        print(f"Number of coefficients: {len(self.model.coef_[0])}")
        print("Model training complete.")
        
           
    def cross_validate(self):
        """
        Perform 10-fold cross-validation and print the results.
        """
        cv_scores = cross_val_score(self.model, self.X_train_smote, self.y_train_smote, cv=10)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average cross-validation score: {cv_scores.mean():.4f}")
        
        
    def evaluate_model(self):
        """
        Evaluates the model on the test set and prints the results.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy and other metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        

    def roc_auc(self):
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]  
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
        print(f"Logistic Regression AUC: {auc_score:.4f}")

        
    def display_confusion_matrix(self):
        self.y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive']) 
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

         
    def feature_importance(self):
        """
        Visualizes the top 10 most important features by plotting the absolute coefficients of the logistic regression model.
        Ensures that the feature list is updated after preprocessing steps and orders features by importance in descending order.
        """
        if self.model.coef_ is None:
            print("Error: Model must be fitted before calculating feature importance.")
            return
        if len(self.features) != len(self.model.coef_[0]):
            print(f"Error: Number of features does not match the model's coefficients. Expected {len(self.model.coef_[0])}, but got {len(self.features)}.")
            return
        fi = pd.DataFrame({
            'feature': self.features,  
            'importance': abs(self.model.coef_[0]) 
        })
        fi = fi.sort_values('importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        plt.barh(fi['feature'], fi['importance'])
        plt.title('Top 10 Feature Importance in Hit Prediction')
        plt.xlabel('Absolute Coefficient Value')
        plt.ylabel('Feature')
        plt.show()
        
        
    def save_model(self, model, filename_prefix='logistic_regression_model'):
        """
        Saves the trained model to a file using joblib, with a dynamic filename 
        based on the scope of features used for training.
        
        Args:
        model: The trained model to be saved.
        feature_scope: A string indicating the scope of features used ('all', 'many', 'few').
        filename_prefix: The prefix of the filename to customize naming further.
        
        Raises:
        ValueError: If the feature_scope is not one of the accepted values.
        """
        valid_scopes = ['all', 'many', 'few']
        if self.feature_scope not in valid_scopes:
            raise ValueError(f"Invalid feature_scope '{self.feature_scope}'. Choose from {valid_scopes}.")
        
        filename = f"{filename_prefix}_smote_{self.feature_scope}.joblib"
        joblib.dump(model, filename)
        print(f"\nModel saved as '{filename}'")


       
    def plot_class_distribution(self, y, title):
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y, palette='Set2')
        plt.title(title)
        plt.xlabel('Accident Class (1 = Fatal, 0 = Non-Fatal)')
        plt.ylabel('Count')
        plt.xticks(ticks=[0, 1], labels=['Non-Fatal', 'Fatal'])
        plt.grid(True)
        plt.show()
        
        
    def feature_correlation(self):
        """
        Plots a heatmap of the feature correlations for the columns defined in 'self.features'
        plus 'ACCLASS'.
        """
        if not isinstance(self.features, list):
            raise ValueError("'self.features' should be a list of column names.")
        
        if not hasattr(self, 'data') or not isinstance(self.data, pd.DataFrame):
            raise ValueError("'self.data' should be a pandas DataFrame.")
        columns_to_check = self.features + ['IS_FATAL']
        missing_cols = [col for col in columns_to_check if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"The following columns are missing in the data: {', '.join(missing_cols)}")
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data[columns_to_check].corr()  # Get correlation matrix
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return self

    def plot_outliers(self):
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        plt.figure(figsize=(15, 6))
        self.data[numeric_cols].boxplot()
        plt.ylim(self.data[numeric_cols].min().min() - 1, self.data[numeric_cols].max().max() + 1)
        plt.xticks(rotation=45)
        plt.title("Box Plot to Identify Outliers")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
          
    def run(self):
        self.display_head()
        self.describe()
        self.drop_columns_manually()
        #self.handle_outliers()
        self.plot_outliers()
        self.drop_columns_with_missing_data()
        self.fill_missing_values()  
        self.assign_target()
        self.feature_correlation()
        self.normalize()   
        self.split()
        self.encode_categorical_features()  
        self.align_dummies()
        self.handle_missing_values()
        self.transform_columns()
        self.apply_smote()
        self.train_model()
        self.cross_validate()
        self.evaluate_model()
        self.roc_auc()
        self.display_confusion_matrix()
        self.feature_importance()
        self.save_model(self.model)
      

MANY_FEATURES = [
    'ACCNUM', 'TIME', 'ROAD_CLASS', 'DISTRICT', 'LONGITUDE', 'TRAFFCTL', 'VISIBILITY',
    'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY', 'LATITUDE',
    'VEHTYPE', 'AUTOMOBILE', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'DIVISION'
]


FEW_FEATURES = ["INJURY", "LATITUDE", "LIGHT", "ACCLASS"]


# Training Phase
columns_to_drop = ["OBJECTID"]  # Define columns to drop
pipeline = KSIFatalityPipeline(missing_threshold=0.20, columns_to_drop=columns_to_drop, model_name="Logistic Regression", selected_features=FEW_FEATURES, use_selected_features=False, feature_scope="few")

# Train and save the model
pipeline.load_data('ksi.csv')
pipeline.run()


import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin


class KSIFatalityPredictor:
    def __init__(self, model_path, csv_file_path, selected_features):
        """
        Initialize the KSIFatalityPredictor with configuration parameters.
        """
        self.model_path = model_path
        self.csv_file_path = csv_file_path
        self.selected_features = selected_features
        self.model = None
        self.df = None
        self.sample_row = None
        self.sample_row_encoded_df = None
        self.aligned_sample_row = None
        self.preprocessor = None
        self.prediction = None

    def load_model(self):
        """
        Load the trained model from a file.
        """
        try:
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully.")
            self.verify_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

    def verify_model(self):
        """
        Verify the type of model loaded.
        """
        if isinstance(self.model, LogisticRegression):
            print("The model is a Logistic Regression model.")
        elif isinstance(self.model, ClassifierMixin):
            print(f"The model is a valid classifier but not Logistic Regression. It is {self.model.__class__.__name__}.")
        else:
            print(f"The model is not a classifier. It is a {self.model.__class__.__name__}.")

    def load_data(self):
        """
        Load data from the CSV file.
        """
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print("CSV file loaded successfully.")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            exit()

    def select_columns(self):
        """
        Select relevant columns from the DataFrame.
        """
        if self.df is not None:
            self.df = self.df[self.selected_features]
        else:
            print("Data not loaded. Call load_data() first.")

    def select_sample_row(self):
        """
        Select a random sample row from the DataFrame.
        """
        if self.df is not None:
            self.sample_row = self.df.sample(n=1).copy()
            print("\n### Sample Selected (Before Processing) ###")
            print(self.sample_row)
        else:
            print("DataFrame is empty or not loaded. Please load data first.")

    def handle_missing_values(self):
        """
        Handle missing values in the sample row.
        """
        if self.sample_row is not None:
            self.sample_row.fillna(self.sample_row.median(numeric_only=True), inplace=True)
        else:
            print("Sample row not available. Call select_sample_row() first.")

    def create_column_transformer(self):
        """
        Define the column transformer for preprocessing.
        """
        if self.sample_row is not None:
            numerical_features = self.sample_row.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = self.sample_row.select_dtypes(include=['object']).columns.tolist()

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
        else:
            print("Sample row not available. Call select_sample_row() first.")

    def preprocess_sample_row(self):
        """
        Preprocess the sample row using the column transformer.
        """
        if self.preprocessor is not None:
            sample_row_encoded = self.preprocessor.fit_transform(self.sample_row)
            self.sample_row_encoded_df = pd.DataFrame(
                sample_row_encoded,
                columns=self.preprocessor.get_feature_names_out()
            )
        else:
            print("Preprocessor not initialized. Call create_column_transformer() first.")

    def align_with_model_columns(self):
        """
        Align the sample row with the model's expected input.
        """
        try:
            if hasattr(self.model, 'feature_names_in_'):
                original_columns = self.model.feature_names_in_
                self.aligned_sample_row = self.sample_row_encoded_df.reindex(columns=original_columns, fill_value=0)
                return self.aligned_sample_row.values.reshape(1, -1)
            else:
                print("Model doesn't have feature_names_in_ attribute.")
                return None
        except AttributeError:
            print("Error aligning sample row with model columns.")
            exit()

    def make_prediction(self):
        """
        Make a prediction using the model and interpret the result.
        """
        try:
            self.prediction = self.model.predict(self.aligned_sample_row)
            interpreted_prediction = self._interpret_prediction(self.prediction)
            print(f"\n### Prediction: {interpreted_prediction} ###")
            return interpreted_prediction
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return None

    def _interpret_prediction(self, prediction):
        """
        Interpret the prediction value into a human-readable string.

        Args:
            prediction (array-like): The model's prediction output.

        Returns:
            str: "Fatal" if prediction is 1, otherwise "Non-Fatal".
        """
        return "Fatal" if prediction[0] == 1 else "Non-Fatal"

    def run(self):
        """
        Execute the prediction process.
        """
        self.load_data()
        self.load_model()
        self.verify_model()
        self.select_columns()
        self.select_sample_row()
        self.handle_missing_values()
        self.create_column_transformer()
        self.preprocess_sample_row()
        self.align_with_model_columns()
        self.make_prediction()


# Example usage
model_path = 'logistic_regression_model_smote_few.joblib'
csv_file_path = 'ksi.csv'
selected_features = ['LIGHT', 'LATITUDE', 'LONGITUDE', 'ACCLASS']  # Replace with actual feature names

predictor = KSIFatalityPredictor(model_path, csv_file_path, selected_features)
predictor.run()
