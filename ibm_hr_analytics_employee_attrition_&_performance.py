"""IBM HR Analytics Employee Attrition & Performance.ipynb

#IBM HR Analytics Employee Attrition & Performance
##1. Dataset Analysis and Preprocessing
"""

import pandas as pd
df = pd.read_csv('/content/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df.head()

df.describe()

df.info()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

!pip install tabulate

from tabulate import tabulate
table_data = []
for column in df.columns:
    unique_values_count = df[column].nunique()
    unique_values = df[column].unique()
    table_data.append([column, unique_values_count, unique_values])
print(tabulate(table_data, headers=["Column", "Number of Unique Values", "Unique Values"]))

df = df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)

import seaborn as sns
import matplotlib.pyplot as plt
attrition_counts = df['Attrition'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Attrition Status Distribution")
plt.show()

numeric_subset = df.select_dtypes(include=[np.number])
plt.figure(figsize=(15, 8))
sns.boxplot(data=numeric_subset)
plt.xticks(rotation=45)
plt.xlabel("Numerical Features")
plt.title("Boxplot of Numerical Features")
plt.show()

"""##Dealing with outliers for 'MonthlyIncome' using log transformation"""

df['MonthlyIncome'] = np.log(df['MonthlyIncome'])
plt.figure(figsize=(10, 6))
sns.histplot(df['MonthlyIncome'], bins=30, kde=True)
plt.title('Histogram of Log-Transformed Monthly Income')
plt.xlabel('Log(Monthly Income)')
plt.ylabel('Frequency')
plt.show()

"""##Relationship between 'Attrition' and categorical features using bar plots"""

categorical_features = df.select_dtypes(include=['object']).columns
plt.figure(figsize=(18, 10))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=df, x=feature, hue='Attrition')
    plt.title(f'Attrition by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""##Correlation check with numerical features"""

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
numerical_features

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Compute correlation between numerical features and the target variable
correlation_with_attrition = df[numerical_features].corrwith(df['Attrition']).abs().sort_values(ascending=False)

# Print correlation values
print("Correlation with 'Attrition':")
print(correlation_with_attrition)

top_numerical_features = correlation_with_attrition.head(5).index  # Select top 5 features
print("Top Numerical Features:")
print(top_numerical_features)

# Concatenate numerical features with the 'Attrition' column
data = df[top_numerical_features].join(df['Attrition'])

# Draw pair plot
sns.pairplot(data, hue='Attrition', diag_kind='kde')
plt.show()

"""#transformation"""

df1=df.copy()

df1.head()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

df1.info()

categorical_features = df1.select_dtypes(include=['object']).columns
numerical_features = df1.select_dtypes(include=['int64', 'float64']).drop(columns=['Attrition']).columns
numerical_transformer = StandardScaler()
label_encoders = [LabelEncoder() for _ in categorical_features]

#LabelEncoder to each categorical feature separately
for i, feature in enumerate(categorical_features):
    df1[feature] = label_encoders[i].fit_transform(df1[feature])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Scale numerical features
        ('cat', 'passthrough', categorical_features)  # Pass through categorical features (already encoded)
    ]
)
preprocessed_data = preprocessor.fit_transform(df1)
df1 = pd.DataFrame(preprocessed_data, columns=list(numerical_features) + list(categorical_features))
print(df1.head())

df1.columns

df1.info()

attrition_column = df['Attrition']
df1 = pd.concat([df1, attrition_column], axis=1)

"""##Correlation after normalization"""

correlation_matrix = df1.corr()

#threshold for correlation
threshold = 0.6
high_correlation = (correlation_matrix.abs() > threshold)
#mask to hide the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix[high_correlation], annot=True, mask=mask,square=True, linewidths=0.5, linecolor='black',vmin = -1, vmax = 1)
plt.title("Correlation Matrix (Features with correlation > 0.6)")
plt.show()

"""##Checking encoded feature names"""

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder for each categorical feature
label_encoders = {feature: LabelEncoder() for feature in categorical_features}

# Fit LabelEncoder to each categorical feature and display mapping
for feature, encoder in label_encoders.items():
    encoder.fit(df[feature])
    print(f"Mapping for feature '{feature}':")
    for category, label in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        print(f"   {category}: {label}")

"""#Handling Imbalanced Data (of target variable) using SMOTE"""

attrition_counts = df1['Attrition'].value_counts()

plt.figure(figsize=(6, 6))
sns.barplot(x=attrition_counts.index, y=attrition_counts.values,color='skyblue')
plt.title("Class Distribution of Attrition")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()
print(attrition_counts)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

X = df1.drop(columns=['Attrition'])
y = df1['Attrition']

# Apply SMOTE to balance the class distribution
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
class_proportions = y_resampled.value_counts(normalize=True)
print("Class Proportions after SMOTE:")
print(class_proportions)

"""#Splitting"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Shape of x_train:", X_train.shape)
print("Shape of x_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

"""##Finding Feature Scores using Mutual Information"""

from sklearn.feature_selection import mutual_info_classif
mutual_info_scores = mutual_info_classif(X_train, y_train)
#DataFrame to store the scores
feature_scores = pd.DataFrame({'Feature': X_train.columns, 'Mutual Information Score': mutual_info_scores})
# Sorting in descending order
feature_scores = feature_scores.sort_values(by='Mutual Information Score', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Mutual Information Score', y='Feature', data=feature_scores)
plt.title('Feature Scores using Mutual Information')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.show()

"""##Anova test for numerical variables"""

from scipy.stats import f_oneway
feature_names = []
f_values = []
for feature in numerical_features:
    # Computation of ANOVA F-value
    f_value, _ = f_oneway(X_train[feature][y_train == 0], X_train[feature][y_train == 1])
    feature_names.append(feature)
    f_values.append(f_value)

plt.figure(figsize=(10, 5))
plt.plot(feature_names, f_values, marker='o', color='blue', linestyle='-')
plt.title('ANOVA F-values for Numerical Features')
plt.xlabel('Feature')
plt.ylabel('ANOVA F-value')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()

#dropping unnecessary features
x_train = X_train.drop(columns = ['Gender', 'BusinessTravel', 'EducationField',  'Department','JobInvolvement','WorkLifeBalance', 'NumCompaniesWorked','MaritalStatus', 'OverTime', 'JobRole', 'DailyRate', 'HourlyRate', 'PerformanceRating', 'MonthlyRate'])
x_test = X_test.drop(columns = ['Gender', 'BusinessTravel', 'EducationField',  'Department','JobInvolvement','WorkLifeBalance', 'NumCompaniesWorked','MaritalStatus', 'OverTime', 'JobRole', 'DailyRate', 'HourlyRate', 'PerformanceRating', 'MonthlyRate'])

x_train.head()

"""#Model Evaluation and Optimization"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

logistic_reg = LogisticRegression()
random_forest = RandomForestClassifier()
svm_classifier = SVC()
decision_tree = DecisionTreeClassifier()
xgb_classifier = XGBClassifier()
knn_classifier = KNeighborsClassifier()

logistic_reg.fit(x_train, y_train)
random_forest.fit(x_train, y_train)
svm_classifier.fit(x_train, y_train)
decision_tree.fit(x_train, y_train)
xgb_classifier.fit(x_train, y_train)
knn_classifier.fit(x_train, y_train)

models = [("Logistic Regression", logistic_reg), ("Random Forest", random_forest), ("Support Vector Machine", svm_classifier), ("Decision Tree", decision_tree), ("XGBoost", xgb_classifier), ("K-Nearest Neighbors", knn_classifier)]

for model_name, model in models:
    print(f"Model: {model_name}")

    # Cross-validation score
    cv_score = cross_val_score(model, x_train, y_train, cv=5)
    print("Cross-validation Score:", cv_score.mean())

    # Predictions
    y_pred = model.predict(x_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

"""#Model: Random Forest


*   Cross-validation Score: 0.9097346269999358
*   Accuracy: 0.8927125506072875
*   Precision: 0.9063829787234042
*   Recall: 0.8729508196721312
*   F1 Score: 0.8893528183716075

#Hyperparameter tuning & Bagging & AdaBoost Classifier ensemble methods
"""

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# Random Forest parameters for hyperparameter tuning
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_classifier = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(x_train, y_train)

#best parameters for Random Forest
best_rf_params = rf_grid_search.best_params_
best_rf_classifier = RandomForestClassifier(random_state=42, **best_rf_params)

# Bagging ensemble method with Random Forest base estimator
bagging_classifier = BaggingClassifier(base_estimator=best_rf_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(x_train, y_train)

# AdaBoost ensemble method with Random Forest base estimator
adaboost_classifier = AdaBoostClassifier(base_estimator=best_rf_classifier, n_estimators=50, random_state=42, learning_rate=1.0)
adaboost_classifier.fit(x_train, y_train)

bagging_accuracy = bagging_classifier.score(x_test, y_test)
adaboost_accuracy = adaboost_classifier.score(x_test, y_test)
print("Bagging Classifier Accuracy:", bagging_accuracy)
print("AdaBoost Classifier Accuracy:", adaboost_accuracy)

"""##AdaBoost Classifier gives a better accuracy: 90.8 %

#FNN Implementation
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import make_scorer, accuracy_score

pip install tensorflow scikeras scikit-learn

from scikeras.wrappers import KerasClassifier

#Using the Functional API
def build_model(input_shape=(x_train.shape[1],)):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

#function to wrap keras model to return a KerasClassifier
def create_keras_model():
    model = build_model()
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return KerasClassifier(build_fn=lambda: model, epochs=10, batch_size=32, verbose=0)

#hyperparameter tuning
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}

grid = GridSearchCV(estimator=create_keras_model(), param_grid=param_grid, cv=3, scoring='accuracy')
grid_result = grid.fit(x_train, y_train)
#print best score & best params
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""#Summary: 100% score with FNN (batch_size': 32, 'epochs': 10)"""
