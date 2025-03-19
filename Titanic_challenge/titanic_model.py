import numpy as np
import pandas as pd
import re
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

# Suppress the specific pandas warning - updated for newer pandas versions
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Load the official Kaggle training dataset
train_data = pd.read_csv(r'C:\Users\ALI\Downloads\titanic\train.csv') #replace it with your file path 
print(f"Training data shape: {train_data.shape}")
print("\nFirst few rows of training data:")
print(train_data.head())

# Load the test data
test_data = pd.read_csv(r'C:\Users\ALI\Downloads\titanic\test.csv') #replace it with your file path
print(f"\nTest data shape: {test_data.shape}")

# Feature engineering for both datasets
print("\nPerforming feature engineering...")

# Extract titles from names
def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

train_data['Title'] = train_data['Name'].apply(extract_title)
test_data['Title'] = test_data['Name'].apply(extract_title)

# Display title distribution
print("\nDistribution of titles in training data:")
print(train_data['Title'].value_counts())

# Group rare titles
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Don': 'Royalty',
    'Sir': 'Royalty',
    'Lady': 'Royalty',
    'Countess': 'Royalty',
    'Jonkheer': 'Royalty',
    'Capt': 'Officer',
}

train_data['Title'] = train_data['Title'].map(lambda t: title_mapping.get(t, 'Other'))
test_data['Title'] = test_data['Title'].map(lambda t: title_mapping.get(t, 'Other'))

# Create family size feature
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Create is_alone feature
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# Create family group categories
def categorize_family(size):
    if size == 1:
        return 'Alone'
    elif 2 <= size <= 4:
        return 'Small'
    else:
        return 'Large'

train_data['FamilyGroup'] = train_data['FamilySize'].apply(categorize_family)
test_data['FamilyGroup'] = test_data['FamilySize'].apply(categorize_family)

# Extract deck from cabin (if available)
def get_deck(cabin):
    if pd.isna(cabin):
        return 'Unknown'
    return cabin[0]

train_data['Deck'] = train_data['Cabin'].apply(get_deck)
test_data['Deck'] = test_data['Cabin'].apply(get_deck)

# Prepare the data for the model
print("\nPreparing data for modeling...")

# Split into features and target
X_train = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train_data['Survived']

# Preprocessing for numerical features
numerical_features = ['Age', 'Fare', 'FamilySize']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'FamilyGroup', 'Deck']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42))
])

# Perform cross-validation to evaluate the model
print("\nEvaluating model with cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train the final model on all training data
print("\nTraining the final model...")
model.fit(X_train, y_train)

# Prepare test data
print("\nPreparing test data...")
X_test = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions.astype(int)
})

submission.to_csv(r'C:\Users\ALI\Downloads\titanic\submission.csv', index=False)
print("\nSubmission file created successfully!")
print(f"Path: C:\\Users\\ALI\\Downloads\\titanic\\submission.csv")

# Feature importance
if hasattr(model[-1], 'feature_importances_'):
    feature_importances = pd.DataFrame(model[-1].feature_importances_,
                                      index=model[:-1].get_feature_names_out(),
                                      columns=['importance'])
    print("\nTop 10 most important features:")
    print(feature_importances.sort_values('importance', ascending=False).head(10))