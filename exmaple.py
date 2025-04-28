import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

#Example dataset
data = {
"num1": [1.0, 2.5, np.nan, 4.0, 5.0, np.nan],
"num2": [3.0, np.nan, 3.5, 4.5, np.nan, 6.0],
"cat1": ["A", "B", np.nan, "A", "C", "B"],
"cat2": [np.nan, "X", "Y", "X", "Z", np.nan],
"target": [0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Define column names by type
numeric_features = ["num1", "num2"]
categorical_features = ["cat1", "cat2"]

# Create pipeline for numerical features.
# The SimpleImputer with add_indicator=True will append extra indicator columns:
# one indicator column per numerical feature that had missing entries.
numeric_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
("scaler", StandardScaler())
])

# Create pipeline for categorical features.
# The SimpleImputer (with strategy="constant") replaces missing values with the string "missing"
# and add_indicator=True appends indicator columns for each categorical feature.
# Then OneHotEncoder converts the (imputed) categorical features into dummy variables.
categorical_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="constant", fill_value="missing", add_indicator=True)),
("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine the numeric and categorical pipelines into a single ColumnTransformer.
preprocessor = ColumnTransformer(transformers=[
("num", numeric_pipeline, numeric_features),
("cat", categorical_pipeline, categorical_features)
])

# Create the final pipeline with preprocessing and the classifier model.
model_pipeline = Pipeline(steps=[
("preprocessor", preprocessor),
("classifier", LogisticRegression(solver="liblinear"))
])

# Optional: split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit the pipeline
model_pipeline.fit(X_train, y_train)

# Evaluate the pipeline using cross-validation (for instance, 5-fold CV)
scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV score:", np.mean(scores))

# Predict on the test set and print predictions
preds = model_pipeline.predict(X_test)
print("Test set predictions:", preds)

Inspect the transformed features (optional)
Transform the training data and convert to an array.
X_train_transformed = model_pipeline.named_steps["preprocessor"].transform(X_train)
print("Shape of transformed training data:", X_train_transformed.shape)