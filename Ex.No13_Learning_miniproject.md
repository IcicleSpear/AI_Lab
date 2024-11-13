# Ex.No: 13 Machine Learning – MiniProject 
### DATE: 23/10/24                                                                           
### REGISTER NUMBER : 212221040004
### AIM: 
To write a program to develop a predictive model for classifying dementia types based on patient data, using a TabNet
###  Algorithm:
1.Import Libraries

2.Load Data

3.Handle Missing Values

4.Encode Data

5.Set Target

6.Select Features

7.Normalize Data and Split Data

8.Define Model

9.Train Model

10.Evaluate Model

11.Display Result
### Program:
# Import necessary libraries
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import requests
from bs4 import BeautifulSoup
```
# Load the dataset
```
df = pd.read_csv('dementia_dataset.csv')
```
# Handle missing values for numeric columns
```
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
```
# Encode non-numeric columns
```
non_numeric_cols = df.select_dtypes(include=['object']).columns
for col in non_numeric_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
```
# Set 'Group' as the target column (1 for Demented, 0 for Nondemented)
```
target_column = 'Group'
df[target_column] = df[target_column].apply(lambda x: 1 if x == 'Demented' else 0)
```
# Select relevant features and target
```
features = df.drop(columns=[target_column, 'Subject ID', 'MRI ID'])  # Drop ID columns
target = df[target_column]
```
# Normalize the features
```
scaler = StandardScaler()
features = scaler.fit_transform(features)
```
# Split the data into training and testing sets (80/20 split)
```
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
```
# Define the TabNet model
```
tabnet_model = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax'  # "entmax" can also be used for sparse gradients
)
```
# Train the model
```
tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['accuracy'],
    max_epochs=50,
    patience=10,
    batch_size=256,
    virtual_batch_size=64,
)
```
# Predict on the test set
```
y_pred = tabnet_model.predict(X_test)
```
# Calculate accuracy
```
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```
# Predict for an individual instance (e.g., the first instance in X_test)
```
individual_instance = X_test[0].reshape(1, -1)
prediction_prob = tabnet_model.predict_proba(individual_instance)
```
# Get the predicted class (0 for Alzheimer's, 1 for Vascular Dementia, etc.)
```
predicted_class_idx = prediction_prob.argmax(axis=1)[0]
```
# Map the predicted class index to the actual dementia type
```
dementia_types = ['Alzheimer', 'Vascular Dementia', 'Frontotemporal Dementia', 'Lewy Body Dementia']
predicted_dementia = dementia_types[predicted_class_idx]
```
# Print the predicted dementia type and probability
```
probability = prediction_prob[0][predicted_class_idx]
print(f"The model predicts the patient has {predicted_dementia} with a probability of {probability*100:.2f}%.")
```
### Output:
![WhatsApp Image 2024-11-13 at 16 26 07_7e0af928](https://github.com/user-attachments/assets/f5f6e6a0-1729-4268-80ff-d46de3315439)


### Result:
Thus the system was trained successfully and the prediction was carried out.
