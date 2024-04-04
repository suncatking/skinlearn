# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:07:44 2024

@author: Administrator
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load historical data
data = pd.read_csv('double_color_ball_data.csv')

# Preprocess the data
data['Combined'] = data['Red'] * 10 + data['Blue']
target = data['Combined']
features = data.drop(['Combined', 'Red', 'Blue'], axis=1)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(features, target)

# Predict the next 10 sets of numbers
predictions = []
for i in range(10):
    # Get the most likely combination
    predicted_probabilities = model.predict_proba(features)[:, 1]
    next_number = features.columns.get_loc('Combined') + 1
    next_prediction = features.columns[predicted_probabilities.argsort()[-next_number:][::-1]][0]
    predictions.append(next_prediction)