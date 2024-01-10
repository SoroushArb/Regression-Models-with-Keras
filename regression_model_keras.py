# Import necessary libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Download and Clean dataset
# (Assuming that the dataset is already available or will be provided separately)

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
concrete_data = pd.read_csv('your_dataset.csv')

# Explore the dataset
concrete_data_shape = concrete_data.shape
concrete_data_description = concrete_data.describe()
concrete_data_null_sum = concrete_data.isnull().sum()

# Extract predictors and target variable
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

# Normalize predictors
predictors_norm = (predictors - predictors.mean()) / predictors.std()
n_cols = predictors_norm.shape[1]  # number of predictors

# Define regression model
def regression_model():
    # Create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and train the model
model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
