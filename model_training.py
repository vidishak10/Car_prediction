import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the datasets
data1 = pd.read_csv('data/car_data1.csv')
data2 = pd.read_csv('data/car_data2.csv')
data3 = pd.read_csv('data/car_data3.csv')

# Rename and select relevant features from dataset 1
data1 = data1.rename(columns={
    'Manufacturer': 'company',
    'Model': 'model',
    'Prod. year': 'year',
    'Category': 'category',
    'Leather interior': 'leather_interior',
    'Fuel type': 'fuel_type',
    'Engine volume': 'engine_volume',
    'Mileage': 'mileage',
    'Cylinders': 'cylinders',
    'Gear box type': 'transmission',
    'Drive wheels': 'drive_wheels',
    'Doors': 'doors',
    'Wheel': 'wheel',
    'Color': 'color',
    'Airbags': 'airbags',
    'Price': 'price'
})
data1 = data1[['company', 'model', 'year', 'fuel_type', 'engine_volume', 'mileage', 'transmission', 'price']]

# Rename and select relevant features from dataset 2
data2 = data2.rename(columns={
    'Company Name': 'company',
    'Model Name': 'model',
    'Model Year': 'year',
    'Engine Type': 'fuel_type',
    'Engine Capacity': 'engine_volume',
    'Mileage': 'mileage',
    'Transmission Type': 'transmission',
    'Price': 'price'
})
data2 = data2[['company', 'model', 'year', 'fuel_type', 'engine_volume', 'mileage', 'transmission', 'price']]

# Rename and select relevant features from dataset 3
data3 = data3.rename(columns={
    'model': 'model',
    'year': 'year',
    'fuelType': 'fuel_type',
    'engineSize': 'engine_volume',
    'mileage': 'mileage',
    'transmission': 'transmission',
    'price': 'price'
})
data3 = data3[['model', 'year', 'fuel_type', 'engine_volume', 'mileage', 'transmission', 'price']]

# Add a 'company' column with missing values to dataset 3
data3['company'] = None

# Concatenate the datasets
combined_data = pd.concat([data1, data2, data3], ignore_index=True)

# Clean 'mileage' column to remove non-numeric characters and convert to numeric
combined_data['mileage'] = combined_data['mileage'].str.replace(' km', '').str.replace(',', '').astype(float)

# Define features and target variable
X = combined_data[['company', 'model', 'year', 'fuel_type', 'engine_volume', 'mileage', 'transmission']]
y = combined_data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numerical_features = ['year', 'engine_volume', 'mileage']
categorical_features = ['company', 'model', 'fuel_type', 'transmission']

# Separate transformers for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Convert 'engine_volume' to numeric (assuming you extract numeric part or handle non-numeric values)
X_train['engine_volume'] = pd.to_numeric(X_train['engine_volume'], errors='coerce')
X_test['engine_volume'] = pd.to_numeric(X_test['engine_volume'], errors='coerce')

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'car_price_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R²: {r2}')
