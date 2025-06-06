# -----(1) IMPORTS------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
import re


# -----(2) LOAD DATA SET------
data = pd.read_csv('laptop_price.csv', encoding='latin1')


# -----(3) DISPLAY DATA IN DATASET------
data.head()


# -----(4) Data Processing------
# -----(4.1) CLEAN Ram COLUMN------
data['Ram'] = data['Ram'].str.replace('GB', '').astype('int32')

data.head()


# -----(4.2) CLEAN Weight COLUMN------
data['Weight'] = data['Weight'].str.replace('kg', '').astype('float32')

data.head()


# -----(4.3) CREATE NEW COLUMN Gpu_name & INSERT GPU NAMES FROM Gpu COLUMN------
data['Gpu_name'] = data['Gpu'].apply(lambda x: x.split()[0])
data = data[data['Gpu_name'] !='ARM']

data.head()


# -----(4.4) CREATE NEW COLUMN Cpu_name & INSERT CPU NAMES FROM Cpu COLUMN------
data['cpu_name'] = data['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

def set_processor(name):
    if name in['Intel Core i5', 'Intel Core i7', 'Intel Core i3']:
        return name
    elif name.split()[0] == 'AMD':
        return 'AMD'
    else:
        return 'Other'

data['cpu_name'] = data['cpu_name'].apply(set_processor)

data[19:25]


# -----(4.5) CLEAN Memory COLUMN & Create NEW Memory COLUMNS------
data['Memory'] = data['Memory'].str.replace('GB', '').str.replace('TB', '000')

data['HDD'] = data['Memory'].apply(lambda x: int(re.search(r'(\d+)', x).group()) if 'HDD' in x else 0)
data['SSD'] = data['Memory'].apply(lambda x: int(re.search(r'(\d+)', x).group()) if 'SSD' in x else 0)
data['Flash_Storage'] = data['Memory'].apply(lambda x: int(re.search(r'(\d+)', x).group()) if 'Flash Storage' in x else 0)
data['Hybrid'] = data['Memory'].apply(lambda x: int(re.search(r'(\d+)', x).group()) if 'Hybrid' in x else 0)

def extract_storage(val, stype):
    parts = val.split('+')
    for part in parts:
        if stype in part:
            return int(re.search(r'(\d+)', part).group())
    return 0

data['HDD'] = data['Memory'].apply(lambda x: extract_storage(x, 'HDD'))
data['SSD'] = data['Memory'].apply(lambda x: extract_storage(x, 'SSD'))
data['Flash_Storage'] = data['Memory'].apply(lambda x: extract_storage(x, 'Flash Storage'))
data['Hybrid'] = data['Memory'].apply(lambda x: extract_storage(x, 'Hybrid'))

data[19:25]


# -----(4.6) CLEAN OpSys COLUMN------
def set_os(inpt):
    if inpt in ['Windows 10', 'Windows 7', 'Windows 10 S']:
        return 'Windows'
    elif inpt in ['macOS', 'Mac OS X']: 
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'

data['OpSys'] = data['OpSys'].apply(set_os)

data[1:6]


# -----(4.7) CLEAN Company COLUMN------
def add_company(inpt):
    if inpt in ['Samsung', 'Mediacom', 'Razer', 'Microsoft', 'Vero', 'Xiaomi', 'Chuwi', 'Fujitsu', 'Google', 'LG', 'Huawei']:
        return 'Other'
    else:
        return inpt

data['Company'] = data['Company'].apply(add_company)

data[30:35]


# -----(4.8) CREATE NEW COLUMNS Touchscreen & IPS------
data['Touchscreen'] = data['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
data['IPS'] = data['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

data[70:75]


# -----(4.9) DROP UNNECESSARY COLUMNS------
data = data.drop(columns=['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu', 'Inches', 'Memory'])

data.head()


# -----(4.10) CONVERT CATEGORICAL COLUMNS TO DUMMY VARIABLES & CONVERT THAT INTO 1 & 0------
data = pd.get_dummies(data)

for col in data.select_dtypes(include='bool').columns:
    data[col] = data[col].astype(int)

data.head()


# -----(5) MODEL TRAINING------
# -----(5.1) SPLIT DATA------
x = data.drop('Price_euros', axis=1)
y = data['Price_euros']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# -----(5.2) TRY MULTIPLE MODELS------
def model_accuracy(model):
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"{type(model).__name__} - Train Accuracy: {train_score:.2f} | Test Accuracy: {test_score:.2f}")

lr = LinearRegression()
model_accuracy(lr)

ls = Lasso()
model_accuracy(ls)

dt = DecisionTreeRegressor()
model_accuracy(dt)

rf = RandomForestRegressor()
model_accuracy(rf)


# -----(5.3) GRID SEARCH FOR BEST MODEL------
rf = RandomForestRegressor()

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=2,
                           scoring='r2')

grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)
print("Best Random Forest Parameters:", grid_search.best_params_)
print(f"Tuned RandomForest - Train Accuracy: {train_score:.2f} | Test Accuracy: {test_score:.2f}")


# -----(6) SAVE BEST MODEL------
with open('laptop_price_predictor.pickle', 'wb') as file:
    pickle.dump(best_model, file)

print("Model saved as laptop_price_predictor.pickle")

