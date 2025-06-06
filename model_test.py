# -----(1) IMPORTS------
import pandas as pd
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


# -----(5) LOAD TRAINED MODEL AND MAKE TEST PREDICTION------
# -----(5.1) PREPARE INPUT DATA FOR PREDICTION------
X = data.drop('Price_euros', axis=1, errors='ignore')


# -----(5.2) LOAD TRAINED MODEL FROM .pickle FILE------
with open("laptop_price_predictor.pickle", "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully.")


# -----(5.3) MAKE PREDICTION ON A SINGLE ROW------
prediction = model.predict(X.iloc[[0]])
print(f"Predicted Price: €{prediction[0]:.2f}")


# -----(5.4) COMPARE WITH ACTUAL PRICE------
actual_price = data['Price_euros'].iloc[0]
print(f"Actual Price: €{actual_price:.2f}")