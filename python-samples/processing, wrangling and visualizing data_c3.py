import csv
import pandas as pd, numpy as np
from sklearn import preprocessing


## DATA COLLECTION
filepath = "C:/Users/ferre/PycharmProjects/processing, wrangling and visualizing data/fruits.csv"
csv_reader = csv.reader(open(filepath), delimiter=',')

csv_rows = list()
csv_attr_dict = {'sno': [], 'fruit': [], 'color': [], 'price': []}
for row in csv_reader:
    csv_rows.append(row)

for row in csv_rows[1:]:
    csv_attr_dict['sno'].append(row[0])
    csv_attr_dict['fruit'].append(row[1])
    csv_attr_dict['color'].append(row[2])
    csv_attr_dict['price'].append(row[3])
#print(csv_attr_dict)

# Another way of doing it
df = pd.read_csv(filepath, sep=',')
#print(df)


## DATA WRANGLING
print("N. of Rows: ", df.shape[0])
print("N. of Columns: ", df.shape[1])
print("Column Names: ", df.columns.values.tolist())
print("Column Data Types: ", df.dtypes)
print("Columns swith Missing Values: ", df.columns[df.isnull().any()].tolist())
print("Number of rows with Missing Values: ", len(pd.isnull(df).any(1).to_numpy().nonzero()[0].tolist()))
print("Sample Indices with Missing Data: ", pd.isnull(df).any(1).to_numpy().nonzero()[0].tolist()[0:5])

print("General Stats:")
print(df.info())

print("Summary Stats:")
print(df.describe())

## DUPLICATED AND MISSING VALUES
print("Duplicated values in fruit: ")
print(df.fruit.duplicated())

print("Drop duplicated values in fruit: ")
df.drop_duplicates( subset='fruit', keep=False, inplace=True)

print("Fruit column after drop rows with missing values: ")
df.fruit.dropna(how="any", inplace=True)
print(df)

print("Fill missing values with mean price: ")
df.price.fillna(value=np.round(df.price.mean(), decimals=2), inplace=True)
print(df)

print("Fill missing values with value from PREVIOUS row: ")
df['fruit'].fillna(method='ffill', inplace=True)

print("Fill missing values with value from NEXT row: ")
df['fruit'].fillna(method='bfill', inplace=True)

## NORMALIZING VALUES
print("Normalizing prices with min-max Scaler: ")
df_normalized = df.dropna().copy()
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_normalized['price'].values.reshape(-1,1))
df_normalized['normalized_price'] = np_scaled.reshape(-1,1)




