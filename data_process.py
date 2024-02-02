import pandas as pd
data = pd.read_parquet("2008.parquet")
try: 
    data_cleaned = pd.read_parquet('2008_cleaned.parquet')
except FileNotFoundError:
    try:
        data = pd.read_parquet("2008.parquet")
    except FileNotFoundError:
        data = pd.read_csv("2008.csv")
    data_cleaned = data.dropna()
        
    data_cleaned.to_parquet('2008_cleaned.parquet')

# Later, read from Parquet (much faster)

columns_with_nan = data_cleaned.columns[data_cleaned.isnull().any()].tolist()
print(data.dropna().shape)




