# Exploracion inicial darwin
# Se importa pandas
import pandas as pd

# Se carga el archivo
df = pd.read_csv("/Users/joseromerodegaetano/desktop/DS/IST_corrected.csv", encoding="latin-1")

# Tipo de datos
df.info()
# Dimensiones
print(df.shape)
# 5 primeras filas 
print(df.head())
# Estadístia descriptica
print(df.describe())
# Valores nulos por columnas
print(df.isnull().sum())
# Nombres de las columnas
print(df.columns)