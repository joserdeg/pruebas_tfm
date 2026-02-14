# Exploracion inicial con pandas
# Se importa pandas
import pandas as pd

# Se carga el archivo
df = pd.read_csv("/Users/joseromerodegaetano/desktop/DARWIN/DARWIN.csv")

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

