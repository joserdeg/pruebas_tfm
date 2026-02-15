# Exploracion inicial migraña kiel
# Se importa pandas
import pandas as pd

# Se carga el archivop
df = pd.read_excel("/Users/joseromerodegaetano/Desktop/Evaluating treatment success in CGRP antibody prophylaxis.xlsx")

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