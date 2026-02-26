# Exploracion inicial darwin
# Se importan las librerias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Proporción 'P' y 'H'
conteo = df['class'].value_counts()
print(conteo)
# Proporción 'P' y 'H' en porcentaje
print(df['class'].value_counts(normalize=True))

# Número de filas y columnas
filas = df.shape[0]
columnas = df.shape[1]
print(f"El dataset tiene {filas} filas y {columnas} columnas")

# Mostrar primer filas
df.iloc[:, 0]

# Gráfico P vs H
sns.countplot(x="class", data=df)
plt.title("Proporción pacientes (P) vs sanos (H)")
plt.show()

# Proporción pacientes
resumen_clases = df.groupby('class').mean(numeric_only=True)
print(resumen_clases.iloc[:, :18]) # Para las 18 primeras variables
