import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset
df = pd.read_csv('insurance.csv')


# Chequeamos y eliminamos filas completamente duplicadas
duplicados_antes = df.duplicated().sum()
print(f"Filas duplicadas encontradas y eliminadas: {duplicados_antes}")
df = df.drop_duplicates()

#  Revisar errores imposibles rápidos
df = df[df['age'] > 0]
df = df[df['bmi'] > 0]


# Separamos la variable objetivo (charges) de las features (X)
X = df.drop('charges', axis=1)
y = df['charges']

# Hacemos la separación 80/20 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42 # Fijamos semilla
)

print(f"Tamaño de Train: {X_train.shape[0]} filas")
print(f"Tamaño de Test: {X_test.shape[0]} filas")

# Guardamos índices para outliers
train_indices = X_train.index
test_indices = X_test.index

# Unimos temporalmente X_train e y_train SOLO para la visualización del EDA
df_train_eda = X_train.copy()
df_train_eda['charges'] = y_train

# Configuramos el estilo visual
sns.set_theme(style="whitegrid")

#  Histogramas de las variables numéricas
df_train_eda.hist(figsize=(10, 8), bins=20, color='steelblue')
plt.suptitle("Distribución de Variables Numéricas (Solo Train)", fontsize=16)
plt.tight_layout()
plt.show()

# Boxplots para detectar visualmente los Outliers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df_train_eda, y='charges', ax=axes[0], color='lightcoral')
axes[0].set_title("Boxplot de Cargos (Target)")
sns.boxplot(data=df_train_eda, y='bmi', ax=axes[1], color='mediumaquamarine')
axes[1].set_title("Boxplot de BMI")
plt.tight_layout()
plt.show()

# 3. Matriz de Correlación
plt.figure(figsize=(8, 6))
# Seleccionamos solo las variables numéricas
numericas = df_train_eda.select_dtypes(include=[np.number])
matriz_corr = numericas.corr()

# Graficamos el Heatmap
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Matriz de Correlación de Pearson (Solo Train)", fontsize=14)
plt.show()