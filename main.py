
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("insurance.csv")

print("--- 1. ESTRUCTURA DEL DATASET ---")
print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
print(df.head())




print("\n--- 2. TIPOS DE DATOS ---")
print(df.dtypes)



print("\n--- 3. DUPLICADOS EXACTOS ---")
n_dup = df.duplicated().sum()
print(f"Filas duplicadas : {n_dup}")

if n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Nuevo tamaño del dataset: {df.shape[0]} filas.")




print("\n--- 4. VALORES FALTANTES ---")
print(df.isna().sum())




print("\n--- 5. CHEQUEO DE VALORES IMPOSIBLES ---")
print("Edades <= 0:", (df["age"] <= 0).sum())
print("BMI <= 0:", (df["bmi"] <= 0).sum())
print("Cargos <= 0:", (df["charges"] <= 0).sum())
print("Hijos < 0:", (df["children"] < 0).sum())


print("\n--- 6. FRECUENCIAS DE VARIABLES CATEGÓRICAS ---")
for col in ["sex", "smoker", "region"]:
    print(f"\nColumna: {col}")
    print(df[col].value_counts())



df.to_csv("insurance_limpio.csv", index=False)
print("\n Dataset guardado ")



#hacemos el split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print("--- SEPARACIÓN REALIZADA ---")
print(f"Dataset completo: {len(df)} filas")
print(f"TRAIN: {len(train_df)} filas")
print(f"TEST: {len(test_df)} filas")


print("\n--- Variables numéricas: comparativa de medias ---")
num_cols = ["age", "bmi", "children", "charges"]
comparacion_num = pd.DataFrame({
    "train_media": train_df[num_cols].mean().round(2),
    "test_media": test_df[num_cols].mean().round(2)
})
print(comparacion_num)

print("\n--- Variables categóricas: comparativa de proporciones (%) ---")
for col in ["sex", "smoker", "region"]:
    print(f"\nColumna: {col}")
    prop = pd.DataFrame({
        "train_%": (train_df[col].value_counts(normalize=True) * 100).round(1),
        "test_%": (test_df[col].value_counts(normalize=True) * 100).round(1),
    })
    print(prop)



# EDA Y PREPROCESAMIENTO

columnas_num = ['age', 'bmi', 'children']
columnas_cat = ['sex', 'smoker', 'region']
todas_num = columnas_num + ['charges']



columnas_num = ['age', 'bmi', 'children', 'charges']
columnas_cat = ['sex', 'smoker', 'region']


print("--- 2. ESTADÍSTICAS DESCRIPTIVAS (Solo Train) ---")
print(train_df[columnas_num].describe().round(2))

# Histogramas
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(columnas_num):
    axes[i].hist(train_df[col], bins=20, color='steelblue', edgecolor='black')
    axes[i].set_title(f"Distribución de {col}")
plt.tight_layout()
plt.show()



plt.figure(figsize=(7, 5))
matriz_corr = train_df[columnas_num].corr()
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matriz de Correlación de Pearson (Variables Numéricas - Solo Train)")
plt.show()


print("\n--- 4. DETECCIÓN DE OUTLIERS POR IQR Y BÚSQUEDA DE CAUSAS ---")

# Detección matemática por IQR en la variable 'charges'
q1 = train_df['charges'].quantile(0.25)
q3 = train_df['charges'].quantile(0.75)
iqr = q3 - q1
lim_sup = q3 + 1.5 * iqr

es_outlier = train_df['charges'] > lim_sup
print(f"Límite superior IQR para charges: {lim_sup:,.2f}")
print(f"Outliers detectados en charges: {es_outlier.sum()} ({es_outlier.mean()*100:.1f}% del train)")


print("\n>>> ANÁLISIS COMPARATIVO: ¿Qué variable cambia drásticamente en los outliers? <<<")

print("\n--- A. Variables Categóricas (% de distribución en cada grupo) ---")
for col in columnas_cat:
    tabla_cat = pd.DataFrame({
        "Resto (Normal) %": (train_df.loc[~es_outlier, col].value_counts(normalize=True) * 100).round(1),
        "Outliers (Charges altos) %": (train_df.loc[es_outlier, col].value_counts(normalize=True) * 100).round(1)
    })
    tabla_cat["Diferencia (pp)"] = tabla_cat["Outliers (Charges altos) %"] - tabla_cat["Resto (Normal) %"]
    print(f"\nVariable: {col}")
    print(tabla_cat)

print("\n--- B. Variables Numéricas (Promedio en cada grupo) ---")
print(pd.DataFrame({
    "Media en Resto (Normal)": train_df.loc[~es_outlier, ['age', 'bmi', 'children']].mean(),
    "Media en Outliers": train_df.loc[es_outlier, ['age', 'bmi', 'children']].mean()
}).round(2))


# Gráfico de confirmación de las dos poblaciones
plt.figure(figsize=(8, 4))
sns.histplot(data=train_df, x='charges', hue='smoker', bins=30, kde=True)
plt.title("Distribución de Charges según Smoker")
plt.show()



print("\n--- 5. APLICANDO TRANSFORMACIONES (Fit en Train, Transform en Test) ---")

# A. One-Hot Encoding para categóricas
cols_num_features = ['age', 'bmi', 'children']
encoder = OneHotEncoder(sparse_output=False)
cat_train_encoded = encoder.fit_transform(train_df[columnas_cat])
cat_test_encoded = encoder.transform(test_df[columnas_cat])
nombres_cat = encoder.get_feature_names_out(columnas_cat)

# B. Normalización Z-Score para numéricas 
scaler = StandardScaler()
num_train_scaled = scaler.fit_transform(train_df[cols_num_features])
num_test_scaled = scaler.transform(test_df[cols_num_features])

# C. Reconstrucción final de DataFrames
columnas_todas = cols_num_features + list(nombres_cat)

X_train_final = pd.DataFrame(
    np.hstack((num_train_scaled, cat_train_encoded)),
    columns=columnas_todas,
    index=train_df.index
)
y_train_final = train_df['charges']

X_test_final = pd.DataFrame(
    np.hstack((num_test_scaled, cat_test_encoded)),
    columns=columnas_todas,
    index=test_df.index
)
y_test_final = test_df['charges']

print("\n--- PREPROCESAMIENTO FINALIZADO ---")
print(f"X_train_final shape: {X_train_final.shape}")
print(f"X_test_final  shape: {X_test_final.shape} (Intacto y procesado sin fugas)")


print("\n--- 6. MATRIZ DE CORRELACIÓN DE PEARSON (Dataset Completo Procesado) ---")

# Unimos las features procesadas con el target para ver la matriz global
df_corr_completa = X_train_final.copy()
df_corr_completa['charges'] = y_train_final

plt.figure(figsize=(10, 8))
matriz_corr_global = df_corr_completa.corr()
sns.heatmap(matriz_corr_global, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matriz de Correlación de Pearson - Todas las Features (Solo Train)")
plt.show()