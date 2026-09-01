# TP1_Equipo-5: Predicción de Costos de Seguro Médico

Este repositorio contiene los notebooks del Trabajo Práctico 1, enfocado en predecir cargos médicos (`charges`) usando modelos de regresión (Lineal y Polinómica) con Scikit-Learn.

##  Archivos del Proyecto

* **`insurance.csv` / `insurance_cleaned.csv`**: Dataset original y versión limpia (sin duplicados y filtrando variables de baja correlación).
* **`01.Limpieza_de_datos.ipynb`**: Análisis Exploratorio (EDA) y preprocesamiento.
* **`02.Regresión_lineal.ipynb`**: Separación Train/Test (80/20) y modelo base (OLS) aplicando escalado seguro con *Pipelines*.
* **`03.Regresión_polynomica.ipynb`**: Búsqueda de hiperparámetros (`GridSearchCV`) probando grados del 1 al 4 y regularización L1 (Lasso).
* **`04.Evaluación.ipynb` / `05.Comparación_de_modelos`**: Evaluación final sobre el conjunto de Test oculto y tabla comparativa.

---

##  Resultados y Conclusión

El de mejor generalización se logró con el **Polinomio de Grado 2**.

**Métricas Finales (en el 20% de Test no visto):**
* **Modelo Lineal Base:** RMSE: 6676.13 | R²: 0.716
* ** Modelo Ganador (Grado 2 + Lasso alpha=17.5):** RMSE: 5537.92 | R²: 0.804

**Conclusión:** Nuestro modelo óptimo explica el **80.4% de la varianza** de los costos y logra reducir el error promedio de predicción en más de **$1,100 dólares** respecto al modelo base. La implementación estricta de *Pipelines* garantizó la ausencia total *data leakage*.

---

##  Librerias
**Python 3** | **Pandas & NumPy** | **Scikit-Learn** | **Matplotlib & Seaborn**
