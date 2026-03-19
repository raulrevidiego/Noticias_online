# Estudio de Online News Popularity Dataset

## Descripción
Análisis exploratorio y modelado predictivo del dataset [Online News Popularity](https://www.kaggle.com/datasets/deepakshende/onlinenewspopularity) de Kaggle, que contiene ~39.000 artículos de Mashable con 58 variables predictoras y el número de shares como variable objetivo.

---

## 1. Limpieza de datos

- Se eliminaron las columnas **`url`** y **`timedelta`** por no ser variables predictoras.
- Las columnas del dataset original tienen **espacios al inicio** en sus nombres — se resolvió con `df.columns.str.strip()`.
- Se aplicó **`np.abs()`** a todo el dataset para convertir valores negativos en positivos, especialmente relevante en las variables de polaridad.
- Se eliminaron columnas con escaso poder predictivo:
  - `kw_max_max` — no significativa estadísticamente (p=0.676)
  - `kw_min_min` — valores artificiales/centinela detectados
  - `kw_min_max` — correlación prácticamente nula (r=0.011)
  - `kw_avg_max` — correlación muy débil (r=0.060)
  - `min_negative_polarity` y `max_negative_polarity` — distribuciones distorsionadas tras aplicar `abs()`

---

## 2. Variables de tokens

- Se analizaron las variables relacionadas con la longitud y estructura del texto (número de palabras, tokens, imágenes, vídeos...) agrupadas en bins y comparadas con el share medio.
- Ninguna variable de tokens muestra una relación lineal fuerte con los shares, aunque algunas muestran tendencias suaves.

---

## 3. Canal de publicación

- Los canales con mayor volumen total de shares son los de **tecnología** y **entretenimiento**.
- Las diferencias entre canales son notables, lo que sugiere que el tema del artículo influye en su difusión.

---

## 4. Variables subjetivas (sentimiento y polaridad)

### Hallazgos principales:
- El dataset tiene un **sesgo claro hacia el lenguaje positivo**: los artículos usan más palabras positivas, con mayor intensidad y mayor dispersión que las negativas.
- `global_rate_positive_words` domina sobre `global_rate_negative_words` — las palabras negativas se concentran cerca de 0.
- `avg_positive_polarity` se centra en torno a 0.35 frente al 0.15 de la negativa — el tono positivo promedio es más intenso.
- `max_positive_polarity` tiene una masa enorme en 1.0: en la mayoría de artículos hay al menos una palabra con polaridad positiva máxima.

### Advertencia:
Las variables `min_negative_polarity` y `max_negative_polarity` quedaron distorsionadas tras aplicar `abs()` (sus valores originalmente negativos se invirtieron), por lo que se eliminaron del análisis.

---

## 5. Palabras clave (variables `kw_*`)

Las 9 variables kw forman una matriz 3×3 (min/max/avg × min/max/avg de shares de las palabras clave).

| Variable | r con shares | Conclusión |
|---|---|---|
| `kw_avg_avg` | 0.222 | ⭐ La más predictiva del grupo |
| `kw_max_avg` | 0.109 | Moderada |
| `kw_min_avg` | 0.109 | Moderada |
| `kw_avg_min` | 0.040 | Débil |
| `kw_max_min` | 0.033 | Débil |
| `kw_min_min` | 0.023 | Muy débil + valores artificiales |
| `kw_avg_max` | 0.060 | Débil |
| `kw_min_max` | 0.011 | Casi nula |
| `kw_max_max` | 0.002 | **No significativa** (p=0.676) |

**Conclusión clave:** artículos que tratan temas cuyas palabras clave ya son populares (`kw_avg_avg` alto) tienden a conseguir más shares — el tema importa tanto como el contenido.

---

## 6. Modelado predictivo

### 6.1 Regresión — Predicción de shares
- Modelo: **Random Forest Regressor** con `n_estimators=100`.
- Se aplicó **`np.log1p()`** al target para reducir el efecto de outliers extremos (artículos con cientos de miles de shares).
- Las métricas se calculan en escala original (deshaciendo con `np.expm1()`) y en escala log.
- El **R² en escala log es más representativo** del ajuste real; el R² en escala original queda penalizado por outliers inevitables.
- Resultado típico: R² ≈ 0.10–0.20, consistente con la literatura sobre este dataset — la viralidad es inherentemente difícil de predecir.

### 6.2 Clasificación binaria — Popular / No popular
- Umbral: **mediana de shares**.
- Modelo: **Random Forest Classifier** con `n_estimators=100`.
- Métricas reportadas: Accuracy, Precision, Recall, **F2-score** (penaliza más los falsos negativos).

### 6.3 Clasificación en 4 clases
- División por **cuartiles** (Q1, Q2, Q3):

```
0 ──── Q1 ──── Q2 ──── Q3 ──── MAX
  Nada    Poco    Algo    Muy
popular popular popular popular
```

- Las clases extremas (**nada popular** y **muy popular**) se predicen mejor que las intermedias, que son más ambiguas.
- El accuracy es menor que en el modelo binario — esperado al tener más clases.

### 6.4 Nota sobre eliminación de columnas
La eliminación de variables débiles **no mejoró los modelos**, lo cual es normal: Random Forest es robusto a variables irrelevantes por diseño, ya que en cada árbol solo evalúa un subconjunto aleatorio de columnas y las débiles se ignoran de forma natural.

---

## 7. Recomendaciones para mejorar los modelos

Para obtener mejores métricas se recomienda explorar:
- **XGBoost o LightGBM** — suelen superar a Random Forest en datasets tabulares.
- **Gradient Boosting** con `learning_rate` bajo y más estimadores.
- **Tuning de hiperparámetros** con `GridSearchCV` o `RandomizedSearchCV`.
- **Feature engineering** — crear nuevas variables a partir de las existentes (ratios, interacciones).

---

## Tecnologías utilizadas

- Python 3
- `pandas`, `numpy` — manipulación de datos
- `matplotlib`, `seaborn` — visualización
- `scikit-learn` — modelado predictivo
- `scipy` — estadística (regresión lineal, correlaciones)
