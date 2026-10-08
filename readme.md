# Sistema de Recomendación de Anime (PCA & Filtrado Colaborativo)

Aplicación interactiva desarrollada en Python y Streamlit que ofrece recomendaciones personalizadas de anime. Implementa un motor híbrido:
1. **Cold-start (Nuevos usuarios):** Recomendación ponderada basada en filtrado por géneros y tipo de formato.
2. **Usuarios recurrentes:** Filtrado colaborativo optimizado mediante reducción de dimensionalidad con **PCA (Análisis de Componentes Principales)** y similitud de coseno.

---

## 📂 Obtención y Preparación de la Base de Datos

Por limitaciones de almacenamiento en GitHub, los archivos de datos no están incluidos en el repositorio. Para ejecutar la aplicación:

1. Descarga el conjunto de datos **⛩️Anime Ratings 〽️Analysis & 🤖Recommender System** desde Kaggle:
   > URL: https://www.kaggle.com/code/hasibalmuzdadid/anime-ratings-analysis-recommender-system
2. Crea una carpeta llamada `data/` en la raíz del proyecto si no existe.
3. Descomprime y coloca los dos archivos CSV dentro de esa carpeta con los siguientes nombres:
   - `anime.csv`
   - `rating.csv`
---
