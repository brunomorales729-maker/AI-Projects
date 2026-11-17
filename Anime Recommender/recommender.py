import os
import gdown
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

DRIVE_URL_RATING = "https://drive.google.com/uc?id=1O1TbLg5H8UFJM2DsBm5cLrGzKz-lQVrD"

def ensure_ratings_exist(rating_path="rating.csv"):
    """Descarga rating.csv desde Google Drive si no existe."""
    if not os.path.exists(rating_path):
        gdown.download(DRIVE_URL_RATING, rating_path, quiet=False)
    return rating_path

def load_and_preprocess_data(anime_path='anime.csv', rating_path='rating.csv'):
    # Verificar y descargar rating.csv si falta
    rating_path = ensure_ratings_exist(rating_path)
    
    # 1. Cargar metadatos
    df_anime = pd.read_csv(anime_path)
    df_anime.dropna(subset=['genre', 'rating', 'type'], inplace=True)
    
    # 2. Cargar ratings optimizando tipos de datos (reduce drásticamente el uso de RAM)
    dtypes = {
        'user_id': 'int32',
        'anime_id': 'int32',
        'rating': 'int8'
    }
    df_ratings = pd.read_csv(rating_path, dtype=dtypes)
    df_ratings = df_ratings[df_ratings['rating'] > 0]
    
    # 3. Filtrar para reducir la matriz y evitar que Streamlit Cloud se quede sin RAM:
    # Conservamos animes con al menos 250 calificaciones y usuarios con al menos 50 calificaciones
    anime_counts = df_ratings['anime_id'].value_counts()
    popular_animes = anime_counts[anime_counts >= 250].index
    df_ratings = df_ratings[df_ratings['anime_id'].isin(popular_animes)]
    
    user_counts = df_ratings['user_id'].value_counts()
    active_users = user_counts[user_counts >= 50].index
    df_ratings = df_ratings[df_ratings['user_id'].isin(active_users)]
    
    return df_anime, df_ratings

def get_genres_list(df_anime):
    genres = set()
    for item in df_anime['genre'].dropna():
        for g in item.split(','):
            genres.add(g.strip())
    return sorted(list(genres))

# 1. Recomendador para usuarios nuevos (Géneros)
def recommend_by_genres(df_anime, selected_genres, anime_type='TV', top_n=10):
    subset = df_anime.copy()
    if anime_type != 'Todos':
        subset = subset[subset['type'] == anime_type]
    
    pattern = '|'.join(selected_genres)
    subset = subset[subset['genre'].str.contains(pattern, case=False, na=False)]
    
    if subset.empty:
        return pd.DataFrame()
    
    C = subset['rating'].mean()
    m = subset['members'].quantile(0.60)
    
    q_anime = subset[subset['members'] >= m].copy()
    q_anime['weighted_score'] = (
        (q_anime['members'] / (q_anime['members'] + m) * q_anime['rating']) +
        (m / (q_anime['members'] + m) * C)
    )
    
    return q_anime.sort_values(by='weighted_score', ascending=False).head(top_n)

# 2. Recomendador Colaborativo con PCA (Ligero en memoria)
def build_pca_similarity_matrix(df_ratings, n_components=30):
    """
    Construye la matriz con tipos flotantes de 32 bits y 30 componentes PCA
    para mantenerse por debajo de los límites de memoria de la nube.
    """
    user_item = df_ratings.pivot_table(
        index='anime_id', 
        columns='user_id', 
        values='rating', 
        fill_value=0
    ).astype(np.float32)
    
    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=n_components, random_state=42)
    anime_pca_matrix = pca.fit_transform(user_item.values)
    
    # Similitud coseno
    similarity = cosine_similarity(anime_pca_matrix)
    anime_ids = list(user_item.index)
    
    return pd.DataFrame(similarity, index=anime_ids, columns=anime_ids)

def recommend_from_user_ratings(user_ratings, sim_matrix, df_anime, top_n=10):
    scores = pd.Series(dtype=float)
    
    for anime_id, rating in user_ratings.items():
        if anime_id in sim_matrix.index:
            weight = rating - 5.0
            sim_scores = sim_matrix[anime_id] * weight
            scores = scores.add(sim_scores, fill_value=0)
            
    for anime_id in user_ratings.keys():
        if anime_id in scores:
            scores.drop(anime_id, inplace=True)
            
    top_anime_ids = scores.sort_values(ascending=False).head(top_n).index
    return df_anime[df_anime['anime_id'].isin(top_anime_ids)]
