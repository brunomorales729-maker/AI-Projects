import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

def load_and_preprocess_data(anime_path='anime.csv', rating_path='rating.csv'):
    df_anime = pd.read_csv(anime_path)
    df_anime.dropna(subset=['genre', 'rating', 'type'], inplace=True)
    
    df_ratings = pd.read_csv(rating_path)
    df_ratings = df_ratings[df_ratings['rating'] > 0]
    
    user_counts = df_ratings['user_id'].value_counts()
    anime_counts = df_ratings['anime_id'].value_counts()
    
    df_ratings = df_ratings[df_ratings['user_id'].isin(user_counts[user_counts >= 30].index)]
    df_ratings = df_ratings[df_ratings['anime_id'].isin(anime_counts[anime_counts >= 100].index)]
    
    return df_anime, df_ratings

def get_genres_list(df_anime):
    genres = set()
    for item in df_anime['genre'].dropna():
        for g in item.split(','):
            genres.add(g.strip())
    return sorted(list(genres))

# 1. Recomendador de Cold Start (Por Géneros)
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

# 2. Recomendador Colaborativo OPTIMIZADO CON PCA
def build_pca_similarity_matrix(df_ratings, n_components=50):
    """
    Construye una matriz de similitud de animes utilizando PCA para comprimir 
    las dimensiones de los usuarios y hacer el sistema altamente eficiente.
    """
    # 1. Crear matriz usuario-ítem y rellenar NAs con 0
    user_item = df_ratings.pivot_table(index='anime_id', columns='user_id', values='rating').fillna(0)
    
    # 2. Aplicar PCA para reducción de dimensionalidad
    # Reducimos las miles de columnas de usuarios a solo 'n_components' (ej. 50 características principales)
    pca = PCA(n_components=n_components, random_state=42)
    anime_pca_matrix = pca.fit_transform(user_item.values)
    
    # 3. Calcular la similitud del coseno sobre la matriz reducida (mucho más rápido)
    similarity = cosine_similarity(anime_pca_matrix)
    
    # 4. Formatear la salida como DataFrame para facilitar la búsqueda
    anime_ids = list(user_item.index)
    sim_df = pd.DataFrame(similarity, index=anime_ids, columns=anime_ids)
    
    return sim_df

def recommend_from_user_ratings(user_ratings, sim_matrix, df_anime, top_n=10):
    """
    user_ratings: diccionario {anime_id: calificacion}
    """
    scores = pd.Series(dtype=float)
    
    for anime_id, rating in user_ratings.items():
        if anime_id in sim_matrix.index:
            weight = rating - 5.0 # Centrar calificación
            sim_scores = sim_matrix[anime_id] * weight
            scores = scores.add(sim_scores, fill_value=0)
            
    for anime_id in user_ratings.keys():
        if anime_id in scores:
            scores.drop(anime_id, inplace=True)
            
    top_anime_ids = scores.sort_values(ascending=False).head(top_n).index
    return df_anime[df_anime['anime_id'].isin(top_anime_ids)]