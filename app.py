import streamlit as st
import pandas as pd
from recommender import (
    load_and_preprocess_data, 
    get_genres_list, 
    recommend_by_genres,
    build_pca_similarity_matrix, # <-- IMPORTAR LA NUEVA FUNCIÓN
    recommend_from_user_ratings
)

st.set_page_config(page_title="Sistema de Recomendación de Anime", layout="wide")

@st.cache_data
def get_data():
    return load_and_preprocess_data()

@st.cache_data
def get_similarity(df_ratings):
    return build_pca_similarity_matrix(df_ratings, n_components=50)

# Interfaz inicial
st.title("🎌 Sistema de Recomendación de Anime")
st.write("Encuentra tu próximo anime según tus géneros favoritos o calificando títulos conocidos.")

df_anime, df_ratings = get_data()
all_genres = get_genres_list(df_anime)

# Pestañas para cada tipo de recomendación
tab1, tab2 = st.tabs(["🆕 Recomendación por Géneros (Nuevos Usuarios)", "⭐ Recomendación por Calificaciones"])

# --- TAB 1: PARA USUARIOS NUEVOS ---
with tab1:
    st.header("Descubre por Géneros")
    st.write("Selecciona tus temáticas de preferencia y formato para obtener las mejores recomendaciones.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_genres = st.multiselect(
            "Elige uno o más géneros:",
            options=all_genres,
            default=["Action", "Adventure"] if "Action" in all_genres else []
        )
    with col2:
        anime_type = st.selectbox("Formato:", ["Todos", "TV", "Movie", "OVA"])
        
    top_k = st.slider("Número de recomendaciones:", min_value=5, max_value=20, value=10)

    if st.button("Buscar Animes", key="btn_genre"):
        if selected_genres:
            results = recommend_by_genres(df_anime, selected_genres, anime_type, top_n=top_k)
            if not results.empty:
                st.subheader("Títulos Recomendados:")
                for _, row in results.iterrows():
                    with st.container():
                        st.markdown(f"### **{row['name']}**")
                        st.caption(f"**Tipo:** {row['type']} | **Episodios:** {row['episodes']} | **Rating:** ⭐ {row['rating']}/10 | **Miembros:** {row['members']:,}")
                        st.write(f"**Géneros:** `{row['genre']}`")
                        st.divider()
            else:
                st.warning("No se encontraron animes para esa combinación.")
        else:
            st.error("Por favor selecciona al menos un género.")

# --- TAB 2: PARA USUARIOS CON CALIFICACIONES ---
with tab2:
    st.header("Califica Animes y Recibe Recomendaciones")
    st.write("Selecciona algunos títulos populares que hayas visto y dales una calificación.")
    
    sim_matrix = get_similarity(df_ratings)
    
    # Lista de animes populares para que el usuario califique
    popular_animes = df_anime[df_anime['anime_id'].isin(sim_matrix.index)].sort_values(by='members', ascending=False).head(30)
    
    user_ratings = {}
    st.write("#### Califica del 1 al 10 los que hayas visto:")
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(popular_animes.head(9).iterrows()):
        col = cols[i % 3]
        with col:
            st.markdown(f"**{row['name']}**")
            rating = st.slider(f"Nota:", min_value=0, max_value=10, value=0, key=f"rate_{row['anime_id']}")
            if rating > 0:
                user_ratings[row['anime_id']] = rating
                
    if st.button("Generar Recomendaciones Personalizadas", key="btn_collab"):
        if user_ratings:
            recs = recommend_from_user_ratings(user_ratings, sim_matrix, df_anime, top_n=top_k)
            st.subheader("Basado en tus calificaciones, te sugerimos:")
            for _, row in recs.iterrows():
                with st.container():
                    st.markdown(f"### **{row['name']}**")
                    st.caption(f"**Tipo:** {row['type']} | **Episodios:** {row['episodes']} | **Rating:** ⭐ {row['rating']}/10")
                    st.write(f"**Géneros:** `{row['genre']}`")
                    st.divider()
        else:
            st.warning("Califica al menos un anime (asigna una puntuación mayor a 0).")