import os
import joblib
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OMDB_KEY = os.getenv("OMDB_API_KEY")

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_data():
    return joblib.load("movies_df.pkl"), joblib.load("similarity.pkl")


@st.cache_data
def get_poster(title, year):
    if not OMDB_KEY:
        return None
    try:
        r = requests.get(
            "https://www.omdbapi.com/",
            params={"t": title, "y": year, "apikey": OMDB_KEY},
            timeout=5,
        )
        poster = r.json().get("Poster")
        return poster if poster and poster != "N/A" else None
    except Exception:
        return None


df, similarity = load_data()

if "sel" not in st.session_state:
    st.session_state.sel = None


def select(i):
    st.session_state.sel = i


def clear():
    st.session_state.sel = None


def show(indices, prefix):
    for start in range(0, len(indices), 5):
        cols = st.columns(5)
        for col, i in zip(cols, indices[start:start + 5]):
            row = df.iloc[i]
            with col:
                poster = get_poster(row["title"], str(row["year"]))
                if poster:
                    st.image(poster)
                else:
                    st.markdown("🎞️ *poster nahi*")
                st.markdown(f"**{row['title']}**")
                st.caption(f"⭐ {row['vote_average']:.1f} | 📅 {row['year']}")
                st.button("Related dekho", key=f"{prefix}_{i}",
                          on_click=select, args=(i,))


st.title("🎬 Movie Recommender")

if st.session_state.sel is None:
    q = st.text_input("Movie ka naam search karo")
    if q:
        indices = list(df[df["title"].str.contains(q, case=False, na=False)].index[:10])
        st.subheader("Search results")
    else:
        indices = list(df.sort_values("vote_count", ascending=False).index[:10])
        st.subheader("Popular movies (kisi par click karo)")

    if indices:
        show(indices, "home")
    else:
        st.warning("Koi movie nahi mili.")
else:
    i = st.session_state.sel
    st.button("⬅ Wapas", on_click=clear)
    st.subheader(f"'{df.iloc[i]['title']}' jaisi movies")
    scores = sorted(enumerate(similarity[i]), key=lambda x: x[1], reverse=True)[1:11]
    show([j for j, _ in scores], "rel")