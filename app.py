import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


db=Chroma(persist_directory="chroma_db", embedding_function=embedding)

print("Total movies in db:", db._collection.count())

retriever = db.as_retriever(search_kwargs={"k": 5})

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

prompt = PromptTemplate(
    template="""You are a movie recommender.
Use only the movies given below. Do not add movies from outside.
Suggest the best picks for the user's request.
For each pick, give the title and one line reason.

Movies:
{movies}

User request: {query}""",
    input_variables=["movies", "query"],
)

parser=StrOutputParser()


chain=prompt|model|parser




st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.card {
    background: #1c1f26;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    border-left: 5px solid #e50914;
    min-height: 170px;
}
.card h4 { margin: 0 0 6px 0; color: #ffffff; }
.meta { color: #f5c518; font-size: 14px; margin-bottom: 8px; }
.desc { color: #c9ccd3; font-size: 13px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


st.sidebar.header("Filters")
k = st.sidebar.slider("Kitni movies?", 3, 10, 6)
min_rating = st.sidebar.slider("Minimum rating", 0.0, 9.0, 0.0, 0.5)
max_runtime = st.sidebar.slider("Max runtime (minutes)", 60, 240, 240, 10)

st.title("🎬 Movie Recommender")
st.caption("Jaise likho: 'space me survival wali movie' ya 'twist wali thriller'")

query = st.text_input("Kaisi movie chahiye?")

if st.button("Suggest karo 🍿") and query:
    results = db.similarity_search(query, k=100)
    found = [
        d for d in results
        if d.metadata["rating"] >= min_rating
        and d.metadata["runtime"] <= max_runtime
    ][:k]

    if not found:
        st.warning("Koi movie nahi mili. Filters kam karo.")
    else:
        cols = st.columns(3)
        for i, doc in enumerate(found):
            m = doc.metadata
            text = doc.page_content[:150] + "..."
            with cols[i % 3]:
                st.markdown(f"""
                <div class="card">
                    <h4>{m['title']}</h4>
                    <div class="meta">⭐ {m['rating']:.1f} | 📅 {m['year']} | ⏱ {int(m['runtime'])} min</div>
                    <div class="desc">{text}</div>
                </div>
                """, unsafe_allow_html=True)

        movies_text = "\n\n".join(d.page_content for d in found)
        with st.spinner("AI soch raha hai..."):
            answer = chain.invoke({"movies": movies_text, "query": query})

        st.subheader("🤖 AI ki pasand")
        st.write(answer)