
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma


load_dotenv()
df = pd.read_csv("cleaned_tmdb.csv")

df["overview"] = df["overview"].fillna("")
df["genres"] = df["genres"].fillna("")
df["keywords"] = df["keywords"].fillna("")

df["content"] = (
    df["title"] + " " +
    df["genres"] + " " +
    df["keywords"] + " " +
    df["overview"]
)


docs = []

for row in df.itertuples():
    doc = Document(
        page_content=row.content,
        metadata={
            "title": row.title,
            "year": row.year,
            "rating": row.vote_average,
            "runtime": row.runtime,
        },
    )
    docs.append(doc)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(embedding_function=embedding, persist_directory="chroma_db")

for i in range(0, len(docs), 500):
    db.add_documents(docs[i:i + 500])
    print(min(i + 500, len(docs)), "done")
    

print(len(docs), "movies saved")