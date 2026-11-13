import pandas as pd 
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split

df=pd.read_csv("cleaned_tmdb.csv")
df["overview"]=df["overview"].fillna("")
df["genres"]=df["genres"].fillna("")
df["keywords"]=df["keywords"].fillna("")

df["soup"]=df["overview"] + " " + df["genres"] + " " + df["keywords"]


tfidf=TfidfVectorizer(stop_words="english",max_features=5000)
matrix=tfidf.fit_transform(df["soup"])

similarity=cosine_similarity(matrix)


joblib.dump(tfidf,"tfidf.pkl")
joblib.dump(similarity,"similarity.pkl")
joblib.dump(df,"movies_df.pkl")

similarity=similarity.astype("float32")

movie_name = input("Enter movie name: ")

if movie_name not in df["title"].values:
    print("Movie nahi mili")
else:
    index = df[df["title"] == movie_name].index[0]
    scores = similarity[index]
    top_movies = scores.argsort()[::-1][1:6]

    for i in top_movies:
        print(df["title"][i])