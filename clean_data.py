import ast
import pandas as pd 
df=pd.read_csv("tmdb_5000_movies.csv")
# print(df.head(5))

# print(df.info())
# print((df.isnull()*100).mean())

df.drop(columns=["budget","revenue","homepage","original_title","production_companies","production_countries","spoken_languages","status","popularity"],inplace=True)
df.drop(columns=["tagline"],inplace=True)
df.dropna(subset=["overview","release_date","runtime"],inplace=True)
# print(df.isnull().sum())
# print(df.shape)

def get_name(text):
    try:
        return " ,".join(d["name"] for d in ast.literal_eval(text))
    except(ValueError,SyntaxError):
        return ""

df["genres"]=df["genres"].apply(get_name)
df["keywords"]=df["keywords"].apply(get_name)

df["year"]=df["release_date"].str[:4]

df.drop_duplicates(subset="title",inplace=True)

df.reset_index(drop=True,inplace=True)

df.to_csv("cleaned_tmdb.csv", index=False)
print("complete the eda")