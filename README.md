# 🎬 Movie Recommender

A movie recommendation system built with **Python, LangChain, Chroma and Streamlit**. It combines a classic content-based model (TF-IDF) with an AI-powered semantic recommender (embeddings + RAG + LLM), and compares the two.

You can use it in two ways:

1. **Describe what you want** (for example, *"space survival movie"* or *"thriller with a twist"*) and get recommendations with a short reason for each pick.
2. **Pick a movie** and click through related movies, with posters, one after another.

---

## Features

- Data cleaning and feature engineering with **pandas**
- **TF-IDF + cosine similarity** baseline model (trained on movie overview, genres and keywords)
- **Semantic search** using sentence embeddings stored in a **Chroma** vector database
- **LangChain RAG chain**: retriever, prompt template, LLM and output parser
- Filters for **minimum rating** and **maximum runtime**
- Click-to-explore UI with **movie posters** (OMDb API)
- Runs locally. Embeddings are free and local (no embedding API limits)

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data processing | pandas, scikit-learn, joblib |
| LLM framework | LangChain (`langchain-core`, `langchain-chroma`, `langchain-huggingface`, `langchain-openai`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (via sentence-transformers) |
| Vector database | Chroma |
| LLM | `gpt-4o-mini` through OpenRouter |
| UI | Streamlit |
| Posters | OMDb API |
| Config | python-dotenv |

---

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata) from Kaggle. The file used is `tmdb_5000_movies.csv`.

**Cleaning steps (`clean_data.py`):**

1. Drop columns that do not help recommendations: `budget`, `revenue`, `homepage`, `original_title`, `production_companies`, `production_countries`, `spoken_languages`, `status`, `popularity`, `tagline`.
2. Remove rows with missing `overview`, `release_date` or `runtime`.
3. Convert `genres` and `keywords` from JSON-like text into plain names (for example `Action, Adventure`) using `ast.literal_eval`.
4. Create a `year` column from `release_date`.
5. Remove duplicate titles and reset the index.
6. Save the result as `cleaned_tmdb.csv` (about 4,796 movies).

---

## How It Works

### 1. Baseline model: TF-IDF (`train_model.py`)

- Builds a text "soup" for each movie: `overview + genres + keywords`.
- `TfidfVectorizer(stop_words="english", max_features=5000)` is fitted on the soup, giving a matrix of shape (4796, 5000).
- Cosine similarity gives a (4796, 4796) similarity matrix.
- Saves `tfidf.pkl`, `similarity.pkl` and `movies_df.pkl` using joblib.
- A recommend function returns the top 5 most similar movies for a given title.

### 2. Semantic index: embeddings + Chroma (`build_index.py`)

- Each movie becomes a LangChain `Document`.
  - `page_content`: title, genres, keywords and overview.
  - `metadata`: title, year, rating (`vote_average`) and runtime.
- Documents are embedded with `all-MiniLM-L6-v2` and stored in Chroma (`chroma_db/`).
- Run once. Takes a few minutes on a laptop.

### 3. AI recommender (`app.py`)

- User types a request in plain language.
- Chroma returns the 100 most similar movies. Rating and runtime filters are then applied in Python, and the top `k` are kept.
- The retrieved movies are passed to the LLM through a LangChain chain: `PromptTemplate | ChatOpenAI | StrOutputParser`.
- The prompt tells the LLM to use only the given movies and to give one short reason for each pick, so it does not invent movies.
- The UI shows movie cards plus the AI's explanation.

### 4. Click-to-explore UI (`app2.py`)

- Uses the saved TF-IDF similarity matrix (no API needed for the model).
- Shows popular movies, or search by title.
- Click **Related dekho** under any movie to see its 10 most similar movies. Click again to keep exploring, or go back.
- Posters are fetched from the OMDb API. If no key is set or a poster is missing, a placeholder is shown.

---

## Project Files

| File | Purpose |
|---|---|
| `clean_data.py` | Cleans the raw dataset and saves `cleaned_tmdb.csv` |
| `train_model.py` | Fits TF-IDF, computes similarity, saves model files |
| `build_index.py` | Builds embeddings and the Chroma index |
| `app.py` | AI recommender (RAG + LLM) |
| `app2.py` | Click-to-explore UI with posters |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps secrets and large files out of Git |
| `README.md` | This file |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. Python 3.9 works for most parts but is no longer supported by several libraries.

### 2. Download the dataset

Download `tmdb_5000_movies.csv` from Kaggle and place it in the project folder.

### 3. Add API keys

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_openrouter_key
OMDB_API_KEY=your_omdb_key
```

- OpenRouter key: used by `app.py` for the LLM.
- OMDb key: free (1,000 requests per day), used by `app2.py` for posters. Activate it using the link sent to your email.
- Never commit `.env`. It is already listed in `.gitignore`.

---

## Run

Run these in order the first time:

```bash
python clean_data.py        # creates cleaned_tmdb.csv
python train_model.py       # creates tfidf.pkl, similarity.pkl, movies_df.pkl
python build_index.py       # creates the chroma_db folder
```

Then start an app:

```bash
streamlit run app.py        # AI recommender
streamlit run app2.py       # click-to-explore UI
```

Always use `streamlit run`, not `python app.py`.

---

## Example Queries (for `app.py`)

- `space survival movie`
- `thriller with a twist`
- `heist action movie`
- `family animated comedy`
- `movies about dinosaurs`

English queries work best because the dataset is in English.

---

## TF-IDF vs Embeddings

| | TF-IDF (baseline) | Embeddings (AI recommender) |
|---|---|---|
| Matches by | Exact words | Meaning |
| "car" vs "automobile" | Treated as different | Treated as similar |
| Cost | Free | Free embeddings, small LLM cost |
| Best for | "Movies similar to X" | Free-text requests |

---

## Troubleshooting

- **"Koi movie nahi mili" / empty results:** the Chroma index is empty or was not built. Delete `chroma_db/` and run `build_index.py` again. It should end with the number of movies saved.
- **Dimension mismatch error:** the embedding model used in `build_index.py` and `app.py` must be exactly the same. If you change the model, delete `chroma_db/` and rebuild.
- **429 quota error with Gemini embeddings:** the free tier allows about 100 embedding requests per minute. This project uses local HuggingFace embeddings to avoid that.
- **No posters:** check that `OMDB_API_KEY` is set and the key is activated. Some titles may not have a poster.
- **Large `similarity.pkl` (about 180 MB):** it is ignored by Git. Regenerate it with `train_model.py`.

---

## Limitations

- TF-IDF does not understand meaning, only word overlap.
- The dataset is limited to about 4,800 movies and is not updated with new releases.
- The LLM only sees the retrieved movies, so a poor retrieval gives poor recommendations.
- There is no user rating data, so this is content-based only.

---

## Future Improvements

- Collaborative filtering with user ratings (MovieLens)
- Hybrid model combining TF-IDF and embeddings, with a proper evaluation
- Watchlist and favourites
- Chat memory, so the bot remembers earlier requests
- Hindi / Hinglish query support
- Deployment on Streamlit Community Cloud or Hugging Face Spaces

---

## Author

**Rishabh Yadav**
GitHub: [sumn22062-netizen](https://github.com/sumn22062-netizen)
