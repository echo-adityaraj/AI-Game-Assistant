from pathlib import Path
import os
import re
import zipfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "steam_games_clean.csv"
DATA_ZIP = ROOT / "Dataset.zip"
ARTIFACT_DIR = ROOT / "artifacts"
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = ARTIFACT_DIR / "tfidf_matrix.joblib"

st.set_page_config(page_title="AI Game Assistant", page_icon="🎮", layout="wide")


def _read_dataset() -> pd.DataFrame:
    """Load the cleaned Steam dataset from CSV or the repository ZIP."""
    if DATA_CSV.exists():
        return pd.read_csv(DATA_CSV)

    if DATA_ZIP.exists():
        with zipfile.ZipFile(DATA_ZIP) as archive:
            csv_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv") and "__MACOSX" not in name
            ]
            if not csv_files:
                raise FileNotFoundError("Dataset.zip does not contain a CSV file.")
            with archive.open(csv_files[0]) as handle:
                return pd.read_csv(handle)

    raise FileNotFoundError(
        "No dataset found. Add Dataset.zip or data/steam_games_clean.csv to the repository."
    )


def _normalise_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_columns = [
        "name", "genres", "categories", "tags", "short_description",
        "detailed_description", "developers", "publishers", "platforms",
        "header_image",
    ]
    for col in text_columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    for col in ["price", "positive", "negative", "positive_ratio", "total_reviews", "metacritic_score"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "total_reviews" not in df.columns or (df["total_reviews"] == 0).all():
        df["total_reviews"] = df["positive"] + df["negative"]

    if "positive_ratio" not in df.columns or (df["positive_ratio"] == 0).all():
        denominator = df["total_reviews"].replace(0, 1)
        df["positive_ratio"] = df["positive"] / denominator

    platform_names = []
    for col, label in [("windows", "Windows"), ("mac", "Mac"), ("linux", "Linux")]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
        platform_names.append((col, label))

    if (df["platforms"].str.strip() == "").all():
        df["platforms"] = df.apply(
            lambda row: ", ".join(label for col, label in platform_names if bool(row[col])),
            axis=1,
        )

    df = df[df["name"].str.strip() != ""].reset_index(drop=True)
    return df


def _build_search_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["name"] + " "
        + "genres " + df["genres"] + " "
        + "categories " + df["categories"] + " "
        + "tags " + df["tags"] + " "
        + "description " + df["short_description"] + " "
        + "details " + df["detailed_description"] + " "
        + "developer " + df["developers"] + " "
        + "publisher " + df["publishers"] + " "
        + "platforms " + df["platforms"]
    )


@st.cache_resource(show_spinner="Loading Steam games and building search index...")
def load_project():
    df = _normalise_dataset(_read_dataset())

    # Use pre-built artifacts when available; otherwise build them automatically.
    if VECTORIZER_PATH.exists() and MATRIX_PATH.exists():
        try:
            import joblib
            vectorizer = joblib.load(VECTORIZER_PATH)
            matrix = joblib.load(MATRIX_PATH)
            if matrix.shape[0] == len(df):
                return df, vectorizer, matrix
        except Exception:
            pass

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=25000,
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(_build_search_text(df))
    return df, vectorizer, matrix


try:
    df, vectorizer, tfidf_matrix = load_project()
except Exception as exc:
    st.error(f"Unable to load the project: {exc}")
    st.stop()


PRICE_PATTERN = re.compile(
    r"(?:under|below|less than|max(?:imum)?|up to)\s*[£$€]?\s*(\d+(?:\.\d+)?)",
    re.I,
)


def infer_query_filters(query: str):
    q = query.lower()
    match = PRICE_PATTERN.search(query)
    max_price = float(match.group(1)) if match else (0.0 if "free" in q else None)

    platform = None
    if re.search(r"\bmac(?:os)?\b", q):
        platform = "mac"
    elif re.search(r"\blinux\b", q):
        platform = "linux"
    elif re.search(r"\bwindows\b|\bpc\b", q):
        platform = "windows"

    return max_price, platform


def search_games(query, top_k=5, max_price=None, platform="Any", min_reviews=0):
    if not isinstance(query, str) or not query.strip():
        return pd.DataFrame()

    inferred_price, inferred_platform = infer_query_filters(query)
    if max_price is None:
        max_price = inferred_price

    selected_platform = str(platform).lower()
    if selected_platform == "any":
        selected_platform = inferred_platform

    scores = cosine_similarity(vectorizer.transform([query]), tfidf_matrix).ravel()
    mask = np.ones(len(df), dtype=bool)

    if max_price is not None:
        mask &= df["price"].to_numpy(dtype=float) <= float(max_price)
    if selected_platform in {"windows", "mac", "linux"}:
        mask &= df[selected_platform].to_numpy(dtype=bool)
    if min_reviews:
        mask &= df["total_reviews"].to_numpy(dtype=float) >= int(min_reviews)

    valid = np.flatnonzero(mask)
    if not len(valid):
        return pd.DataFrame()

    ranked = valid[np.argsort(scores[valid])[::-1]][:top_k]
    result = df.iloc[ranked].copy()
    result.insert(0, "match_score", scores[ranked])
    return result.reset_index(drop=True)


def format_price(value):
    value = float(value)
    return "Free" if value <= 0 else f"£{value:.2f}"


def fallback_answer(question, results):
    if results.empty:
        return "I could not find a game matching those filters. Try widening your requirements."

    lines = ["Here are the closest matches I found:"]
    for i, row in results.iterrows():
        ratio = float(row.get("positive_ratio", 0))
        total = float(row.get("total_reviews", 0))
        reviews = f"{ratio * 100:.0f}% positive" if total > 0 else "reviews unavailable"
        lines.append(
            f'{i + 1}. **{row["name"]}** — {row.get("genres", "")} | '
            f'{format_price(row.get("price", 0))} | {row.get("platforms", "")} | '
            f'{reviews}. {row.get("short_description", "")}'
        )
    return "\n\n".join(lines)


def resolve_groq_model(api_key, preferred=""):
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    response.raise_for_status()
    ids = [item.get("id", "") for item in response.json().get("data", [])]

    if preferred and preferred in ids:
        return preferred
    for keyword in ["llama", "qwen", "gemma", "mixtral"]:
        found = next((model_id for model_id in ids if keyword in model_id.lower()), None)
        if found:
            return found
    if not ids:
        raise RuntimeError("No Groq models are currently available.")
    return ids[0]


def build_context(results):
    return "\n\n---\n\n".join(
        f'Game: {row["name"]}\n'
        f'Genres: {row.get("genres", "")}\n'
        f'Price: {format_price(row.get("price", 0))}\n'
        f'Platforms: {row.get("platforms", "")}\n'
        f'Positive ratio: {float(row.get("positive_ratio", 0)):.3f}\n'
        f'Reviews: {int(float(row.get("total_reviews", 0)))}\n'
        f'Description: {row.get("short_description", "")}'
        for _, row in results.iterrows()
    )


def answer_with_langchain(question, results, api_key, preferred_model=""):
    if not api_key:
        return fallback_answer(question, results), "Retrieval only"

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda

        model = resolve_groq_model(api_key, preferred_model)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI Game Assistant. Use only the retrieved context. "
                    "Do not invent game facts.",
                ),
                ("human", "Request:\n{question}\n\nRetrieved context:\n{context}"),
            ]
        )

        def call_groq(prompt_value):
            messages = []
            for message in prompt_value.to_messages():
                role = "assistant" if message.type == "ai" else (
                    "system" if message.type == "system" else "user"
                )
                messages.append({"role": role, "content": str(message.content)})

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages, "temperature": 0.2},
                timeout=45,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        answer = (prompt | RunnableLambda(call_groq)).invoke(
            {"question": question, "context": build_context(results)}
        )
        return answer, f"LangChain + Groq ({model})"
    except Exception as exc:
        return (
            fallback_answer(question, results)
            + f"\n\nLLM unavailable: {type(exc).__name__}",
            "Retrieval fallback",
        )


st.title("AI Game Assistant")
st.caption(
    "Steam game discovery with TF-IDF retrieval, natural-language filters, "
    "LangChain and an optional Groq LLM."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Groq API key (optional)",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
    )
    preferred_model = st.text_input(
        "Preferred Groq model (optional)",
        value=os.getenv("GROQ_MODEL", ""),
    )
    top_k = st.slider("Number of games", 3, 10, 5)
    platform = st.selectbox("Platform", ["Any", "Windows", "Mac", "Linux"])
    use_price = st.checkbox("Set maximum price")
    max_price = st.slider("Maximum price (£)", 0, 100, 30) if use_price else None
    min_reviews = st.number_input(
        "Minimum review count", min_value=0, value=0, step=100
    )
    st.caption(f"Dataset contains {len(df):,} games.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input('Ask for a game, e.g. "co-op horror under £20 for Windows"')
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    results = search_games(query, top_k, max_price, platform, min_reviews)
    answer, mode = answer_with_langchain(
        query, results, api_key, preferred_model
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"Mode: {mode}")

        if not results.empty:
            st.subheader("Retrieved games")
            for _, row in results.iterrows():
                image_col, text_col = st.columns([1, 3])
                image = str(row.get("header_image", "")).strip()

                with image_col:
                    if image.startswith("http"):
                        st.image(image, use_container_width=True)

                with text_col:
                    st.markdown(f'### {row["name"]}')
                    st.write(row.get("short_description", ""))
                    st.write(
                        f'**Genres:** {row.get("genres", "Not listed")}  |  '
                        f'**Price:** {format_price(row.get("price", 0))}  |  '
                        f'**Platforms:** {row.get("platforms", "Not listed")}'
                    )
                    if float(row.get("total_reviews", 0)) > 0:
                        st.write(
                            f'**Positive reviews:** '
                            f'{float(row.get("positive_ratio", 0)) * 100:.1f}%'
                        )
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
