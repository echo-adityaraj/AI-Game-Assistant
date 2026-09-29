# AI Game Assistant

Steam game discovery assistant built with Python, Pandas, scikit-learn TF-IDF retrieval, optional LangChain + Groq, and a Streamlit chat UI.

The assistant works without an API key. With a Groq key, it can turn retrieved games into conversational recommendations.

## Features

- Steam game search and recommendation over 20,000 cleaned game records
- TF-IDF retrieval over titles, genres, tags, categories, descriptions, developers, publishers, and platforms
- Natural-language price and platform filters
- Optional LangChain + Groq conversational answers
- Retrieval-only fallback when no API key is supplied
- Streamlit chat interface
- Automatically builds the TF-IDF index when deployed

## Example Usage

After starting the Streamlit app, enter a game-related request in the chat box.

Example queries:

```text
Recommend me a co-op horror game under £20 for Mac
```

```text
Recommend me 5 adventure games for Windows
```

```text
I want a relaxing strategy game for Mac
```

```text
Suggest highly rated RPG games
```

### Sidebar controls

- **Number of games** — choose how many recommendations to return
- **Platform** — filter by Windows, Mac, or Linux
- **Set maximum price** — limit recommendations to your budget
- **Minimum review count** — exclude games with very few reviews
- **Groq API key** — optional; enables conversational LLM responses

## Project Layout

```text
AI-Game-Assistant/
├── AI_Game_Assistant.ipynb
├── app.py
├── Dataset.zip
├── requirements.txt
├── README.md
└── .gitignore
```

`Dataset.zip` contains the cleaned Steam dataset used by the deployed application. The Streamlit app reads the CSV from the ZIP and builds the TF-IDF search index automatically, so pre-generated artifact files are not required for deployment.

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the requirements:

```bash
python3 -m pip install -r requirements.txt
```

## Run the Streamlit App

```bash
python3 -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.


## Data

Primary source: `FronkonGames/steam-games-dataset` on Hugging Face.

The repository contains a cleaned 20,000-game sample packaged as `Dataset.zip` for the application.

