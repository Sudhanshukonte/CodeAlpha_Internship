# CodeAlpha FAQ Chatbot

A retrieval-based FAQ chatbot using NLP preprocessing, TF-IDF, cosine similarity, and Streamlit.

## Features
- Interactive chat interface
- FAQ knowledge base stored in JSON
- TF-IDF text representation and cosine-similarity matching
- Confidence threshold and graceful fallback response
- Similarity score and matched FAQ shown for transparency

## Requirements
- Python 3.10 or newer
- Internet access for the first-time package installation

## Run locally

### Windows (PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Linux / macOS / Ubuntu (WSL)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## How it works
1. Load FAQ question/answer pairs from `data/faqs.json`.
2. Fit a TF-IDF vectorizer on the FAQ questions.
3. Convert the user's question to a vector.
4. Rank FAQ questions using cosine similarity.
5. Return the answer for the strongest match if it meets the confidence threshold.

## Customize
Edit `data/faqs.json` to add questions and answers. Keep each entry in this format:
```json
{
  "question": "Your question?",
  "answer": "Your answer."
}
```

The threshold in `app.py` is a simple baseline and should be tuned using test questions. TF-IDF matching is not semantic understanding; paraphrases may be missed. For production, add tests, logging, better evaluation, and potentially embeddings.

## Suggested GitHub repository
`CodeAlpha_FAQ_Chatbot`

## Demo checklist
- Ask a direct question, such as “What is Python?”
- Ask a paraphrase, such as “Can you explain TF IDF?”
- Ask an unrelated question to demonstrate the fallback
- Add one FAQ and verify it can be retrieved
