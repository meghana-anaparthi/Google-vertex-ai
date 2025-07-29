#!/usr/bin/env python3
import os
from google import genai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. Init GenAI client ───────────────────────────────────────────────────────
# (Assumes you’ve done `gcloud auth application-default login`
#  or have GOOGLE_APPLICATION_CREDENTIALS set)
client = genai.Client(
    vertexai=True,
    project="YOUR_PROJECT_ID",
    location="us-central1",
)

# ── 2. Read your CSV ───────────────────────────────────────────────────────────
# Replace with your filename
CSV_IN  = "qa_pairs.csv"
CSV_OUT = "qa_pairs_with_scores.csv"
df = pd.read_csv(CSV_IN)

# ── 3. Prepare lists to hold scores ────────────────────────────────────────────
emb_scores = []
cov_scores = []

# ── 4. Loop through each row ──────────────────────────────────────────────────
for idx, row in df.iterrows():
    golden = row["Golden"]
    llm_ans = row["LLM_Response"]  # or whatever your LLM column is called

    # 4a. Embedding call (batch of 2)
    embed_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[golden, llm_ans]
    )
    embs = embed_resp.embeddings
    def _unpack(e):
        return e.values if hasattr(e, "values") else e
    gold_emb, bot_emb = _unpack(embs[0]), _unpack(embs[1])

    # 4b. Cosine sim → 0–100
    emb_score = round(cosine_similarity([gold_emb], [bot_emb])[0][0] * 100, 2)
    emb_scores.append(emb_score)

    # 4c. LLM coverage prompt
    prompt = (
        f"You are an evaluator.\n"
        f"Golden answer: \"{golden}\"\n"
        f"Chatbot answer: \"{llm_ans}\"\n"
        f"On a scale from 0 to 100, how much of the golden content is covered?\n"
        f"Just reply with the number."
    )
    llm_resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    ).text.strip()

    try:
        cov_scores.append(float(llm_resp))
    except ValueError:
        cov_scores.append(None)

# ── 5. Attach scores and save ──────────────────────────────────────────────────
df["Embedding_Score"] = emb_scores
df["Coverage_Score"]  = cov_scores
df.to_csv(CSV_OUT, index=False)

print(f"✅ Scored {len(df)} rows – results in {CSV_OUT}")
