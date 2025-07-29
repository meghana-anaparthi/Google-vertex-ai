#!/usr/bin/env python3
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. Initialize the GenAI client for Vertex AI ─────────────────────────────
# (Assumes you’ve run `gcloud auth application-default login` or set
# GOOGLE_APPLICATION_CREDENTIALS in your env.)
client = genai.Client(
    vertexai=True,
    project="vertex-ai-learn-447422",
    location="us-central1",
)

# ── 2. Hard‑coded test pair ───────────────────────────────────────────────────
golden  = "Elephants have two tusks that they use for feeding and defense."
chatbot = "Elephants, magnificent giants of the savanna, each possess a pair of ivory tusks which they expertly wield to uproot vegetation for food, dig waterholes in dry seasons, and fend off predators."

# ── 3. Get embeddings for both texts ───────────────────────────────────────────
embed_resp = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[golden, chatbot]
)
# unpack embeddings (some versions return .values, others a list)
embs = embed_resp.embeddings
def _unpack(e):
    return e.values if hasattr(e, "values") else e
gold_emb, bot_emb = _unpack(embs[0]), _unpack(embs[1])

# ── 4. Compute cosine similarity → 0–100% ──────────────────────────────────────
emb_score = round(cosine_similarity([gold_emb], [bot_emb])[0][0] * 100, 2)
print(f"Embedding similarity score: {emb_score}%")

# ── 5. Build an evaluator prompt for coverage scoring ─────────────────────────
eval_prompt = (
    f"You are an evaluator.\n"
    f"Golden answer: \"{golden}\"\n"
    f"Chatbot answer: \"{chatbot}\"\n"
    f"On a scale from 0 to 100, how much of the golden content is covered?\n"
    f"Just reply with the number."
)

# ── 6. Ask Gemini for the coverage score ──────────────────────────────────────
llm_resp = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=eval_prompt
)
raw = llm_resp.text.strip()
try:
    cov_score = float(raw)
    print(f"LLM coverage score:   {cov_score}%")
except ValueError:
    print("Unexpected LLM output:", raw)
