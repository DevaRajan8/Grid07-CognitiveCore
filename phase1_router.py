import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

load_dotenv()

class SentenceTransformerEmbeddings(Embeddings):
    """Custom embeddings using sentence-transformers"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False).tolist()

BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. I am highly optimistic "
            "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}


def build_persona_vectorstore():
    embeddings = SentenceTransformerEmbeddings()
    docs = []
    for bot_id, persona in BOT_PERSONAS.items():
        doc = Document(
            page_content=persona["description"],
            metadata={"bot_id": bot_id, "name": persona["name"]},
        )
        docs.append(doc)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, embeddings


def route_post_to_bots(post_content: str, threshold: float = 0.30):
    vectorstore, embeddings = build_persona_vectorstore()

    results = vectorstore.similarity_search_with_score(post_content, k=len(BOT_PERSONAS))

    matched_bots = []
    print(f"\n[PHASE 1] Routing post: '{post_content}'")
    print(f"[PHASE 1] Using similarity threshold: {threshold}")
    print("-" * 60)

    for doc, score in results:
        # FAISS returns L2 distance; convert to cosine-like similarity
        # Lower score = more similar for L2. We normalize to 0-1 range.
        similarity = 1 / (1 + score)
        bot_id = doc.metadata["bot_id"]
        bot_name = doc.metadata["name"]
        print(f"  Bot: {bot_name} ({bot_id}) | Similarity Score: {similarity:.4f}")
        if similarity >= threshold:
            matched_bots.append({
                "bot_id": bot_id,
                "name": bot_name,
                "similarity": similarity,
            })

    print("-" * 60)
    if matched_bots:
        print(f"[PHASE 1] Matched bots: {[b['bot_id'] for b in matched_bots]}")
    else:
        print("[PHASE 1] No bots matched above threshold.")

    return matched_bots


if __name__ == "__main__":
    test_post = "OpenAI just released a new model that might replace junior developers."
    matched = route_post_to_bots(test_post, threshold=0.30)
    print(f"\n[RESULT] Bots that care about this post: {[b['bot_id'] for b in matched]}")

    print("\n" + "=" * 60)
    test_post_2 = "Bitcoin hits new highs as the Fed signals rate cuts next quarter."
    matched_2 = route_post_to_bots(test_post_2, threshold=0.30)
    print(f"\n[RESULT] Bots that care about this post: {[b['bot_id'] for b in matched_2]}")
