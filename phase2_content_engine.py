import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

MOCK_NEWS_DB = {
    "crypto": [
        "Bitcoin hits new all-time high amid regulatory ETF approvals.",
        "Ethereum upgrade promises 10x throughput improvements.",
        "SEC approves spot Bitcoin ETF, institutional money floods in.",
    ],
    "ai": [
        "OpenAI releases GPT-5 with autonomous agent capabilities.",
        "Google DeepMind achieves human-level reasoning on new benchmarks.",
        "AI replaces 200,000 white-collar jobs in Q2 report.",
    ],
    "tech": [
        "Elon Musk's xAI raises $10B in latest funding round.",
        "Apple Vision Pro 2 sold out globally within minutes of launch.",
        "SpaceX Starship completes first crewed Mars mission simulation.",
    ],
    "capitalism": [
        "Amazon workers strike across 50 warehouses demanding fair wages.",
        "Big Tech monopoly investigation expands to 12 countries.",
        "Social media addiction lawsuits target Meta and TikTok.",
    ],
    "market": [
        "S&P 500 surges 3% on strong earnings season.",
        "Fed signals two rate cuts in H2, bond yields drop sharply.",
        "Hedge funds post 18% returns Q2 via AI-driven trading algorithms.",
    ],
    "finance": [
        "Goldman Sachs deploys AI trading bot managing $50B AUM.",
        "Interest rate swap volumes hit record highs on volatility.",
        "Private equity dry powder reaches $3.7T as deals stall.",
    ],
    "environment": [
        "Climate scientists warn of irreversible tipping points by 2030.",
        "EU bans single-use plastics; industry lobbying intensifies.",
        "Data centers now consume 3% of global electricity.",
    ],
}

BOT_PERSONAS = {
    "bot_a": (
        "You are Bot A, a Tech Maximalist. You believe AI and crypto will solve all human "
        "problems. You are highly optimistic about technology, Elon Musk, and space exploration. "
        "You dismiss regulatory concerns. You write short, punchy, opinionated takes. "
        "You use emojis and tech buzzwords."
    ),
    "bot_b": (
        "You are Bot B, a Doomer/Skeptic. You believe late-stage capitalism and tech monopolies "
        "are destroying society. You are highly critical of AI, social media, and billionaires. "
        "You value privacy and nature. You write cynical, sharp, anti-establishment posts."
    ),
    "bot_c": (
        "You are Bot C, a Finance Bro. You strictly care about markets, interest rates, trading "
        "algorithms, and making money. You speak in finance jargon and view everything through "
        "the lens of ROI. You use terms like alpha, yield, basis points, and liquidity."
    ),
}


@tool
def mock_searxng_search(query: str) -> str:
    """Search for news articles matching the given query."""
    query_lower = query.lower()
    results = []
    for keyword, headlines in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            results.extend(headlines)
    if not results:
        results = ["No specific news found. General tech trends dominate headlines today."]
    return " | ".join(results[:3])


class PostState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_results: str
    final_post: str


def decide_search_node(state: PostState) -> PostState:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    prompt = (
        f"You are this persona: {state['persona']}\n\n"
        "Based on your personality, decide ONE topic you want to post about today. "
        "Respond with ONLY a short 3-6 word search query, nothing else."
    )
    response = llm.invoke(prompt)
    search_query = response.content.strip().strip('"').strip("'")
    print(f"[PHASE 2] Bot {state['bot_id']} decided search query: '{search_query}'")
    return {**state, "search_query": search_query}


def web_search_node(state: PostState) -> PostState:
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"[PHASE 2] Search results: {results}")
    return {**state, "search_results": results}


def draft_post_node(state: PostState) -> PostState:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9)

    system_prompt = (
        f"{state['persona']}\n\n"
        "CRITICAL: You must ONLY output a valid JSON object. No extra text, no markdown, no explanation. "
        "The JSON must have exactly these keys: bot_id, topic, post_content. "
        "post_content must be under 280 characters and highly opinionated in your persona's voice."
    )

    user_prompt = (
        f"Recent news context: {state['search_results']}\n\n"
        f"Write a post about the topic: {state['search_query']}\n"
        f"Output JSON only: {{\"bot_id\": \"{state['bot_id']}\", \"topic\": \"...\", \"post_content\": \"...\"}}"
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    post_json = json.loads(raw)
    print(f"[PHASE 2] Generated post JSON: {json.dumps(post_json, indent=2)}")
    return {**state, "final_post": json.dumps(post_json)}


def build_content_graph():
    graph = StateGraph(PostState)
    graph.add_node("decide_search", decide_search_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("draft_post", draft_post_node)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()


def run_content_engine(bot_id: str) -> dict:
    if bot_id not in BOT_PERSONAS:
        raise ValueError(f"Unknown bot_id: {bot_id}")

    app = build_content_graph()
    initial_state = PostState(
        bot_id=bot_id,
        persona=BOT_PERSONAS[bot_id],
        search_query="",
        search_results="",
        final_post="",
    )

    print(f"\n[PHASE 2] Running content engine for {bot_id}...")
    print("=" * 60)
    result = app.invoke(initial_state)
    return json.loads(result["final_post"])


if __name__ == "__main__":
    for bot_id in ["bot_a", "bot_b", "bot_c"]:
        output = run_content_engine(bot_id)
        print(f"\n[FINAL OUTPUT - {bot_id}]")
        print(json.dumps(output, indent=2))
        print("=" * 60)
