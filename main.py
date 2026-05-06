"""
main.py — Yojana Mitra  (Production MCP Architecture)
=======================================================

Fixes applied:
  1. DB_THRESHOLD raised to 20
  2. web_search made MANDATORY when scheme not in DB
  3. Conversation history passed as real messages (not just text in prompt)
  4. Stronger system prompt rules to prevent "I couldn't find" fallback
"""

import os
import re
import json
import time
import asyncio
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

from mcp_client import MCPClient

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "schemes"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama-3.1-8b-instant"
TOP_K           = 5
DB_THRESHOLD    = 20          # ← RAISED from 15 to catch more valid schemes

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Yojana Mitra API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────
print("[Startup] Loading embedding model …")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("[Startup] Embedding model ready.")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection     = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
print(f"[Startup] ChromaDB ready. Schemes in DB: {collection.count()}")

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("[Startup] Groq client initialised.")
else:
    groq_client = None
    print("[Startup] WARNING: GROQ_API_KEY not set!")


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB helpers
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_from_db(query: str) -> list[tuple]:
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    return list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ))


def build_db_context(hits: list[tuple], limit: int = 3) -> str:
    parts = []
    for i, (doc, meta, dist) in enumerate(hits[:limit], 1):
        parts.append(
            f"--- Scheme {i}: {meta['name']} ---\n"
            f"Official URL  : {meta.get('source_url', 'N/A')}\n"
            f"Distance score: {dist:.2f}\n"
            f"{doc}\n"
        )
    return "\n".join(parts) if parts else "No matching scheme found in local database."


# ══════════════════════════════════════════════════════════════════════════════
# Session profile extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_profile(history: list[dict]) -> dict:
    user_text = " ".join(
        h["content"] for h in history if h.get("role") == "user"
    ).lower()

    profile: dict = {}

    age_match = re.search(r'\b(\d{1,2})\s*(?:years?\s*old|yrs?\.?|y\.o\.?)\b', user_text)
    if age_match:
        age = int(age_match.group(1))
        if 10 <= age <= 100:
            profile["age"] = age

    income_patterns = [
        (r'(\d+(?:\.\d+)?)\s*lakh(?:s)?\s*(?:per\s*year|annually|pa|rupees?)?', "lakh"),
        (r'(?:income|salary|earn(?:ing)?|makes?)[^\d]{0,10}(\d[\d,]+)', "direct"),
        (r'(\d[\d,]+)\s*(?:per\s*year|annually|pa|\/year)', "direct"),
    ]
    for pattern, kind in income_patterns:
        m = re.search(pattern, user_text)
        if m:
            raw = m.group(1).replace(",", "")
            income = float(raw)
            if kind == "lakh":
                income *= 100_000
            income = int(income)
            if 1_000 <= income <= 10_000_000:
                profile["annual_income"] = f"₹{income:,}"
                break

    if re.search(r'\b(?:female|woman|girl|she|her)\b', user_text):
        profile["gender"] = "female"
    elif re.search(r'\b(?:male|man|boy|he|him)\b', user_text):
        profile["gender"] = "male"

    state_map = {
        "karnataka": "Karnataka",       "maharashtra": "Maharashtra",
        "tamil nadu": "Tamil Nadu",     "kerala": "Kerala",
        "andhra pradesh": "Andhra Pradesh", "andhra": "Andhra Pradesh",
        "telangana": "Telangana",       "rajasthan": "Rajasthan",
        "gujarat": "Gujarat",           "punjab": "Punjab",
        "bihar": "Bihar",               "uttar pradesh": "Uttar Pradesh",
        "west bengal": "West Bengal",   "delhi": "Delhi",
        "goa": "Goa",                   "odisha": "Odisha",
        "madhya pradesh": "Madhya Pradesh", "haryana": "Haryana",
        "assam": "Assam",               "jharkhand": "Jharkhand",
        "chhattisgarh": "Chhattisgarh", "uttarakhand": "Uttarakhand",
        "himachal pradesh": "Himachal Pradesh",
    }
    for key, value in state_map.items():
        if key in user_text:
            profile["state"] = value
            break

    occupations = [
        "farmer", "student", "daily wage worker", "labourer", "laborer",
        "factory worker", "government employee", "private employee",
        "teacher", "shopkeeper", "housewife", "artisan", "weaver",
        "unemployed", "self employed", "self-employed", "freelancer",
        "gig worker", "domestic worker", "street vendor", "fisherman",
        "shepherd", "nurse", "doctor",
    ]
    for occ in occupations:
        if occ in user_text:
            profile["occupation"] = occ
            break

    categories = ["sc", "st", "obc", "general", "minority", "ews"]
    for cat in categories:
        if re.search(rf'\b{cat}\b', user_text):
            profile["category"] = cat.upper()
            break

    return profile


def profile_to_text(profile: dict) -> str:
    if not profile:
        return "Not provided yet in this session."
    return " | ".join(f"{k}: {v}" for k, v in profile.items())


# ══════════════════════════════════════════════════════════════════════════════
# System prompt builder
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(
    profile_text: str,
    history_text: str,
    db_context: str,
    scheme_found: bool,
    source_url: str,
) -> str:

    # ── Tool hint — strong and explicit ──────────────────────────────────
    if scheme_found and source_url:
        tool_hint = (
            f"- Scheme IS in local DB. Use the context above to answer. "
            f"If user wants MORE details or latest updates, call fetch_url with: {source_url}"
        )
    elif scheme_found:
        tool_hint = (
            "- Scheme IS in local DB. Use the context above to answer."
        )
    else:
        tool_hint = (
            "- Scheme NOT in local DB. "
            "YOU MUST call web_search IMMEDIATELY before doing anything else. "
            "Do NOT say 'I couldn't find information'. "
            "Do NOT redirect to myscheme.gov.in or 14567 as your first response. "
            "SEARCHING IS MANDATORY. Call web_search now."
        )

    return f"""You are Yojana Mitra, a warm, knowledgeable, and helpful AI assistant \
specialising exclusively in Indian government welfare schemes.

You have two tools you can call whenever needed:
  • web_search(query)  — search the web for scheme info; always prefer .gov.in results
  • fetch_url(url)     — fetch full text from any government website URL

The LLM (you) decides autonomously when to use these tools.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER PROFILE  (THIS SESSION ONLY — resets when user starts a new chat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{profile_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECENT CONVERSATION HISTORY (last 8 turns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{history_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCAL DATABASE RESULTS FOR THIS QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{db_context}

Tool hint: {tool_hint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE RULES — follow exactly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. OUT OF SCOPE
   If the user asks about weather, sports, movies, recipes, or ANYTHING unrelated
   to Indian government schemes, reply:
   "I'm Yojana Mitra — I'm here to help you with Indian government welfare schemes only!
   Feel free to ask about housing, farming, health, education, or employment schemes. 😊"
   Do NOT use any tools for out-of-scope queries.

2. GREETING
   If the user says hi / hello / namaste / hey, respond warmly as Yojana Mitra and
   ask how you can help. Do NOT re-greet in ongoing conversations.

3. SCHEME DETAILS  ("what is X", "tell me about X", "explain X scheme")
   Always use this EXACT structure:

   **[Scheme Name]**

   **What is it?**
   [2–3 sentence overview]

   **Eligibility Criteria**
   - [criterion 1]
   - [criterion 2]

   **Benefits**
   - [benefit 1]

   **Documents Required**
   - [document 1]

   **How to Apply**
   - [step 1]

   **Official Website:** [url]

4. SCHEME NOT IN LOCAL DB — MANDATORY WEB SEARCH
   If DB context says "No matching scheme found":
   → You MUST call web_search FIRST. No exceptions.
   → NEVER say "I couldn't find reliable information" without calling web_search first.
   → NEVER redirect to myscheme.gov.in or helpline 14567 as your FIRST response.
   → After getting search results, answer using the structured format in Rule 3.
   → Only if web_search also returns nothing useful, THEN redirect to myscheme.gov.in.

5. MORE DETAILS / FOLLOW-UP ON A SCHEME
   When user asks for more info on a scheme just discussed
   (e.g. "tell me more", "step by step", "how to apply", "what documents"):
   - Use conversation history to identify which scheme they mean
   - If a source URL is available → call fetch_url to get live updated content
   - Answer based on DB context + fetched content
   - NEVER say "I couldn't find" for a scheme already discussed in this conversation

6. "WHAT SCHEMES AM I ELIGIBLE FOR?"
   - If profile incomplete → ask: "Could you share your age, annual income, state,
     and occupation? This helps me find the most relevant schemes for you."
   - If profile available → list 4–8 relevant schemes:
     • **[Scheme Name]** — [one-line benefit]
     End with: "Want full details or eligibility info for any of these?"

7. ELIGIBILITY CHECK  ("am I eligible for X?")
   - If profile incomplete → ask for missing fields
   - If profile available → compare against criteria and respond:
     "Based on your profile, you [are likely / may not be] eligible because [reasons].
      Please verify at the official website to confirm."

8. CONVERSATION CONTINUITY
   Always use conversation history to maintain context.
   Never ask for information already given in this session.
   Connect answers naturally to the thread of conversation.

9. GENERAL RULES
   - Never invent scheme details. Only use DB data or tool results.
   - Use bullet points for scheme details; never long paragraphs.
   - Be warm, concise, and encouraging.
   - End with a relevant follow-up question when appropriate.
   - Currency: always use ₹ symbol.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Core LLM orchestration with MCP tool loop
# ══════════════════════════════════════════════════════════════════════════════

async def run_llm_with_tools(
    messages: list[dict],
    mcp_client: MCPClient,
    max_tool_rounds: int = 3,
) -> tuple[str, list[str]]:

    tools_used: list[str] = []
    current_messages = list(messages)

    for round_num in range(max_tool_rounds + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=current_messages,
                max_tokens=1200,
                temperature=0.3,
                tools=mcp_client.groq_tools,   # ← MCP tools injected here
                tool_choice="auto",             # ← LLM decides when to use them
            )
        except Exception as exc:
            print(f"[LLM] Groq API error (round {round_num}): {exc}")
            if round_num == 0:
                time.sleep(2)
                continue
            return "", tools_used

        choice = response.choices[0]
        msg    = choice.message

        # LLM wants to call tools
        if msg.tool_calls:
            print(f"[LLM] Round {round_num}: {len(msg.tool_calls)} tool call(s) requested")

            current_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            async def execute_tool(tool_call):
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                tools_used.append(fn_name)
                result_text = await mcp_client.call_tool(fn_name, fn_args)
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": result_text,
                }

            tool_results = await asyncio.gather(
                *[execute_tool(tc) for tc in msg.tool_calls]
            )
            current_messages.extend(tool_results)

        else:
            # LLM gave direct answer
            answer = (msg.content or "").strip()
            print(f"[LLM] Answer after {round_num} tool round(s).")
            return answer, tools_used

    # Max rounds reached — ask for final answer without tools
    print("[LLM] Max rounds reached — requesting final answer.")
    try:
        fallback = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=current_messages,
            max_tokens=1200,
            temperature=0.3,
        )
        return fallback.choices[0].message.content.strip(), tools_used
    except Exception as exc:
        print(f"[LLM] Fallback failed: {exc}")
        return "", tools_used


# ══════════════════════════════════════════════════════════════════════════════
# /chat endpoint
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq not configured. Set GROQ_API_KEY in .env")

    print(f"\n{'═' * 60}")
    print(f"[Chat] User: {question}")

    history = request.conversation_history

    # ── Session profile ───────────────────────────────────────────────────
    profile      = extract_profile(history)
    profile_text = profile_to_text(profile)
    print(f"[Chat] Session profile: {profile_text}")

    # ── History text for system prompt ────────────────────────────────────
    history_text = (
        "No previous conversation in this session."
        if not history
        else "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in history[-8:]
        )
    )

    # ── ChromaDB retrieval ────────────────────────────────────────────────
    hits          = retrieve_from_db(question)
    best_distance = hits[0][2] if hits else 999
    relevant_hits = [(doc, meta, dist) for doc, meta, dist in hits if dist < DB_THRESHOLD]
    scheme_found  = len(relevant_hits) > 0
    db_context    = build_db_context(relevant_hits) if relevant_hits else "No matching scheme found in local database."
    source_url    = relevant_hits[0][1].get("source_url", "") if relevant_hits else ""

    print(
        f"[Chat] ChromaDB: best_dist={best_distance:.2f} | "
        f"scheme_found={scheme_found} | "
        f"top={hits[0][1]['name'] if hits else 'None'}"
    )

    # ── System prompt ─────────────────────────────────────────────────────
    system_prompt = build_system_prompt(
        profile_text=profile_text,
        history_text=history_text,
        db_context=db_context,
        scheme_found=scheme_found,
        source_url=source_url,
    )

    # ── Messages — system + real history + current question ──────────────
    messages = [{"role": "system", "content": system_prompt}]

    # Real conversation history as proper message turns (fixes follow-up bug)
    for h in history[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})

    # Current question
    messages.append({"role": "user", "content": question})

    # ── MCP + LLM ─────────────────────────────────────────────────────────
    async with MCPClient() as mcp_client:
        answer, tools_used = await run_llm_with_tools(messages, mcp_client)

    print(f"[Chat] Tools used: {tools_used if tools_used else 'none'}")

    # ── Fallback ──────────────────────────────────────────────────────────
    if not answer:
        answer = (
            "I couldn't find reliable information on that right now. "
            "Please visit **myscheme.gov.in** or call helpline **14567** for assistance."
        )

    # ── Confidence ────────────────────────────────────────────────────────
    if scheme_found and best_distance < 10:
        confidence = "high"
    elif scheme_found or tools_used:
        confidence = "medium"
    else:
        confidence = "no_match"

    # ── Sources ───────────────────────────────────────────────────────────
    seen: set       = set()
    sources: list[dict] = []
    if scheme_found:
        for _, meta, _ in relevant_hits[:2]:
            if meta["name"] not in seen:
                seen.add(meta["name"])
                sources.append({
                    "name"         : meta["name"],
                    "category"     : meta.get("category", ""),
                    "source_url"   : meta.get("source_url", ""),
                    "last_verified": meta.get("last_verified", ""),
                })

    return ChatResponse(answer=answer, sources=sources, confidence=confidence)


# ══════════════════════════════════════════════════════════════════════════════
# Utility endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status"       : "ok",
        "scheme_count" : collection.count(),
        "groq_enabled" : groq_client is not None,
        "groq_model"   : GROQ_MODEL,
        "db_threshold" : DB_THRESHOLD,
    }


@app.get("/schemes")
async def list_schemes():
    results = collection.get(include=["metadatas"])
    return [
        {"name": m["name"], "category": m.get("category", "")}
        for m in results["metadatas"]
    ]