import asyncio
import threading
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import urllib3

# optional: silence InsecureRequestWarning for local dev (only for dev)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from crew import CrewInit
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

from utils import websocket_stream_worker
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    import re
    import time

    await websocket.accept()
    crew = CrewInit().create_crew()
    loop = asyncio.get_running_loop()

    FINAL_RE = re.compile(r"final answer\s*:\s*", re.IGNORECASE)
    JSON_LIKELY_RE = re.compile(r'"\s*:\s*')   # `"key":` indicates JSON-like
    CHUNK_NUMBER_RE = re.compile(r"chunk\d+", re.IGNORECASE)

    # Phrases that indicate internal evaluator suggestions / instructions
    SUGGESTION_PATTERNS = [
        r"\bthe model should\b",
        r"\bthe model must\b",
        r"\bshould ensure\b",
        r"\bshould avoid\b",
        r"\bavoid adding\b",
        r"\bensure that\b",
        r"\binclude examples\b",
        r"\badd test cases\b",
        r"\bexplicitly state\b",
        r"\bthe tool's output\b",
        r"\bsuggestions\b",
        r"\bquality\b",
        r"\bentities\b",
        r"\bimprove its ability\b",
        r"\bthe model should improve\b",
        r"\bthe model should avoid\b",
        r"\bthe model should ensure\b",
        r"\boutputting the exact string\b",
        r"\bthe model should not\b",
        r"\binstruction\b",
    ]
    SUGGESTION_RE = re.compile("|".join(SUGGESTION_PATTERNS), re.IGNORECASE)

    # Reused reasoning detection
    REASONING_PATTERNS = [
        r"\bthe user\b",
        r"\bi should\b",
        r"\btherefore\b",
        r"\bthought\b",
        r"\baction\b",
        r"\baction input\b",
    ]
    REASONING_RE = re.compile("|".join(REASONING_PATTERNS), re.IGNORECASE)

    def sanitize_chunk(text: str) -> str:
        """Normalize and remove chunk artifacts and prefixes."""
        if not text:
            return ""
        t = text.strip()

        # Drop obvious JSON-like chunks immediately
        if JSON_LIKELY_RE.search(t):
            return ""

        # Remove chunk numbering artifacts
        t = CHUNK_NUMBER_RE.sub("", t)

        # Remove typical prefixes
        for p in ("Bot:", "bot:", "Assistant:", "assistant:", "assistant-"):
            if t.startswith(p):
                t = t[len(p):].lstrip()

        # Strip surrounding quotes/brackets/commas
        t = t.strip(" \t\n\r,\"'[]{}")

        return t

    def is_suggestion_only(text: str) -> bool:
        """Return True if the chunk is a pure suggestion/evaluator sentence."""
        if not text:
            return True
        # If chunk starts/ends with quote block or contains many quotes -> likely suggestion
        if text.startswith('"') or text.endswith('"'):
            return True
        # If matches suggestion patterns anywhere -> likely suggestion
        if SUGGESTION_RE.search(text):
            # if the chunk also contains concrete user-facing keywords (pharmacy names, phone, 🏪),
            # we'll treat it as mixed and try to salvage non-suggestion sentences later.
            return True
        # very short single tokens like "None" or "OK" considered internal
        if re.fullmatch(r"(none|ok|true|false|-|—)", text.strip(), re.IGNORECASE):
            return True
        return False

    def is_internal_reasoning(text: str) -> bool:
        """Detect CoT / action / control lines."""
        if not text:
            return True
        if REASONING_RE.search(text):
            return True
        if re.fullmatch(r"(action|thought|final answer|none|ok|true|false)\b[:]?.*", text, re.IGNORECASE):
            return True
        return False

    def strip_suggestion_sentences(text: str) -> str:
        """
        If text mixes useful content with suggestion sentences, remove the
        suggestion sentences and return the remainder.
        """
        # Split into sentences conservatively (punctuation or newline)
        parts = re.split(r"(?<=[\.\?\!])\s+|\n", text)
        kept = []
        for p in parts:
            p_clean = p.strip()
            if not p_clean:
                continue
            # drop if suggestion-like or JSON-like or internal
            if JSON_LIKELY_RE.search(p_clean) or SUGGESTION_RE.search(p_clean) or REASONING_RE.search(p_clean):
                continue
            kept.append(p_clean)
        return " ".join(kept).strip()

    try:
        while True:
            query = await websocket.receive_text()
            if query.lower().strip() in ("exit", "quit"):
                await websocket.send_text("Goodbye! 👋")
                await websocket.close()
                break

            q: asyncio.Queue = asyncio.Queue()

            def run_sync():
                """Background thread: push raw chunk content into the async queue."""
                try:
                    for chunk in crew.kickoff(inputs={"query": query}):
                        raw = getattr(chunk, "content", None)
                        if raw is None:
                            try:
                                raw = str(chunk)
                            except Exception:
                                raw = None
                        if raw:
                            asyncio.run_coroutine_threadsafe(q.put(str(raw)), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(q.put(f"[ERROR]{e}"), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(q.put(None), loop)

            threading.Thread(target=run_sync, daemon=True).start()

            send_buffer = ""
            buffering_final = False

            async def flush_buffer():
                nonlocal send_buffer
                if send_buffer:
                    await websocket.send_text(send_buffer.strip())
                    send_buffer = ""

            while True:
                raw_chunk = await q.get()
                if raw_chunk is None:
                    await flush_buffer()
                    break

                # error pass-through
                if isinstance(raw_chunk, str) and raw_chunk.startswith("[ERROR]"):
                    await websocket.send_text(raw_chunk)
                    continue

                text = sanitize_chunk(str(raw_chunk))
                if not text:
                    continue

                # If Final Answer marker present, prefer after it
                if FINAL_RE.search(text):
                    buffering_final = True
                    text = FINAL_RE.split(text)[-1].strip()
                    if not text:
                        continue

                # If the whole chunk looks like suggestion/internal -> drop it
                if is_suggestion_only(text) and not re.search(r"\b(🏪|pharmacy|phone|contact|📞|\d{6,})", text):
                    # If suggestion-like but contains a phone number / pharmacy emoji, we will try to salvage
                    continue

                # If mixed (contains suggestion phrases but also real content), strip suggestion sentences
                if SUGGESTION_RE.search(text) or REASONING_RE.search(text):
                    stripped = strip_suggestion_sentences(text)
                    if not stripped:
                        continue
                    text = stripped

                # Accumulate for smoother TTS; flush on punctuation or size
                send_buffer += (" " + text) if send_buffer else text
                if re.search(r"[.!?]\s*$", text) or len(send_buffer) > 160:
                    await flush_buffer()

            # end per-query loop

    except WebSocketDisconnect:
        print("❌ Client disconnected")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)