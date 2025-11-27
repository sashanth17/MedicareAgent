import asyncio
import threading
import re
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import urllib3

# Silence InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from crew import CrewInit

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CrewStreamParser:
    """
    Advanced Parser to handle fragmented streaming tokens.
    It buffers the stream to detect the 'Final Answer' start
    and strictly cuts off the stream BEFORE the 'suggestions' JSON block starts.
    """
    def __init__(self):
        self.buffer = ""
        self.streaming_active = False # Have we found "Final Answer"?
        self.finished = False         # Have we hit the JSON block?
        
        # Pattern to find the start of the answer
        self.start_pattern = re.compile(r"(Final Answer\s*:|Final Answer\s*|Answer\s*:)\s*", re.IGNORECASE)
        
        # Pattern to detect the START of the memory/suggestion block.
        # It handles newlines and spaces between { and "suggestions"
        # Matches: { "suggestions" OR { "quality" OR { "entities"
        self.stop_pattern = re.compile(r'\{\s*"\s*(suggestions|quality|entities)', re.DOTALL)

    def process_chunk(self, chunk: str) -> str:
        """
        Input: Raw chunk from CrewAI
        Output: Clean text to send to user (or empty string if buffering)
        """
        if self.finished:
            return ""

        # 1. Clean artifacts
        # Remove "Bot:", "TEXT_CHUNK:", etc.
        clean_chunk = chunk
        for prefix in ["Bot:", "TEXT_CHUNK:", "User:", "You:"]:
            if prefix in clean_chunk:
                clean_chunk = clean_chunk.replace(prefix, "")
        
        self.buffer += clean_chunk

        # 2. PHASE 1: Waiting for "Final Answer"
        if not self.streaming_active:
            match = self.start_pattern.search(self.buffer)
            if match:
                self.streaming_active = True
                # Discard the prefix "Final Answer:", keep the rest
                self.buffer = self.buffer[match.end():]
            else:
                # Fallback: If buffer gets huge (>400 chars) without "Final Answer", 
                # but also without "Action:", assume it's small talk and force start.
                if len(self.buffer) > 400 and "Action:" not in self.buffer:
                     self.streaming_active = True

        # 3. PHASE 2: Streaming content (with JSON protection)
        output_text = ""
        if self.streaming_active:
            # Check if the STOP pattern is fully present
            stop_match = self.stop_pattern.search(self.buffer)
            
            if stop_match:
                # FOUND IT! The JSON block has started.
                # Return everything UP TO the start of the match (the '{')
                self.finished = True
                cutoff_index = stop_match.start()
                output_text = self.buffer[:cutoff_index]
                self.buffer = "" # clear buffer, we are done
                return output_text.strip()
            
            # EDGE CASE: The Buffer might end with half a token like '{' or '{ "'
            # We must NOT send this yet, because the next chunk might be 'suggestions":'
            
            # Find the last open brace
            last_brace_index = self.buffer.rfind('{')
            
            if last_brace_index != -1:
                # If there is a brace, check if it's "suspiciously" close to the end
                # (i.e. we are waiting for the rest of the JSON key)
                suspicious_length = len(self.buffer) - last_brace_index
                if suspicious_length < 50: 
                    # Hold back text from the brace onwards
                    # Send everything before the brace
                    output_text = self.buffer[:last_brace_index]
                    self.buffer = self.buffer[last_brace_index:] # Keep the suspicious part in buffer
                    return output_text
            
            # If no suspicious braces, flush the whole buffer
            output_text = self.buffer
            self.buffer = ""
            
        return output_text

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    crew = CrewInit().create_crew()
    loop = asyncio.get_running_loop()
    
    # Instantiate the parser
    parser = CrewStreamParser()

    try:
        while True:
            query = await websocket.receive_text()
            if query.lower().strip() in ("exit", "quit"):
                await websocket.send_text("Goodbye! 👋")
                await websocket.close()
                break

            # Reset parser for new turn
            parser = CrewStreamParser()
            queue = asyncio.Queue()

            def run_crew():
                try:
                    stream = crew.kickoff(inputs={"query": query})
                    for chunk in stream:
                        content = getattr(chunk, "content", str(chunk))
                        asyncio.run_coroutine_threadsafe(queue.put(content), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(f"[ERROR] {e}"), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            threading.Thread(target=run_crew, daemon=True).start()

            while True:
                raw_chunk = await queue.get()
                
                if raw_chunk is None: 
                    # Stream ended. 
                    # If there's anything left in buffer that wasn't JSON, flush it.
                    if parser.buffer and not parser.finished:
                        # Safety check: if buffer looks like just a brace, ignore it
                        if parser.buffer.strip() != "{":
                            await websocket.send_text(parser.buffer)
                    break
                
                if isinstance(raw_chunk, str) and raw_chunk.startswith("[ERROR]"):
                    continue # Skip error logs in stream

                # Parse and Send
                clean_text = parser.process_chunk(str(raw_chunk))
                if clean_text:
                    await websocket.send_text(clean_text)

    except WebSocketDisconnect:
        print("❌ Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)