import asyncio
import threading
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import urllib3

# optional: silence InsecureRequestWarning for local dev (only for dev)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from crew import CrewInit
load_dotenv()
app = FastAPI()





from utils import websocket_stream_worker

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    crew = CrewInit().create_crew()
    loop = asyncio.get_running_loop()  # capture main asyncio loop

    try:
        while True:
            query = await websocket.receive_text()
            if query.lower().strip() in ("exit", "quit"):
                await websocket.send_text("Goodbye! 👋")
                await websocket.close()
                return

            q: asyncio.Queue = asyncio.Queue()

            def run_blocking_kickoff():
                try:
                    streaming = crew.kickoff(inputs={"query": query})
                    for chunk in streaming:
                        if isinstance(chunk, tuple) and len(chunk) >= 2:
                            text = str(chunk[1])
                        elif hasattr(chunk, "content"):
                            text = str(chunk.content)
                        else:
                            text = str(chunk)
                        # Push to async queue safely from thread
                        asyncio.run_coroutine_threadsafe(q.put(text), loop)
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(q.put(f"[error] {repr(exc)}"), loop)
                finally:
                    # signal end-of-stream
                    asyncio.run_coroutine_threadsafe(q.put(None), loop)

            thread = threading.Thread(target=run_blocking_kickoff, daemon=True)
            thread.start()

            # Consume queue in async
            while True:
                text = await q.get()
                if text is None:
                    break
                await websocket.send_text(text)

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        print("Connection closed")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)