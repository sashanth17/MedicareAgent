import asyncio
from fastapi import WebSocket


async def websocket_stream_worker(websocket: WebSocket, queue: asyncio.Queue[str]) -> None:
    """
    Async consumer that reads messages from an asyncio queue and forwards
    them to a WebSocket client in real-time.

    Args:
        websocket (WebSocket): The WebSocket connection to send messages to.
        queue (asyncio.Queue[str]): Queue containing messages to stream. 
            A None value signals the end of the stream.
    """
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:  # Sentinel for end-of-stream
                break

            await websocket.send_text(chunk)
            # Yield control to the event loop to allow other tasks to run
            await asyncio.sleep(0)

    except Exception as e:
        # Expected on client disconnects; log for debugging
        print("websocket_stream_worker error:", e)