import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from queue import Queue, Empty, Full
import threading
import time

# === TTS payload ===
payload = {
    "text": "Hello This is a test of the Zonos TTS WebSocket endpoint. We are checking if it is streaming live or not as it is our final task of the day so this is the final querry text that we are using",
}

samplerate = 44100
channels = 1
audio_queue = Queue(maxsize=100)
audio_chunks = []

# === These should match server schedule ===
chunk_schedule = [17] + list(range(9, 100))
frame_size = 1024

# Increased buffer size for smoother playback
expected_buffer_samples = sum(chunk_schedule[:4]) * frame_size

def stream_player():
    with sd.OutputStream(samplerate=samplerate, channels=channels, dtype='float32', 
                        blocksize=frame_size, latency='high') as stream:
        while True:
            try:
                chunk = audio_queue.get(timeout=0.2)
                if chunk is None:
                    break
                stream.write(chunk.reshape(-1, 1))
            except Empty:
                continue

async def test_tts_ws():
    uri = "wss://hcwdxdukdbckql-8000.proxy.runpod.net/ws/tts"
    

    player_thread = None
    accumulated_samples = 0
    last_chunk_time = time.time()

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(payload))
        print("Sent TTS request...")

        try:
            while True:
                message = await websocket.recv()

                if isinstance(message, bytes):
                    chunk = np.frombuffer(message, dtype=np.float32)
                    audio_chunks.append(chunk)
                    
                    # Add timing check to prevent queue overflow
                    current_time = time.time()
                    if current_time - last_chunk_time > 0.1:  # If gap is too large
                        print("Warning: Large gap detected in audio stream")
                    
                    # Try to put chunk in queue with timeout
                    try:
                        audio_queue.put(chunk, timeout=0.1)
                    except Queue.Full:
                        print("Warning: Audio queue is full, dropping chunk")
                        continue
                        
                    accumulated_samples += len(chunk)
                    last_chunk_time = current_time

                    # Start the playback thread once enough samples have accumulated
                    if not player_thread and accumulated_samples >= expected_buffer_samples:
                        print("Starting playback after buffering...")
                        player_thread = threading.Thread(target=stream_player, daemon=True)
                        player_thread.start()

                elif isinstance(message, str) and "STREAM_COMPLETE" in message:
                    print("Stream complete.")
                    break

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket closed by server.")

    # Finalize playback
    audio_queue.put(None)
    if player_thread:
        player_thread.join()

    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        int_audio = np.int16(full_audio * 32767)
        wavfile.write("tts_stream.wav", samplerate, int_audio)
        print("Audio saved to 'tts_stream.wav'.")

asyncio.run(test_tts_ws())
