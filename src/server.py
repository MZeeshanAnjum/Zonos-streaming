import datetime
import logging
import numpy as np
import torch
from fastapi import FastAPI
from fastrtc import Stream, AsyncStreamHandler, wait_for_item
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
import asyncio
from pydantic import BaseModel
from typing import Optional, List
import torchaudio
import time
 
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
 
# === Request schema ===
class TTSRequest(BaseModel):
    text: str
    emotion: Optional[List[float]] = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
    vqscore: Optional[float] = 0.78
    fmax: int = 24000
    pitch_std: float = 45.0
    speaking_rate: float = 15.0
    dnsmos_ovrl: float = 4.0
    speaker_noised: bool = False
    cfg_scale: float = 2.8
    unconditional_keys: Optional[List[str]] = ["speaking_rate"]
    sampling_params: Optional[dict] = {
        "top_p": 0.85,
        "top_k": 20,
        "min_p": 0.25,
        "linear": 0.3,
        "conf": 0.4,
        "quad": 0.0,
    }
    speaker_audio_base64: Optional[str] = None  # Removed since we are no longer processing speaker audio per request
 
# === Utility ===
def fix_base64_padding(b64_string):
    return b64_string + '=' * (-len(b64_string) % 4)
 
def decode_audio(file_path):
    # Assuming neutral audio is a fixed reference file for now
    audio_tensor, sr = torchaudio.load(file_path, normalize=True)
    return audio_tensor, sr
 
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
logging.info(device)
model.requires_grad_(False).eval()
# Preload neutral speaker embedding
logging.info("Preloading neutral speaker embedding...")
neutral_audio_tensor, _ = decode_audio("neutral.mp3")  # Assuming you have a neutral audio sample file
neutral_speaker_embedding = model.make_speaker_embedding(neutral_audio_tensor, 44100).to(device, dtype=torch.bfloat16)
 
# === Model Warmup (ONCE) ===
logging.info("Starting model warmup...")
warmup_start = time.time()
with torch.no_grad():
    logging.info("Generating 20 tokens for warmup...")
    dummy_cond = make_cond_dict(
        text="Warmup inference for zonos model to reduce first call latency",
        language="en-us",
        speaker=neutral_speaker_embedding,
        device=device
    )
    dummy_conditioning = model.prepare_conditioning(dummy_cond)
    _ = model.generate(prefix_conditioning=dummy_conditioning, max_new_tokens=20)
warmup_time = time.time() - warmup_start
logging.info(f"Model warmup completed in {warmup_time:.2f} seconds")
 
sample_rate = model.autoencoder.sampling_rate
startup_complete_time = time.time()
logging.info(f"Server sample rate: {sample_rate}")
 
 
class ZonosTTSHandler(AsyncStreamHandler):
    def __init__(self, model: model):
        super().__init__(output_sample_rate=sample_rate)
        self.model = model
        self.audio_queue = asyncio.Queue()
        self.text = None
        self._stream_started = False
 
    def copy(self):
        return ZonosTTSHandler(self.model)
 
    async def start_up(self):
        logging.info("[TTS] Handler startup complete.")
 
    async def receive(self, text: str):
        self.text = text
        self._stream_started = False
        logging.info(f"[TTS] Received text chunk {text}")
 
    async def emit(self):
 
        if not self._stream_started and self.text is not None:
            self._stream_started = True
 
            req = TTSRequest(text=self.text)
            speaker_embedding = neutral_speaker_embedding
 
            # Step 2: Prepare conditioning
            # logging.info("[WS] Creating conditioning dictionary...")
            # start = time.time()
            emotion_tensor = torch.tensor([req.emotion], device=device)
            vq_tensor = torch.tensor([[req.vqscore] * 8], device=device)
            cond_dict = make_cond_dict(
                text=req.text,
                language="en-us",
                speaker=speaker_embedding,
                emotion=emotion_tensor,
                vqscore_8=vq_tensor,
                fmax=req.fmax,
                pitch_std=req.pitch_std,
                speaking_rate=req.speaking_rate,
                dnsmos_ovrl=req.dnsmos_ovrl,
                speaker_noised=req.speaker_noised,
                device=device,
                unconditional_keys=["speaking_rate"],
            )
            conditioning = model.prepare_conditioning(cond_dict)
 
            stream_generator = model.stream(
                prefix_conditioning=conditioning,
                audio_prefix_codes=None,
                chunk_schedule=[17, *range(9, 20)],
                chunk_overlap=1,
                cfg_scale=req.cfg_scale,
                sampling_params=req.sampling_params,
            )
 
 
            for audio_chunk in stream_generator:
 
                yield (sample_rate, audio_chunk.cpu().numpy().astype(np.float32))
 
 
    async def shutdown(self):
        logging.info("[TTS] Shutdown called.")
        self._stream_started = False
 
handler = ZonosTTSHandler(model)
 
# FastAPI app
app = FastAPI()
 
# FastRTC stream
stream = Stream(
    handler=handler,
    modality="audio",
    mode="send-receive",
)
stream.mount(app, path="/rtc/tts")
