"""Local real-time voice agent.

Browser mic (16 kHz PCM16) -> WebSocket -> Silero VAD -> Smart Turn v3 -> Parakeet (MLX) -> LLM (OpenAI-compatible,
DeepSeek on Bedrock by default) -> Phonon TTS (CPU, streaming) -> WebSocket -> browser speaker (24 kHz PCM16).

The LLM streams into a sentence chunker, and each chunk is synthesized while the LLM keeps writing, so the
first audio goes out as soon as the first clause exists. User speech during a reply cancels it (barge-in).

    .venv/bin/python server.py      then open http://127.0.0.1:8000
"""
import asyncio, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://bedrock-mantle.us-west-2.api.aws/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek.v3.2")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))
ASR_BACKEND = os.getenv("ASR_BACKEND", "mlx")  # "mlx" (parakeet-mlx) or "redux" (Moondream Parakeet Redux, CPU; needs moondream + torch<2.15)
ASR_MODEL = os.getenv("ASR_MODEL", "moondream/parakeet-redux" if ASR_BACKEND == "redux" else "mlx-community/parakeet-tdt-0.6b-v3")
TTS_VOICE = os.getenv("TTS_VOICE", "Harper")
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", "900"))  # longest pause tolerated before a turn is forced to end
TURN_CHECK_MS = int(os.getenv("TURN_CHECK_MS", "200"))  # silence after which Smart Turn is asked "are they done?"
TURN_THRESHOLD = float(os.getenv("TURN_THRESHOLD", "0.5"))  # Smart Turn probability that counts as "done"
BARGE_IN_MS = int(os.getenv("BARGE_IN_MS", "250"))  # sustained speech needed to interrupt the bot
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
PORT = int(os.getenv("PORT", "8000"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are Ava, a warm, quick-witted voice assistant talking with someone out loud, in real time.
Everything you write is spoken by a text-to-speech voice, so:
- Talk like a person on a call: casual, natural, contractions, the occasional "oh", "hmm" or "yeah" where it fits.
- Keep replies short: usually one to three sentences. Offer more only if asked.
- Never use markdown, lists, headings, emojis, code or URLs. Write numbers and symbols the way they are said.
- If something you heard seems garbled or cut off, just ask them to say it again.
- You can't browse the web or see live data; say so briefly if asked.""")

SR_IN, FRAME = 16000, 512  # Silero VAD takes 512-sample frames at 16 kHz (32 ms)


# ---------------------------------------------------------------- models
class SileroVAD:
    """Silero VAD v5 over ONNX; one instance per connection, since it is stateful."""
    session = None

    def __init__(self):
        if SileroVAD.session is None:
            o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
            SileroVAD.session = ort.InferenceSession(str(ROOT / "models/silero_vad.onnx"), o, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, 64), np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame[None, :]], axis=1)
        out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": np.array(SR_IN, np.int64)})
        self.context = x[:, -64:]
        return float(out[0, 0])


class SmartTurn:
    """Pipecat Smart Turn v3 (Whisper-tiny encoder, int8 ONNX): looks at the audio of the turn so far and says
    whether the speaker is finished or just pausing. ~10-30 ms on CPU. None if the model file is missing."""
    WINDOW = SR_IN * 8

    def __init__(self, path: Path):
        from transformers import WhisperFeatureExtractor
        o = ort.SessionOptions(); o.intra_op_num_threads = 2; o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(chunk_length=8)
        self(np.zeros(SR_IN, np.float32))  # warm up

    def __call__(self, audio: np.ndarray) -> float:
        audio = audio[-self.WINDOW:]
        f = self.fe(audio, sampling_rate=SR_IN, return_tensors="np", padding="max_length", max_length=self.WINDOW,
                    truncation=True, do_normalize=True)["input_features"].astype(np.float32)
        return float(self.session.run(None, {"input_features": f})[0].reshape(-1)[0])


_st_path = ROOT / "models/smart-turn-v3.2-cpu.onnx"
smart_turn = SmartTurn(_st_path) if _st_path.exists() else None
turn_pool = ThreadPoolExecutor(1, thread_name_prefix="turn")

asr_pool = ThreadPoolExecutor(1, thread_name_prefix="asr")  # MLX stays on one thread
asr_model = None


def _load_asr():
    global asr_model
    if ASR_BACKEND == "redux":
        import moondream as md
        asr_model = md.photon(ASR_MODEL, device="cpu")
    else:
        from parakeet_mlx import from_pretrained
        asr_model = from_pretrained(ASR_MODEL)
    _transcribe(np.zeros(SR_IN, np.float32))


def _transcribe(pcm: np.ndarray) -> str:
    if ASR_BACKEND == "redux":
        return asr_model.transcribe(audio=pcm, sample_rate=SR_IN)["text"].strip()
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel
    mel = get_logmel(mx.array(pcm), asr_model.preprocessor_config)
    return asr_model.generate(mel)[0].text.strip()


import ptts  # noqa: E402

tts = ptts.TTS("hf://gradium/phonon-31m-en", lang="en", quant="q8_0", device="cpu", voice=TTS_VOICE)
TTS_SR = tts.sample_rate
VOICES = sorted(v["name"] for v in ptts.catalog_voices() if v.get("language") == "en")

from openai import AsyncOpenAI  # noqa: E402

llm = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "missing", timeout=30, max_retries=1)


# ---------------------------------------------------------------- text helpers
_SENT_END = re.compile(r"([.!?…]+[\"')\]]*)(\s+)")
_CLAUSE = re.compile(r"([,;:—–])(\s+)")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")


def clean_for_speech(t: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)
    t = _EMOJI.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def pop_chunk(buf: str, first: bool) -> tuple[str | None, str]:
    """Cut the next speakable piece off the front of `buf`. The first piece may end at a clause so audio starts sooner."""
    m = _SENT_END.search(buf)
    if m and m.end(1) >= 8:
        return buf[:m.end(1)], buf[m.end():]
    if first or len(buf) > 120:
        for m in _CLAUSE.finditer(buf):
            if m.end(1) >= (14 if first else 60):
                return buf[:m.end(1)], buf[m.end():]
    return None, buf


# ---------------------------------------------------------------- per-connection session
class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.vad = SileroVAD()
        self.voice = TTS_VOICE
        self.history: list[dict] = []
        self.reply: asyncio.Task | None = None
        self.client_playing = False
        # VAD / turn state
        self.pending = np.zeros(0, np.float32)
        self.preroll: list[np.ndarray] = []
        self.speech: list[np.ndarray] = []
        self.in_speech = False
        self.speech_ms = 0
        self.silence_ms = 0
        self.next_check_ms = TURN_CHECK_MS
        self.spec: asyncio.Future | None = None  # speculative ASR of the audio so far

    async def send(self, **msg):
        await self.ws.send_text(json.dumps(msg))

    @property
    def bot_active(self) -> bool:
        return self.client_playing or (self.reply is not None and not self.reply.done())

    async def on_audio(self, data: bytes):
        self.pending = np.concatenate([self.pending, np.frombuffer(data, np.int16).astype(np.float32) / 32768])
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            await self.on_frame(frame)

    async def on_frame(self, frame: np.ndarray):
        p = self.vad(frame)
        voiced = p >= VAD_THRESHOLD
        if not self.in_speech:
            self.preroll = (self.preroll + [frame])[-10:]  # keep ~320 ms before speech onset
            self.speech_ms = self.speech_ms + 32 if voiced else 0
            if self.speech_ms >= 96:  # three voiced frames in a row starts a turn
                self.in_speech, self.silence_ms = True, 0
                self.speech = list(self.preroll)
                await self.send(type="vad", speaking=True)
            return
        self.speech.append(frame)
        if voiced:
            self.speech_ms += 32
            self.silence_ms = 0
            if self.bot_active and self.speech_ms >= BARGE_IN_MS:
                await self.interrupt()
        else:
            self.silence_ms += 32
            if smart_turn is None:  # no model: plain silence timeout
                if self.silence_ms >= min(END_SILENCE_MS, 550):
                    await self.end_turn()
            elif self.silence_ms >= END_SILENCE_MS:
                await self.end_turn()
            elif self.silence_ms >= self.next_check_ms:
                self.next_check_ms += TURN_CHECK_MS
                await self.check_turn()

    async def check_turn(self):
        """Pause detected: guess the transcript and ask Smart Turn if the user is done, concurrently. If they are,
        the transcript is usually ready by the time we commit; if not, the guess is thrown away."""
        loop = asyncio.get_running_loop()
        audio = np.concatenate(self.speech)
        if len(audio) < SR_IN * 0.3:
            return
        if self.spec is None or self.spec.done():
            self.spec = loop.run_in_executor(asr_pool, _transcribe, audio)
            spec = self.spec
        else:
            spec = None  # earlier guess still running; it's stale either way
        p = await loop.run_in_executor(turn_pool, smart_turn, audio)
        if p >= TURN_THRESHOLD and spec is not None:
            await self.end_turn(spec)
        elif p >= TURN_THRESHOLD:
            await self.end_turn()

    async def end_turn(self, spec: asyncio.Future | None = None):
        audio = np.concatenate(self.speech)
        waited = self.silence_ms
        self.in_speech, self.speech, self.speech_ms, self.silence_ms = False, [], 0, 0
        self.next_check_ms = TURN_CHECK_MS
        self.spec = None
        await self.send(type="vad", speaking=False)
        if len(audio) < SR_IN * 0.3:
            return
        t_end = time.perf_counter() - waited / 1000  # when the user actually stopped talking
        await self.interrupt()
        self.reply = asyncio.create_task(self.respond(audio, t_end, spec))

    async def interrupt(self):
        if self.reply and not self.reply.done():
            self.reply.cancel()
            try:
                await self.reply
            except (asyncio.CancelledError, Exception):
                pass
        if self.client_playing:
            await self.send(type="interrupt")
            self.client_playing = False

    async def respond(self, audio: np.ndarray, t_end: float, spec: asyncio.Future | None = None):
        loop = asyncio.get_running_loop()
        t_turn = time.perf_counter()
        text = await (spec or loop.run_in_executor(asr_pool, _transcribe, audio))
        t_asr = time.perf_counter()
        if not text or not re.search(r"\w", text):
            return
        # stage durations: turn = silence before we decided, asr = wait for the transcript after that
        await self.reply_to(text, t_end, {"turn_ms": round((t_turn - t_end) * 1000), "asr_ms": round((t_asr - t_turn) * 1000)})

    async def reply_to(self, text: str, t_end: float, metrics: dict):
        await self.send(type="user", text=text)
        self.history.append({"role": "user", "content": text})
        spoken: list[str] = []
        chunks: asyncio.Queue[str | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self.speak(chunks, spoken, metrics, t_end))
        try:
            await self.stream_llm(chunks, metrics, t_end)
            await tts_task
        except asyncio.CancelledError:
            tts_task.cancel()
            raise
        except Exception as e:  # LLM/network errors: show them instead of dying silently
            tts_task.cancel()
            await self.send(type="error", text=f"LLM error: {e}")
        finally:
            said = " ".join(spoken).strip()
            if said:  # history keeps only what was actually voiced, so barge-in leaves it truthful
                self.history.append({"role": "assistant", "content": said})
            elif self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            self.history = self.history[-30:]
        await self.send(type="assistant_done", metrics=metrics)

    async def stream_llm(self, chunks: asyncio.Queue, metrics: dict, t_end: float):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        t_llm = time.perf_counter()
        stream = await llm.chat.completions.create(model=LLM_MODEL, messages=messages, stream=True,
                                                   max_tokens=LLM_MAX_TOKENS, temperature=0.7)
        buf, first = "", True
        try:
            async for ev in stream:
                if not ev.choices:
                    continue
                delta = ev.choices[0].delta.content or ""
                if not delta:
                    continue
                if "llm_ttft_ms" not in metrics:
                    metrics["llm_ttft_ms"] = round((time.perf_counter() - t_llm) * 1000)
                buf += delta
                while True:
                    piece, buf = pop_chunk(buf, first)
                    if piece is None:
                        break
                    first = False
                    await chunks.put(piece)
        finally:
            await stream.close()
        if buf.strip():
            await chunks.put(buf)
        await chunks.put(None)

    async def speak(self, chunks: asyncio.Queue, spoken: list[str], metrics: dict, t_end: float):
        loop = asyncio.get_running_loop()
        while (piece := await chunks.get()) is not None:
            text = clean_for_speech(piece)
            if not re.search(r"\w", text):
                continue
            await self.send(type="assistant", text=text)
            out: asyncio.Queue = asyncio.Queue()
            stop = False

            def run(text=text):
                with tts.stream(text, voice=self.voice) as s:
                    for ch in s:
                        if stop:
                            break
                        loop.call_soon_threadsafe(out.put_nowait, ch)
                loop.call_soon_threadsafe(out.put_nowait, None)

            fut = loop.run_in_executor(None, run)
            try:
                while (ch := await out.get()) is not None:
                    if "first_audio_ms" not in metrics:
                        metrics["first_audio_ms"] = round((time.perf_counter() - t_end) * 1000)
                        metrics["tts_ms"] = metrics["first_audio_ms"] - sum(metrics.get(k, 0) for k in ("turn_ms", "asr_ms", "llm_ttft_ms"))
                        await self.send(type="metrics", metrics=metrics)
                    self.client_playing = True
                    await self.ws.send_bytes((np.clip(ch, -1, 1) * 32767).astype("<i2").tobytes())
            finally:
                stop = True
            await fut
            spoken.append(text)


# ---------------------------------------------------------------- app
app = FastAPI()


@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    t = time.perf_counter()
    await loop.run_in_executor(asr_pool, _load_asr)
    tts.synth("Ready.")
    print("turn detection:", "Smart Turn v3.2" if smart_turn else f"silence timeout (models/smart-turn-v3.2-cpu.onnx missing)")
    print(f"models ready in {time.perf_counter() - t:.1f}s  |  LLM {LLM_MODEL} @ {LLM_BASE_URL}  |  http://127.0.0.1:{PORT}")
    if not LLM_API_KEY:
        print("WARNING: LLM_API_KEY is not set; put your Bedrock API key in .env")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/config")
async def config():
    return {"voices": VOICES, "voice": TTS_VOICE, "model": LLM_MODEL, "sample_rate": TTS_SR}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    s = Session(ws)
    await s.send(type="ready", sample_rate=TTS_SR)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await s.on_audio(msg["bytes"])
            elif msg.get("text"):
                m = json.loads(msg["text"])
                if m["type"] == "playback":
                    s.client_playing = m["playing"]
                elif m["type"] == "voice" and m["voice"] in VOICES:
                    s.voice = m["voice"]
                elif m["type"] == "reset":
                    await s.interrupt()
                    s.history.clear()
                elif m["type"] == "text":  # typed input, same pipeline minus ASR
                    await s.interrupt()
                    s.reply = asyncio.create_task(s.reply_to(m["text"], time.perf_counter(), {"turn_ms": 0, "asr_ms": 0}))
    except WebSocketDisconnect:
        pass
    finally:
        if s.reply:
            s.reply.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
