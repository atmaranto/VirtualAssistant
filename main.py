import time

from io import BytesIO
from faster_whisper import WhisperModel

from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from threading import Thread, Lock

import numpy as np

model = ChatOllama(model="llama3.1")

class Eventable:
    def __init__(self):
        self._event_handlers = {}

    def on(self, event_name, handler):
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    def emit(self, event_name, *args, **kwargs):
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                handler(*args, **kwargs)

class Assistant(Eventable):
    def __init__(self, llm: BaseChatModel, model: WhisperModel, audio_rate: int = 16000, wake_words=[]):
        super().__init__()
        self.llm = llm
        self.model = model
        self.wake_words = wake_words

        self.large_buffer_length = 15
        self.large_buffer_start = time.time()
        self.audio_rate = audio_rate

        self.large_buffer = []
        self.large_silence_for = 0

        self.silence_margin = 0.3  # seconds of silence before processing

        self.transcription_history_short = []
        self.transcription_history = []

        self.transcription_lock = Lock()
        self.on('transcription', self.transcription)
    
    def transcription(self, transcription):
        segments, ti = transcription
        segments = list(segments)
        self.transcription_history.append(segments)

        for word in segments:
            self.emit('transcription_word', word.text)
    
    @property
    def large_buffer_size(self):
        return self.large_buffer_length * self.audio_rate
    
    def feed(self, audio: bytes):
        # Accumulate up to buffer_size * audio_rate samples
        audio = np.frombuffer(audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        average_volume = np.mean(np.abs(audio))
        dB = 20 * np.log10(average_volume) if average_volume > 0 else -np.inf
        if dB < -50:
            self.emit('audio_too_quiet', dB)
            self.large_silence_for += 1 / self.audio_rate * len(audio)
        else:
            if not self.large_buffer:
                self.large_buffer_start = time.time()
            self.large_silence_for = 0
            self.large_buffer.append(audio)
            
        if sum(len(b) for b in self.large_buffer) >= self.large_buffer_size or (time.time() - self.large_buffer_start) > self.large_buffer_length or self.large_silence_for > self.silence_margin:
            # Process the accumulated audio
            self.large_buffer_start = time.time()
            self.process_audio('large')
    
    def _process_audio(self, audio: np.ndarray, type: str = 'small'):
        with self.transcription_lock:
            transcription = self.model.transcribe(audio, beam_size=5, language="en", word_timestamps=True, vad_filter=True)
        
        self.emit('transcription_' + type, transcription)
    
    def process_audio(self, type: str = 'small'):
        # Combine small buffer into a single audio chunk
        if not self.large_buffer:
            return
        audio_stream = np.concatenate(self.large_buffer)
        self.large_buffer = []
        
        self.emit('audio_process_' + type, audio_stream)

        with self.transcription_lock:
            # Convert bytes to numpy array
            thread = Thread(target=self._process_audio, args=(audio_stream, type))
            thread.start()

def process_text(text):
    
    print(f"Processed text: {text}")

if __name__ == '__main__':
    wmodel = WhisperModel("small", device="cpu")
    assistant = Assistant(llm=model, model=wmodel, wake_words=["hey"])

    # Get audio from ffmpeg dshow and feed it to the assistant
    import subprocess
    process = subprocess.Popen(
        ["ffmpeg", "-f", "dshow", "-i", "audio=Microphone (HyperX SoloCast)", "-ac", "1", "-ar", "16000", "-af", "silenceremove=1:0:-50dB", "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
        # stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        # bufsize=10**7  # Set buffer size to 100MB
    )

    assistant.on('wake_word_detected', lambda wake_word: print(f"Wake word detected: {wake_word}"))
    assistant.on('transcription_word', lambda transcription: print(f"Large transcription: {transcription}"))

    assistant.on('audio_process', lambda audio: print(f"Processing large audio chunk of size {len(audio)}"))

    # assistant.on('audio_too_quiet', lambda dB: print(f"Audio too quiet: {dB} dB"))

    try:
        while True:
            audio_chunk = process.stdout.read(1024 * 16)  # Read 16KB chunks
            if not audio_chunk and process.poll() is not None:
                break
            assistant.feed(audio_chunk)
            # time.sleep(0.1)  # Adjust sleep time as needed for your application

    except KeyboardInterrupt:
        print("Stopping audio feed...")
    finally:
        process.terminate()
        process.wait()
        print("Audio feed stopped.")