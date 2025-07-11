import time
from threading import Thread, Lock
import numpy as np
from .eventable import Eventable

class Transcriber(Eventable):
    def __init__(self, model, audio_rate: int = 16000, wake_words=[], large_buffer_length: int = 15):
        super().__init__()
        self.model = model
        self.wake_words = wake_words

        self.large_buffer_length = large_buffer_length
        self.large_buffer_start = time.time()
        self.audio_rate = audio_rate

        self.large_buffer = []
        self.large_silence_for = 0

        self.silence_margin = 0.3  # seconds of silence before processing

        self.transcription_history_short = []
        self.transcription_history = []

        self.transcription_lock = Lock()
        self.on('transcription_raw', self.transcription_raw)
    
    def transcription_raw(self, segments, ti):
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
            self.process_audio()
    
    def _process_audio(self, audio: np.ndarray, type: str = 'small'):
        with self.transcription_lock:
            segs, ti = self.model.transcribe(audio, beam_size=5, language="en", word_timestamps=True, vad_filter=True)
        
        segs = list(segs)
        self.emit('transcription_raw', segs, ti)
    
    def process_audio(self):
        # Combine small buffer into a single audio chunk
        if not self.large_buffer:
            return
        audio_stream = np.concatenate(self.large_buffer)
        self.large_buffer = []
        
        self.emit('audio_process', audio_stream)

        with self.transcription_lock:
            # Convert bytes to numpy array
            thread = Thread(target=self._process_audio, args=(audio_stream, type))
            thread.start()
