import time

from io import BytesIO
from faster_whisper import WhisperModel

from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from threading import Thread, Lock

import numpy as np

from assistant.transcriber import Transcriber

model = ChatOllama(model="llama3.1")

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