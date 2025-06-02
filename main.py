import time

from langchain_core.runnables import RunnableLambda
from langchain_community.tools import BraveSearch

from io import BytesIO
from faster_whisper import WhisperModel

from threading import Thread, Lock

import numpy as np

from assistant import Assistant
from assistant.assistant import create_basic_llm

if __name__ == '__main__':
    wmodel = WhisperModel("small", device="cpu")
    model, chat_history = create_basic_llm()

    model = RunnableLambda(lambda x: dict(
        system_message="You are a helpful agent named Frame that runs on a set of advanced smart glasses. You will respond in a cheeky but accurate way to user queries based on the provided context and history.",
        optional_user_prompt=[],
        **x)
    ) | model

    assistant = Assistant(llm=model, model=wmodel, wake_words=["hey"], configuration={"session_id": "test_session"})

    # Get audio from ffmpeg dshow and feed it to the assistant
    import subprocess

    import platform
    if platform.system() == "Windows":
        process = subprocess.Popen(
            ["ffmpeg", "-f", "dshow", "-i", "audio=Microphone (HyperX SoloCast)", "-ac", "1", "-ar", "16000", "-af", "silenceremove=1:0:-50dB", "-f", "s16le", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            # bufsize=10**7  # Set buffer size to 100MB
        )
    else:
        process = subprocess.Popen(
            ["arecord", "-f", "S16_LE", "-c", "1", "-r", "16000"],
            stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            # bufsize=10**7  # Set buffer size to 100MB
        )

    assistant.on('wake_word_detected', lambda wake_word, transcription: print(f"Wake word detected: {wake_word}"))
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