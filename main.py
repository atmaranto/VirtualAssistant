import time

from langchain_core.runnables import RunnableLambda
from langchain_community.tools import tool

from io import BytesIO
from faster_whisper import WhisperModel

from threading import Thread, Lock

from langchain_ollama import ChatOllama
import numpy as np

from assistant import Assistant
from assistant.assistant import create_basic_llm

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Run the assistant with audio input.")
    
    ap.add_argument("--model", type=str, default="small", help="Whisper model size (e.g., small, medium, large)")
    ap.add_argument("--mic-name", type=str, default=None, help="Name of the microphone to use (if applicable)")

    args = ap.parse_args()

    import subprocess
    import platform
    import re
    import datetime
    import colorama

    wmodel = WhisperModel(args.model, device="cpu")

    orig_model = ChatOllama(model="qwen3:8b", extract_reasoning=True)

    @tool
    def get_current_time() -> str:
        """Get the current time in HH:MM:SS format."""
        return datetime.datetime.now().strftime("%H:%M:%S")

    def tool_call(tool_call, tool_responses):
        """Handle tool calls from the assistant."""
        if tool_call["name"] == "get_current_time":
            response = get_current_time.invoke(tool_call["args"])
            tool_responses.append(response)
            print(f"Tool call response: {response}")
        else:
            print(f"Unknown tool call: {tool_call.name}")
    
    orig_model = orig_model.bind_tools([get_current_time])

    model, chat_history = create_basic_llm(orig_model)

    model = RunnableLambda(lambda x: dict(
        system_message="You are a helpful agent named Frame that runs on a set of advanced smart glasses. You will respond in a cheeky but accurate way to user queries based on the provided context and history.",
        optional_user_prompt=[],
        **x)
    ) | model

    assistant = Assistant(llm=model, model=wmodel, wake_words=["hey"], configuration={"session_id": "test_session"})

    # Get audio from ffmpeg dshow and feed it to the assistant

    if platform.system() == "Windows":
        mic_name = args.mic_name
        if not mic_name:
            # Default to the first available microphone
            devices = subprocess.Popen(["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], stderr=subprocess.STDOUT, text=True, stdout=subprocess.PIPE).communicate()[0]

            mic_name = re.search(r'\"(.*?)\" \(audio\)', devices).group(1)
            print(f"Using microphone: {mic_name}")

        process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-f", "dshow", "-i", f"audio={mic_name}", "-ac", "1", "-ar", "16000", "-af", "silenceremove=1:0:-50dB", "-f", "s16le", "-"],
            stdout=subprocess.PIPE,
            # stderr=subprocess.DEVNULL,
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
    assistant.on('transcription_word', lambda transcription: print(f"Transcription: {colorama.Fore.RED}{transcription}{colorama.Style.RESET_ALL}"))
    assistant.on("assistant_speak_word", lambda text: print(f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}", end="", flush=True))
    assistant.on('tool', tool_call)

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