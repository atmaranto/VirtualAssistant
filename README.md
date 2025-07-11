# Description
VirtualAssistant (or `virtual-assistant`) is an extremely simple Python module that allows one to create their own voice assistants, using the transcription ability of `faster-whisper`, the messaging ability of a configurable LLM, and the tooling ability of `langchain`.

## Caveat

This is not meant for large projects. I primarily wrote it to use in my [frame-assist](https://github.com/atmaranto/frame-assist) package.

## Compatbility

Tested on Debian Linux and Windows 10.

## Example

[main.py](main.py) contains a fairly simple example of an Ollama-powered large language model chatbot that answers user questions.

#### Usage

Create a virtual environment, if you'd like. Then run:
```bash
python3 -m pip install -r requirements.txt
python3 main.py
```
