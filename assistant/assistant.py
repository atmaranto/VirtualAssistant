from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

from .transcriber import Transcriber

from typing import Optional

from collections import defaultdict

import re

class Assistant(Transcriber):
    def __init__(self, llm: BaseChatModel, model, wake_words=[], vector_store: Optional[VectorStore] = None, configuration: Optional[dict] = None):
        super().__init__(model=model, wake_words=wake_words)
        self.llm = llm
        self.vector_store = vector_store
        self.configuration = configuration if configuration is not None else {}

        self.on('transcription_raw', self.process_transcription)
        self.on('wake_word_detected', self.handle_wake_word)
    
    def process_transcription(self, segments, ti):
        for wake_word in self.wake_words:
            for segment in segments:
                if wake_word.lower() in re.sub(r"[^\w\s]", "", segment.text.lower()):
                    self.emit('wake_word_detected', wake_word, segment.text)
                    return
    
    def handle_wake_word(self, wake_word, segment_text):
        response = self.llm.stream({"input": ("human", segment_text)}, self.configuration)
        while True:
            total = ""
            tool_responses = []
            for message in response:
                if getattr(message, 'tool_calls', None):
                    for tool_call in message.tool_calls:
                        self.emit('tool', tool_call, tool_responses)
                # print(message.content, end="", flush=True)
                if message.content:
                    self.emit('assistant_speak_word', message.content)
                total += message.content
            # print()
            self.emit('assistant_speak', total)
            if not tool_responses:
                break
            response = self.llm.stream({"input": ("tool", "\n".join(tool_responses))}, self.configuration)

def create_basic_llm(model):
    template = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder(variable_name="optional_user_prompt"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Initialize the chat history
    chat_history = defaultdict(InMemoryChatMessageHistory)

    model = RunnableWithMessageHistory(template | model, chat_history.__getitem__, input_messages_key="input", history_messages_key="history")

    return model, chat_history
