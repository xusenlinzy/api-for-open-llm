from typing import Dict, Any, List

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from .registry import BaseParent


class ChatGLMConversationBufferWindowMemory(ConversationBufferWindowMemory):

    human_prefix: str = "问"
    ai_prefix: str = "答"

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""

        if self.return_messages:
            buffer: Any = self.buffer[-self.k * 2:]
        else:
            buffer = self.get_buffer_string(
                self.buffer[-self.k * 2:],
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: buffer}

    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "问", ai_prefix: str = "答"
    ) -> str:
        """Get buffer string of messages."""
        string_messages, i = [], 0
        for m in messages:
            if isinstance(m, HumanMessage):
                role = human_prefix
                string_messages.append(f"[Round {i}]\n{role}：{m.content}")
                i += 1
            elif isinstance(m, AIMessage):
                role = ai_prefix
                string_messages.append(f"{role}：{m.content}")
            else:
                raise ValueError(f"Got unsupported message type: {m}")

        return "\n".join(string_messages) + f"\n[Round {i}]"


class ChineseAlpacaConversationBufferWindowMemory(ChatGLMConversationBufferWindowMemory):

    human_prefix: str = "### Instruction"
    ai_prefix: str = "### Response"

    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "### Instruction", ai_prefix: str = "### Response"
    ) -> str:
        """Get buffer string of messages."""
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = human_prefix
                string_messages.append(f"{role}:\n\n{m.content}")
            elif isinstance(m, AIMessage):
                role = ai_prefix
                string_messages.append(f"{role}:\n\n{m.content}")
            else:
                raise ValueError(f"Got unsupported message type: {m}")

        return "\n\n".join(string_messages)


class FireFlyConversationBufferWindowMemory(ChatGLMConversationBufferWindowMemory):

    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"

    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "Assistant"
    ) -> str:
        """Get buffer string of messages."""
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                string_messages.append(f"<s>{m.content}</s>")
            elif isinstance(m, AIMessage):
                string_messages.append(f"</s>{m.content}</s>")
            else:
                raise ValueError(f"Got unsupported message type: {m}")

        return "".join(string_messages)


class PhoenixConversationBufferWindowMemory(ChatGLMConversationBufferWindowMemory):

    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"

    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "Assistant"
    ) -> str:
        """Get buffer string of messages."""
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = human_prefix
            elif isinstance(m, AIMessage):
                role = ai_prefix
            else:
                raise ValueError(f"Got unsupported message type: {m}")
            string_messages.append(f"{role}: <s>{m.content}</s>")
        return "".join(string_messages)


class MossConversationBufferWindowMemory(ChatGLMConversationBufferWindowMemory):

    human_prefix: str = "<|Human|>"
    ai_prefix: str = "<|MOSS|>"

    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "<|Human|>", ai_prefix: str = "<|MOSS|>"
    ) -> str:
        """Get buffer string of messages."""
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                string_messages.append(f"{human_prefix}: {m.content}<eoh>")
            elif isinstance(m, AIMessage):
                string_messages.append(f"{ai_prefix}: {m.content}<eom>")
            else:
                raise ValueError(f"Got unsupported message type: {m}")
        return "\n".join(string_messages)


class GuanacoConversationBufferWindowMemory(ChatGLMConversationBufferWindowMemory):

    human_prefix: str = "### Human"
    ai_prefix: str = "### Assistant"

    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "### Human", ai_prefix: str = "### Assistant"
    ) -> str:
        """Get buffer string of messages."""
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                string_messages.append(f"{human_prefix}: {m.content}")
            elif isinstance(m, AIMessage):
                string_messages.append(f"{ai_prefix}: {m.content}")
            else:
                raise ValueError(f"Got unsupported message type: {m}")
        return "\n".join(string_messages)


class CustomConversationBufferWindowMemory(BaseParent):

    registry = {}


CustomConversationBufferWindowMemory.add_to_registry("gpt-3.5-turbo", ConversationBufferWindowMemory)
CustomConversationBufferWindowMemory.add_to_registry("chatglm", ChatGLMConversationBufferWindowMemory)
CustomConversationBufferWindowMemory.add_to_registry("chinese-llama-alpaca", ChineseAlpacaConversationBufferWindowMemory)
CustomConversationBufferWindowMemory.add_to_registry("firefly", FireFlyConversationBufferWindowMemory)
CustomConversationBufferWindowMemory.add_to_registry("phoenix", PhoenixConversationBufferWindowMemory)
CustomConversationBufferWindowMemory.add_to_registry("moss", MossConversationBufferWindowMemory)
CustomConversationBufferWindowMemory.add_to_registry("guanaco", GuanacoConversationBufferWindowMemory)
