from typing import Dict, Union

OpenAISystemMessage = Dict[str, str]
OpenAIAssistantMessage = Dict[str, str]
OpenAIUserMessage = Dict[str, str]
OpenAIChatMessage = Union[OpenAIUserMessage, OpenAIAssistantMessage]
OpenAIMessage = Union[OpenAISystemMessage, OpenAIChatMessage]

from .base import BaseMessage  # noqa: E402
from .system_messages import (  # noqa: E402
    SystemMessage, AssistantSystemMessage, UserSystemMessage,
)
from .chat_messages import (  # noqa: E402
    ChatMessage, AssistantChatMessage, UserChatMessage,
)

MessageType = Union[BaseMessage, SystemMessage, AssistantSystemMessage,
                    UserSystemMessage, ChatMessage, AssistantChatMessage,
                    UserChatMessage]
SystemMessageType = Union[SystemMessage, AssistantSystemMessage,
                          UserSystemMessage]
ChatMessageType = Union[ChatMessage, AssistantChatMessage, UserChatMessage]

__all__ = [
    'OpenAISystemMessage',
    'OpenAIAssistantMessage',
    'OpenAIUserMessage',
    'OpenAIChatMessage',
    'OpenAIMessage',
    'BaseMessage',
    'SystemMessage',
    'AssistantSystemMessage',
    'UserSystemMessage',
    'ChatMessage',
    'AssistantChatMessage',
    'UserChatMessage',
    'MessageType',
    'SystemMessageType',
    'ChatMessageType',
]
