from dataclasses import dataclass
from typing import Dict, Optional

from camel.messages import BaseMessage
from camel.typing import RoleType

@dataclass
class SystemMessage(BaseMessage):
    r"""Class for system messages used in CAMEL chat system.

    Args:
        role_name (str): The name of the user or assistant role.
        role_type (RoleType): The type of role, either
            :obj:`RoleType.ASSISTANT` or :obj:`RoleType.USER`.
        meta_dict (Optional[Dict[str, str]]): Additional metadata dictionary
            for the message.
        role (str): The role of the message in OpenAI chat system.
            (default: :obj:`"system"`)
        content (str): The content of the message. (default: :obj:`""`)
    """
    role_name: str
    role_type: RoleType
    meta_dict: Optional[Dict[str, str]] = None
    role: str = "system"
    content: str = ""


@dataclass
class AssistantSystemMessage(SystemMessage):
    r"""Class for system messages from the assistant used in the CAMEL chat
    system.

    Args:
        role_name (str): The name of the assistant role.
        role_type (RoleType): The type of role, always
            :obj:`RoleType.ASSISTANT`.
        meta_dict (Optional[Dict[str, str]]): Additional metadata dictionary
            for the message.
        role (str): The role of the message in OpenAI chat system.
            (default: :obj:`"system"`)
        content (str): The content of the message. (default: :obj:`""`)
    """
    role_name: str
    role_type: RoleType = RoleType.ASSISTANT
    meta_dict: Optional[Dict[str, str]] = None
    role: str = "system"
    content: str = ""


@dataclass
class UserSystemMessage(SystemMessage):
    r"""Class for system messages from the user used in the CAMEL chat system.

    Args:
        role_name (str): The name of the user role.
        role_type (RoleType): The type of role, always :obj:`RoleType.USER`.
        meta_dict (Optional[Dict[str, str]]): Additional metadata dictionary
            for the message.
        role (str): The role of the message in OpenAI chat system.
            (default: :obj:`"system"`)
        content (str): The content of the message. (default: :obj:`""`)
    """
    role_name: str
    role_type: RoleType = RoleType.USER
    meta_dict: Optional[Dict[str, str]] = None
    role: str = "system"
    content: str = ""
