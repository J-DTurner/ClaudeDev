from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

from camel.agents import BaseAgent
from camel.configs import ClaudeConfig
from camel.messages import ChatMessage, MessageType, SystemMessage
from camel.model_backend import ModelBackend, ModelFactory
from camel.typing import ModelType, RoleType
from camel.utils import (
    get_model_token_limit,
    num_tokens_from_messages,
)
from chatdev.utils import log_visualize

@dataclass(frozen=True)
class ChatAgentResponse:
    r"""Response of a ChatAgent.

    Attributes:
        msgs (List[ChatMessage]): A list of zero, one or several messages.
            If the list is empty, there is some error in message generation.
            If the list has one message, this is normal mode.
            If the list has several messages, this is the critic mode.
        terminated (bool): A boolean indicating whether the agent decided
            to terminate the chat session.
        info (Dict[str, Any]): Extra information about the chat message.
    """
    msgs: List[ChatMessage]
    terminated: bool
    info: Dict[str, Any]

    @property
    def msg(self):
        if self.terminated:
            raise RuntimeError("error in ChatAgentResponse, info:{}".format(str(self.info)))
        if len(self.msgs) > 1:
            raise RuntimeError("Property msg is only available for a single message in msgs")
        elif len(self.msgs) == 0:
            if len(self.info) > 0:
                raise RuntimeError("Empty msgs in ChatAgentResponse, info:{}".format(str(self.info)))
            else:
                # raise RuntimeError("Known issue that msgs is empty and there is no error info, to be fix")
                return None
        return self.msgs[0]

class ChatAgent(BaseAgent):
    def __init__(
            self,
            system_message: SystemMessage,
            memory=None,
            model: Optional[ModelType] = None,
            model_config: Optional[Any] = None,
            message_window_size: Optional[int] = None,
    ) -> None:
        self.system_message: SystemMessage = system_message
        self.role_name: str = system_message.role_name
        self.role_type: RoleType = system_message.role_type
        self.model: ModelType = model if model is not None else ModelType.CLAUDE_3_OPUS
        self.model_config: ClaudeConfig = model_config or ClaudeConfig()
        self.model_token_limit: int = get_model_token_limit(self.model)
        self.message_window_size: Optional[int] = message_window_size
        self.model_backend: ModelBackend = ModelFactory.create(self.model, self.model_config.__dict__)
        self.terminated: bool = False
        self.info: bool = False
        self.init_messages()
        if memory is not None and self.role_name in ["Code Reviewer", "Programmer", "Software Test Engineer"]:
            self.memory = memory.memory_data.get("All")
        else:
            self.memory = None

    def reset(self) -> List[MessageType]:
        r"""Resets the :obj:`ChatAgent` to its initial state and returns the
        stored messages.

        Returns:
            List[MessageType]: The stored messages.
        """
        self.terminated = False
        self.init_messages()
        return self.stored_messages

    def get_info(
            self,
            id: Optional[str],
            usage: Optional[Dict[str, int]],
            termination_reasons: List[str],
            num_tokens: int,
    ) -> Dict[str, Any]:
        r"""Returns a dictionary containing information about the chat session.

        Args:
            id (str, optional): The ID of the chat session.
            usage (Dict[str, int], optional): Information about the usage of
                the LLM model.
            termination_reasons (List[str]): The reasons for the termination of
                the chat session.
            num_tokens (int): The number of tokens used in the chat session.

        Returns:
            Dict[str, Any]: The chat session information.
        """
        return {
            "id": id,
            "usage": usage,
            "termination_reasons": termination_reasons,
            "num_tokens": num_tokens,
        }

    def init_messages(self) -> None:
        r"""Initializes the stored messages list with the initial system
        message.
        """
        self.stored_messages: List[MessageType] = [self.system_message]

    def update_messages(self, message: ChatMessage) -> List[MessageType]:
        r"""Updates the stored messages list with a new message.

        Args:
            message (ChatMessage): The new message to add to the stored
                messages.

        Returns:
            List[ChatMessage]: The updated stored messages.
        """
        self.stored_messages.append(message)
        return self.stored_messages
    
    def use_memory(self,input_message) -> List[MessageType]:
        if self.memory is None :
            return None
        else:
            if self.role_name == "Programmer":
                result = self.memory.memory_retrieval(input_message,"code")
                if result != None:
                    target_memory,distances, mids,task_list,task_dir_list = result
                    if target_memory != None and len(target_memory) != 0:
                        target_memory="".join(target_memory)
                        #self.stored_messages[-1].content = self.stored_messages[-1].content+"Here is some code you've previously completed:"+target_memory+"You can refer to the previous script to complement this task."
                        log_visualize(self.role_name,
                                            "thinking back and found some related code: \n--------------------------\n"
                                            + target_memory)
                else:
                    target_memory = None
                    log_visualize(self.role_name,
                                         "thinking back but find nothing useful")

            else:
                result = self.memory.memory_retrieval(input_message, "text")
                if result != None:
                    target_memory, distances, mids, task_list, task_dir_list = result
                    if target_memory != None and len(target_memory) != 0:
                        target_memory=";".join(target_memory)
                        #self.stored_messages[-1].content = self.stored_messages[-1].content+"Here are some effective and efficient instructions you have sent to the assistant :"+target_memory+"You can refer to these previous excellent instructions to better instruct assistant here."
                        log_visualize(self.role_name,
                                            "thinking back and found some related text: \n--------------------------\n"
                                            + target_memory)
                else:
                    target_memory = None
                    log_visualize(self.role_name,
                                         "thinking back but find nothing useful")

        return target_memory

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def step(
            self,
            input_message: ChatMessage,
    ) -> ChatAgentResponse:
        messages = self.update_messages(input_message)
        if self.message_window_size is not None and len(messages) > self.message_window_size:
            messages = [self.system_message] + messages[-self.message_window_size:]
        
        claude_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        num_tokens = num_tokens_from_messages(claude_messages, self.model)

        output_messages: Optional[List[ChatMessage]]
        info: Dict[str, Any]

        if num_tokens < self.model_token_limit:
            response = self.model_backend.run(
                model=self.model.value,
                messages=claude_messages,
                max_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                n=self.model_config.n,
                stream=self.model_config.stream,
                stop=self.model_config.stop,
            )

            output_messages = [
                ChatMessage(
                    role_name=self.role_name,
                    role_type=self.role_type,
                    meta_dict=dict(),
                    role=response["role"],
                    content="".join([content_block["text"] for content_block in response["content"]]),
                )
            ]
            info = self.get_info(
                response["id"],
                response["usage"],
                [response["stop_reason"]],
                num_tokens,
            )

            if output_messages[0].content.split("\n")[-1].startswith("<INFO>"):
                self.info = True
        else:
            self.terminated = True
            output_messages = []
            info = self.get_info(
                None,
                None,
                ["max_tokens_exceeded_by_camel"],
                num_tokens,
            )

        return ChatAgentResponse(output_messages, self.terminated, info)

    def __repr__(self) -> str:
        r"""Returns a string representation of the :obj:`ChatAgent`.

        Returns:
            str: The string representation of the :obj:`ChatAgent`.
        """
        return f"ChatAgent({self.role_name}, {self.role_type}, {self.model})"
