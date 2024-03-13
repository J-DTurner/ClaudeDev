from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Sequence
import requests, os
from camel.typing import ModelType

CLAUDE_API_KEY = os.environ['CLAUDE_API_KEY']
if 'CLAUDE_BASE_URL' in os.environ:
    CLAUDE_BASE_URL = os.environ['CLAUDE_BASE_URL']
else:
    CLAUDE_BASE_URL = "https://api.anthropic.com"


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be Claude API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = -1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        r"""Runs the query to the backend model.

        Args:
            model (str): The model to use for generating the response.
            messages (List[Dict[str, str]]): The list of input messages.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float, optional): The temperature to use for sampling.
            top_p (float, optional): The top-p value to use for sampling.
            n (int, optional): The number of responses to generate.
            stream (bool, optional): Whether to stream the response.
            stop (Optional[Union[str, Sequence[str]]], optional): The stop sequence(s).
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The response from the backend model.
        """
        pass

class ClaudeModel(ModelBackend):
    r"""Claude API in a unified ModelBackend interface."""
    def __init__(self, model_type: ModelType, model_config_dict: Dict, api_key: str) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict
        self.api_key = 'INSERT YOUR CLAUDE API KEY HERE'

    def run(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = -1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        if messages[0]["role"] != "user":
            messages[0]["role"] = "user"

        system_message = None
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
                messages.remove(message)
                break

        if len(messages) == 1 and messages[0]["role"] == "user":
            claude_messages = [
                {"role": "user", "content": messages[0]["content"]},
                {"role": "assistant", "content": "Assistant"}
            ]
        else:
            claude_messages = []
            last_role = None
            for message in messages:
                if message["role"] == last_role:
                    claude_messages[-1]["content"] += "\n" + message["content"]
                else:
                    claude_messages.append({"role": message["role"], "content": message["content"]})
                    last_role = message["role"]

        payload = {
            "model": model,
            "messages": claude_messages,
            "max_tokens": 4096,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop
        }

        if system_message:
            payload["system"] = system_message

        response = requests.post(
            f"https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
        print(f"Response: {response.text}")

        response.raise_for_status()
        response_json = response.json()

        # Parse Claude's response JSON into the format expected by CAMEL
        camel_response = {
            "id": response_json["id"],
            "type": response_json["type"],
            "role": response_json["role"],
            "content": response_json["content"],
            "model": response_json["model"],
            "stop_reason": response_json["stop_reason"],
            "stop_sequence": response_json["stop_sequence"],
            "usage": response_json["usage"],
        }

        return camel_response

class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            role="assistant",
            content=[dict(type="text", text=ARBITRARY_STRING)],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.CLAUDE_2_1

        if model_type in {ModelType.CLAUDE_2_1, ModelType.CLAUDE_3_OPUS, None}:
            model_class = ClaudeModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError("Unknown model")

        if model_type is None:
            model_type = default_model_type

        api_key = "PUT YOUR CLAUDE API KEY HERE"
        inst = model_class(model_type, model_config_dict, api_key)
        return inst