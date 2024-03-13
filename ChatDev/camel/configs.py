from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

@dataclass(frozen=True)
class ClaudeConfig:
    r"""Defines the parameters for generating chat completions using the Claude API.

    Args:
        temperature (float, optional): Sampling temperature to use, between
            :obj:`0` and :obj:`1`. Higher values make the output more random,
            while lower values make it more focused and deterministic.
            (default: :obj:`1.0`)
        top_p (float, optional): An alternative to sampling with temperature,
            called nucleus sampling, where the model considers the results of
            the tokens with top_p probability mass. So :obj:`0.1` means only
            the tokens comprising the top 10% probability mass are considered.
            (default: :obj:`-1.0`)
        n (int, optional): How many chat completion choices to generate for
            each input message. (default: :obj:`1`)
        stream (bool, optional): If True, partial message deltas will be sent
            as data-only server-sent events as they become available.
            (default: :obj:`False`)
        stop (str or list, optional): Up to :obj:`4` sequences where the API
            will stop generating further tokens. (default: :obj:`None`)
        max_tokens (int, optional): The maximum number of tokens to generate
            in the chat completion. The total length of input tokens and
            generated tokens is limited by the model's context length.
            (default: :obj:`None`)
        model (str, optional): The model to use for generating the chat
            completion. (default: :obj:`None`)
    """
    temperature: float = 1.0
    top_p: float = -1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, Sequence[str]]] = None
    max_tokens: Optional[int] = None
    model: Optional[str] = None