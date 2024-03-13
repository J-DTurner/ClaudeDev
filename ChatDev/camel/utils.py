import os, re, zipfile, requests, tiktoken, time
from functools import wraps
from typing import Any, Callable, List, Optional, Set, TypeVar
from camel.typing import ModelType, TaskType

F = TypeVar('F', bound=Callable[..., Any])

def count_tokens_claude_chat_models(
        messages: List[dict],
        encoding: Any,
) -> int:
    r"""Counts the number of tokens required to generate a Claude chat based
    on a given list of messages.

    Args:
        messages (List[dict]): The list of messages.
        encoding (Any): The encoding method to use.

    Returns:
        int: The number of tokens required.
    """
    num_tokens = 0
    for message in messages:
        num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def num_tokens_from_messages(
        messages: List[dict],
        model: ModelType,
) -> int:
    r"""Returns the number of tokens used by a list of messages.

    Args:
        messages (List[dict]): The list of messages to count the
            number of tokens for.
        model (ModelType): The Claude model used to encode the messages.

    Returns:
        int: The total number of tokens used by the messages.

    Raises:
        NotImplementedError: If the specified `model` is not implemented.
    """
    try:
        value_for_tiktoken = model.value_for_tiktoken
        encoding = tiktoken.encoding_for_model(value_for_tiktoken)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {ModelType.CLAUDE_2_1, ModelType.CLAUDE_3_OPUS, ModelType.STUB}:
        return count_tokens_claude_chat_models(messages, encoding)
    else:
        raise NotImplementedError(
            f"`num_tokens_from_messages` is not presently implemented "
            f"for model {model}. "
        )


def get_model_token_limit(model: ModelType) -> int:
    r"""Returns the maximum token limit for a given model.

    Args:
        model (ModelType): The type of the model.

    Returns:
        int: The maximum token limit for the given model.
    """
    if model == ModelType.CLAUDE_2_1:
        return 80000
    elif model == ModelType.CLAUDE_3_OPUS:
        return 150000
    elif model == ModelType.STUB:
        return 150000
    else:
        raise ValueError("Unknown model type")


def openai_api_key_required(func: F) -> F:
    r"""Decorator that checks if the OpenAI API key is available in the
    environment variables.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The decorated function.

    Raises:
        ValueError: If the OpenAI API key is not found in the environment
            variables.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from camel.agents.chat_agent import ChatAgent
        if not isinstance(self, ChatAgent):
            raise ValueError("Expected ChatAgent")
        if self.model == ModelType.STUB:
            return func(self, *args, **kwargs)
        elif 'OPENAI_API_KEY' in os.environ:
            return func(self, *args, **kwargs)
        else:
            raise ValueError('OpenAI API key not found.')

    return wrapper


def print_text_animated(text, delay: float = 0.005, end: str = ""):
    r"""Prints the given text with an animated effect.

    Args:
        text (str): The text to print.
        delay (float, optional): The delay between each character printed.
            (default: :obj:`0.02`)
        end (str, optional): The end character to print after the text.
            (default: :obj:`""`)
    """
    for char in text:
        print(char, end=end, flush=True)
        time.sleep(delay)
    print('\n')


def get_prompt_template_key_words(template: str) -> Set[str]:
    r"""Given a string template containing curly braces {}, return a set of
    the words inside the braces.

    Args:
        template (str): A string containing curly braces.

    Returns:
        List[str]: A list of the words inside the curly braces.

    Example:
        >>> get_prompt_template_key_words('Hi, {name}! How are you {status}?')
        {'name', 'status'}
    """
    return set(re.findall(r'{([^}]*)}', template))


def get_first_int(string: str) -> Optional[int]:
    r"""Returns the first integer number found in the given string.

    If no integer number is found, returns None.

    Args:
        string (str): The input string.

    Returns:
        int or None: The first integer number found in the string, or None if
            no integer number is found.
    """
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    else:
        return None


def download_tasks(task: TaskType, folder_path: str) -> None:
    # Define the path to save the zip file
    zip_file_path = os.path.join(folder_path, "tasks.zip")

    # Download the zip file from the Google Drive link
    response = requests.get("https://huggingface.co/datasets/camel-ai/"
                            f"metadata/resolve/main/{task.value}_tasks.zip")

    # Save the zip file
    with open(zip_file_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)

    # Delete the zip file
    os.remove(zip_file_path)
