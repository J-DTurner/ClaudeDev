from camel.agents import BaseAgent

class BaseToolAgent(BaseAgent):
    r"""Creates a :obj:`BaseToolAgent` object with the specified name and
        description.

    Args:
        name (str): The name of the tool agent.
        description (str): The description of the tool agent.
    """

    def __init__(self, name: str, description: str) -> None:

        self.name = name
        self.description = description

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
