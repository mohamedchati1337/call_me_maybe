import argparse
from typing import Any, Dict
from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    """Validation schema for functional metadata definitions.

    Attributes:
        name: The unique string identifier of the function.
        description: A text explanation of what the function does.
        parameters: Structural schema definition of input arguments.
        returns: Structural schema definition of the output block.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class TestPrompt(BaseModel):
    """Validation schema for evaluation benchmark prompt cases.

    Attributes:
        prompt: The raw user query text string to test.
    """

    prompt: str


def args() -> argparse.Namespace:
    """Parse and retrieve command-line configurations for the pipeline.

    Defines arguments for configuring input test paths, functional
    definition schemas, and final structured logging outputs.

    Returns:
        An argparse Namespace object populated with file paths.
    """
    arguments = argparse.ArgumentParser()

    arguments.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json"
    )

    arguments.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json"
    )

    arguments.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json"
    )

    return arguments.parse_args()
