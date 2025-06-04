import json
from json import JSONDecodeError

from rich.pretty import pprint
from termcolor import cprint
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get a logger instance
logger = logging.getLogger(__name__)
def step_logger(steps):
    """
    log the steps of an agent's response in a formatted way.
    Note: stream need to be set to False to use this function.
    Args:
    steps: List of steps from an agent's response.
    """
    for i, step in enumerate(steps):
        step_type = type(step).__name__
        logger.info(f"Step {i+1}: {step_type}")
        if step_type == "ToolExecutionStep":
            logger.info("Executing tool...")
            try:
                pprint(json.loads(step.tool_responses[0].content))
            except (TypeError, JSONDecodeError):
                # tool response is not a valid JSON object
                pprint(step.tool_responses[0].content)
        else:
            if step.api_model_response.content:
                logger.info("Model Response:")
                logger.info(step.api_model_response.content)
            elif step.api_model_response.tool_calls:
                tool_call = step.api_model_response.tool_calls[0]
                logger.info("Tool call Generated:")
                logger.info(f"Tool call: {tool_call.tool_name}, Arguments: {json.loads(tool_call.arguments_json)}")
    logger.info("Query processing completed")