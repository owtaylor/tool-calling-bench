# llm_backends/anthropic_backend.py
import anthropic
import json
import logging
from typing import Tuple, Optional, List, Dict, Any
from ..secrets_manager import Secrets  # <-- Import Secrets for type hinting

logger = logging.getLogger(__name__)


# --- Add this helper function ---
def format_tools_for_anthropic(
    tools_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Converts OpenAI-style tool definitions to Anthropic format."""
    anthropic_tools = []
    for tool in tools_data:
        # Check if it looks like the OpenAI format we expect
        if tool.get("type") == "function" and "function" in tool:
            func_details = tool["function"]
            if (
                isinstance(func_details, dict)
                and "name" in func_details
                and "parameters" in func_details
            ):
                # Map OpenAI fields to Anthropic fields
                anthropic_tools.append(
                    {
                        "name": func_details["name"],
                        "description": func_details.get(
                            "description", ""
                        ),  # Handle optional description
                        "input_schema": func_details[
                            "parameters"
                        ],  # Rename 'parameters' to 'input_schema'
                    }
                )
            else:
                logger.warning(
                    f"Skipping tool due to missing 'name' or 'parameters' in function details: {tool}"
                )
        else:
            # If it's already in Anthropic format or some other unknown format, log a warning.
            # For now, we assume the input is always OpenAI format needing conversion.
            # If it might already be correct, add a check here.
            # Example check: if "name" in tool and "input_schema" in tool: anthropic_tools.append(tool)
            logger.warning(f"Skipping tool with unexpected format for Anthropic conversion: {tool}")
    return anthropic_tools


# --- End of helper function ---


def convert_to_anthropic_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts messages from Ollama format to Anthropic format."""
    anthropic_messages = []
    for message in messages:
        if message["role"] == "tool":
            # Convert tool result to "tool_result" block
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_use_id"],
                            "content": message["content"],
                        }
                    ],
                }
            )
        elif message["role"] == "assistant" and "tool_calls" in message:
            # Convert tool call requests to "tool_use" block
            tool_call = message["tool_calls"][0]
            anthropic_messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": tool_call["function"]["arguments"],
                        }
                    ],
                }
            )
        else:
            # Keep other messages unchanged
            anthropic_messages.append(message)
    return anthropic_messages


def invoke_anthropic(
    model: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],  # This is still the original format from tools.json
    temperature: float,
    secrets: Secrets,
    max_tokens: int = 1024,
    **kwargs,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Invokes the Anthropic API (Claude) with tool calling support.
    Args:
        ... (docstring args remain the same, 'tools' refers to input format) ...
    """
    api_key = secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in secrets file or environment variables.")
        return None, "Error: ANTHROPIC_API_KEY not configured."

    # --- Add this line to format the tools ---
    formatted_tools = format_tools_for_anthropic(tools)
    # ---

    # Convert messages to Anthropic format
    anthropic_messages = convert_to_anthropic_format(messages)

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # --- Use formatted_tools in the log and API call ---
        logger.debug(
            f"Anthropic Request: model={model}, system='{system_prompt[:50]}...', messages={anthropic_messages}, tools={formatted_tools}, temperature={temperature}, max_tokens={max_tokens}"
        )

        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=anthropic_messages,
            tools=formatted_tools,  # <-- Pass the correctly formatted tools
            tool_choice={"type": "auto"},
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        # ---

        logger.debug(f"Anthropic Raw Response: {response}")

        called_tool_name = None
        response_content = None

        # Check response content and tool calls
        if response.content:
            text_block = next((block for block in response.content if block.type == "text"), None)
            if text_block:
                response_content = text_block.text

            tool_use_block = next(
                (block for block in response.content if block.type == "tool_use"), None
            )
            if tool_use_block:
                called_tool_name = tool_use_block.name
                logger.info(f"Anthropic model requested tool call: {called_tool_name}")

        # Handle stop reasons (no changes needed here)
        if response.stop_reason == "tool_use":
            if not called_tool_name:
                logger.warning(
                    "Anthropic response stop_reason is 'tool_use' but no tool_use block found."
                )
            pass
        elif response.stop_reason in ["end_turn", "max_tokens"]:
            if called_tool_name:
                logger.warning(
                    f"Anthropic response stopped due to {response.stop_reason} but also contained a tool call request ({called_tool_name}). Prioritizing tool call."
                )
            elif not response_content:
                logger.warning(
                    f"Anthropic response stopped due to {response.stop_reason} but no text content found."
                )
                response_content = (
                    f"Warning: Response stopped due to {response.stop_reason} with no text content."
                )
        else:
            logger.warning(
                f"Anthropic response stopped due to unexpected reason: {response.stop_reason}"
            )
            if not response_content and not called_tool_name:
                response_content = (
                    f"Warning: Response stopped unexpectedly ({response.stop_reason})."
                )

        # Prioritize returning the tool name if called (no changes needed here)
        if called_tool_name:
            return called_tool_name, response_content
        else:
            return None, response_content

    except anthropic.APIConnectionError as e:
        logger.error(f"Anthropic API connection error: {e}")
        return None, f"Error: API Connection Error - {e.__cause__}"
    except anthropic.RateLimitError as e:
        logger.error(f"Anthropic API rate limit exceeded: {e}")
        return None, "Error: Rate Limit Exceeded"
    except anthropic.APIStatusError as e:
        # Log the actual error message from Anthropic for better debugging
        error_details = "Unknown error structure"
        try:
            # Attempt to parse the response body if it's JSON
            error_details = e.response.json()
        except (json.JSONDecodeError, AttributeError):
            # Fallback if parsing fails or response has no json() method
            error_details = (
                str(e.response.content) if hasattr(e.response, "content") else str(e.response)
            )

        logger.error(
            f"Anthropic API status error: {e.status_code} - {error_details}"
        )  # Log parsed details
        # Return a more informative error message
        error_message = (
            error_details.get("error", {}).get("message", "Unknown API error")
            if isinstance(error_details, dict)
            else str(error_details)
        )
        return None, f"Error: API Status Error {e.status_code} - {error_message}"
    except Exception as e:
        logger.error(f"An unexpected error occurred invoking Anthropic: {e}", exc_info=True)
        return None, f"Error: Unexpected error - {e}"
