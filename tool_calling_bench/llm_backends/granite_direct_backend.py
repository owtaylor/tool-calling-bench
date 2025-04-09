# llm_backends/granite_direct_backend.py
import ollama
import json
import logging
from typing import Tuple, Optional, List, Dict, Any

from .ollama_backend import get_num_ctx
from ..secrets_manager import Secrets

logger = logging.getLogger(__name__)


def format_granite_prompt(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] = None,
) -> str:
    """
    Format messages according to the Granite template format.

    Args:
        system_prompt: The system prompt to use
        messages: List of message dictionaries with role and content
        tools: Optional list of tool definitions

    Returns:
        A string formatted according to the Granite template
    """
    formatted_messages = []

    # Handle system part
    formatted_messages.append("<|start_of_role|>system<|end_of_role|>")
    if system_prompt:
        formatted_messages.append(system_prompt)
    else:
        formatted_messages.append(
            "Knowledge Cutoff Date: April 2024.\nYou are Granite, developed by IBM."
        )
        if tools:
            formatted_messages.append(
                " You are a helpful AI assistant with access to the following tools. "
                "When a tool is required to answer the user's query, respond with <|tool_call|> "
                "followed by a JSON list of tools used. If a tool does not exist in the provided "
                "list of tools, notify the user that you do not have the ability to fulfill the request."
            )
        else:
            formatted_messages.append(" You are a helpful AI assistant.")
    formatted_messages.append("<|end_of_text|>")

    # Handle tools part
    if tools:
        formatted_messages.append("<|start_of_role|>tools<|end_of_role|>[")
        tool_json_strings = []
        for tool in tools:
            tool_json_strings.append(json.dumps(tool))
        formatted_messages.append(",".join(tool_json_strings))
        formatted_messages.append("]<|end_of_text|>")
    else:
        formatted_messages.append(" You are a helpful AI assistant.<|end_of_text|>")

    # Process conversation messages
    for i, msg in enumerate(messages):
        role = msg["role"]
        if role == "tool":
            role_text = "tool_response"
        else:
            role_text = role

        formatted_messages.append(f"<|start_of_role|>{role_text}<|end_of_role|>")

        # Handle content or tool calls
        if "content" in msg and msg["content"]:
            formatted_messages.append(msg["content"])
        elif "tool_calls" in msg and msg["tool_calls"]:
            formatted_messages.append("<|tool_call|>")
            for tool_call in msg["tool_calls"]:
                function = tool_call["function"]
                formatted_messages.append(
                    f'{{"name": "{function["name"]}", "arguments": {function["arguments"]}}}'
                )

        # Handle end of conversation vs. continuing
        is_last_message = i == len(messages) - 1
        if is_last_message:
            if role == "assistant":
                # No need to add anything for last assistant message
                pass
            else:
                formatted_messages.append("<|end_of_text|>")
                formatted_messages.append("<|start_of_role|>assistant<|end_of_role|>")
        else:
            formatted_messages.append("<|end_of_text|>")

    return "\n".join(formatted_messages)


def extract_first_json_object(text: str, start_position: int = 0) -> Tuple[Optional[dict], int]:
    """
    Parse the first complete JSON object from a string, starting from a specified position.

    Args:
        text: String that may contain one or more JSON objects
        start_position: Index to start searching from (will ignore whitespace from this position)

    Returns:
        Tuple of (parsed_object, end_position) where:
            - parsed_object is the first JSON object found or None if none could be parsed
            - end_position is the index where the parsed object ends in the original string
    """
    if start_position < 0 or start_position >= len(text):
        return None, -1  # Invalid start position

    # Find the first non-whitespace character from start_position
    start_idx = start_position
    while start_idx < len(text) and text[start_idx].isspace():
        start_idx += 1

    # Check if we've reached the end of the string or if the first non-whitespace
    # character is not the start of a JSON object/array
    if start_idx >= len(text) or text[start_idx] not in ["{", "["]:
        return None, -1  # No JSON object found at specified position (after whitespace)
    # Keep track of brackets to find the complete object
    stack = []
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        char = text[i]

        # Handle string literals, which might contain brackets that don't count
        if char == '"' and not escape_next:
            in_string = not in_string
        elif in_string and char == "\\" and not escape_next:
            escape_next = True
            continue
        elif not in_string:
            if char in ["{", "["]:
                stack.append(char)
            elif char == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif char == "]" and stack and stack[-1] == "[":
                stack.pop()

        escape_next = False

        # When stack is empty, we've found the complete object
        if i >= start_idx and not stack:
            try:
                json_obj = json.loads(text[start_idx : i + 1])
                return json_obj, i + 1
            except json.JSONDecodeError:
                # This should rarely happen as our parsing logic should ensure valid JSON
                pass

    # If we reach here, we didn't find a complete object
    return None, -1


def extract_tool_call(response_text: str) -> Tuple[Optional[str], str]:
    """
    Extract tool call from the response text if present.

    Args:
        response_text: Raw response text from the model

    Returns:
        Tuple of (tool_name, cleaned_response) or (None, original_response)
    """
    # Handle both <|tool_call|> and <tool_call> formats
    tool_call_markers = ["<|tool_call|>", "<tool_call>"]

    # Find the first tool call marker in the response
    marker_position = -1
    marker_used = None

    for marker in tool_call_markers:
        pos = response_text.find(marker)
        if pos != -1 and (marker_position == -1 or pos < marker_position):
            marker_position = pos
            marker_used = marker

    if marker_position == -1 or marker_used is None:
        # No tool call marker found
        return None, response_text

    # Split the response at the tool call marker
    content_part = response_text[:marker_position].strip()
    tool_call_part = response_text[marker_position + len(marker_used) :].strip()

    # Extract the first complete JSON object from the tool call part
    tool_obj, json_end_pos = extract_first_json_object(tool_call_part)

    if (
        isinstance(tool_obj, list)
        and len(tool_obj) > 0
        and isinstance(tool_obj[0], dict)
        and "name" in tool_obj[0]
    ):
        # Successfully extracted tool call
        tool_name = tool_obj[0]["name"]

        # If there's any text after the JSON object, add it back to the content
        if json_end_pos < len(tool_call_part):
            remaining_text = tool_call_part[json_end_pos:].strip()
            if remaining_text:
                if content_part != "":
                    content_part = f"{content_part}\n{remaining_text}"
                else:
                    content_part = remaining_text

        return tool_name, content_part

    # JSON parsing failed - the response contains a tool call marker but no valid JSON
    logger.warning(
        f"Found {marker_used} marker but couldn't parse tool call: {tool_call_part[:100]}..."
    )
    return None, response_text


def invoke_granite_direct(
    model: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: float,
    secrets: Secrets,
    host: str | None = None,
    **kwargs,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Invokes the Ollama API using the generate endpoint with Granite formatting.

    Args:
        model: The Ollama model name (e.g., 'granite').
        system_prompt: The system prompt string.
        messages: A list of message dictionaries.
        tools: A list of tool definitions in the standard format.
        temperature: The sampling temperature.
        secrets: The Secrets object.
        host: Optional URL of the Ollama host.
        **kwargs: Additional keyword arguments for the Ollama client.

    Returns:
        A tuple containing:
        - The name of the called tool (str) or None if no tool was called.
        - The content of the assistant's response (str) or an error message.
    """
    client_args = {}
    effective_host = host
    if effective_host:
        client_args["host"] = effective_host

    try:
        client = ollama.Client(**client_args)
    except Exception as e:
        logger.error(f"Failed to create Ollama client (host={effective_host}): {e}")
        return None, f"Error: Failed to create Ollama client - {e}"

    # Format the prompt according to Granite's template
    formatted_prompt = format_granite_prompt(system_prompt, messages, tools)

    # Prepare generate options
    generate_options = {
        "temperature": temperature,
    }

    # Add any other relevant options from kwargs
    for key in ["num_ctx", "top_k", "top_p", "stop"]:
        if key in kwargs:
            generate_options[key] = kwargs[key]

    try:
        logger.debug(
            f"Granite Direct Request: model={model}, host={effective_host}, prompt={formatted_prompt}, options={generate_options}"
        )

        # Use the generate API instead of chat
        response = client.generate(
            model=model,
            prompt=formatted_prompt,
            options=generate_options,
            raw=True,
        )

        num_ctx = get_num_ctx(client, model, kwargs)
        if response.prompt_eval_count + response.prompt_eval_count >= num_ctx:
            return None, f"Error: context length exceeded"

        logger.debug(f"Granite Direct Response: {response}")

        # Extract response content
        response_text = response.get("response", "")

        # Try to extract tool calls from the response
        tool_name, content = extract_tool_call(response_text)

        return tool_name, content

    except ollama.ResponseError as e:
        logger.error(f"Ollama API Response Error: Status Code: {e.status_code}, Error: {e.error}")
        return None, f"Error: Ollama API Error {e.status_code} - {e.error}"
    except Exception as e:
        logger.error(
            f"Granite Direct invocation error (host={effective_host}, model={model}): {e}",
            exc_info=True,
        )
        return None, f"Error: Granite Direct invocation failed - {e}"
