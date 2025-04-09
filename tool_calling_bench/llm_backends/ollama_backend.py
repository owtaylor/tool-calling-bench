# llm_backends/ollama_backend.py
import ollama
import json
import logging
import re
from typing import Tuple, Optional, List, Dict, Any
from ..secrets_manager import Secrets  # <-- Import Secrets for type hinting

logger = logging.getLogger(__name__)


def format_tools_for_ollama(tools_data: list[dict]) -> list[dict]:
    """Formats the generic tool definition for Ollama API."""
    formatted_tools = []
    for tool in tools_data:
        # Ensure the tool definition has the expected structure for Ollama
        if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
            # Ollama expects 'function' to contain 'name', 'description', 'parameters'
            func_details = tool["function"]
            if (
                isinstance(func_details, dict)
                and "name" in func_details
                and "parameters" in func_details
            ):
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": func_details["name"],
                            "description": func_details.get(
                                "description", ""
                            ),  # Add description if available
                            "parameters": func_details["parameters"],
                        },
                    }
                )
            else:
                logger.warning(
                    f"Skipping tool due to missing 'name' or 'parameters' in function details: {tool}"
                )
        else:
            logger.warning(f"Skipping tool due to unexpected format: {tool}")
    return formatted_tools


def get_num_ctx(client: ollama.Client, model: str, kwargs: dict = None) -> int:
    """
    Retrieve the context window size from the model parameters or kwargs.

    Args:
        client: Ollama client instance
        model: The model name
        kwargs: Additional keyword arguments that might contain num_ctx

    Returns:
        int: The context window size, defaulting to 2048 if not found
    """
    # First check if num_ctx is provided in kwargs
    if kwargs and "num_ctx" in kwargs:
        return kwargs["num_ctx"]

    try:
        # Get model information from Ollama API
        model_info = client.show(model)

        # In Ollama, model_info.parameters is a string containing model parameters
        if "parameters" in model_info and model_info["parameters"]:
            params_str = model_info["parameters"]

            # Look for num_ctx parameter
            pattern = r"num_ctx\s*=\s*(\d+)"
            match = re.search(pattern, params_str)
            if match:
                return int(match.group(1))

        # If we couldn't find it in parameters, return default value
        return 2048
    except Exception as e:
        # Log the error but don't fail
        logger.error(f"Error retrieving context window size for {model}: {e}")
        return 2048


def invoke_ollama(
    model: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: float,
    secrets: Secrets,
    host: str | None = None,
    # Add other Ollama specific parameters if needed from config
    **kwargs,  # Catch any other backend-specific params from config
) -> Tuple[Optional[str], Optional[str]]:
    """
    Invokes the Ollama API and returns the called tool name and the raw response content.

    Args:
        model: The Ollama model name (e.g., 'llama3:instruct').
        system_prompt: The system prompt string.
        messages: A list of message dictionaries.
        tools: A list of tool definitions in the standard format.
        temperature: The sampling temperature.
        secrets: The Secrets object (currently unused by Ollama backend, but included for consistency).
        host: Optional URL of the Ollama host.
        **kwargs: Additional keyword arguments for the Ollama client or chat options.

    Returns:
        A tuple containing:
        - The name of the called tool (str) or None if no tool was called.
        - The content of the assistant's response (str) or an error message.
    """
    # NOTE: Ollama typically doesn't require API keys like cloud services.
    # The 'secrets' object is included for interface consistency but might not be used
    # unless a specific setup (e.g., authenticated proxy) requires a secret fetched via secrets.get().
    # Example: auth_token = secrets.get("OLLAMA_AUTH_TOKEN")

    client_args = {}
    # Allow host override from config, potentially fetched from secrets if needed in future
    # host_from_secrets = secrets.get("OLLAMA_HOST")
    # effective_host = host_from_secrets or host
    effective_host = host  # Keep using host from config for now
    if effective_host:
        client_args["host"] = effective_host

    try:
        client = ollama.Client(**client_args)
    except Exception as e:
        logger.error(f"Failed to create Ollama client (host={effective_host}): {e}")
        return None, f"Error: Failed to create Ollama client - {e}"

    ollama_tools = format_tools_for_ollama(tools)
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    # Separate Ollama-specific options from general kwargs if needed
    ollama_options = {"temperature": temperature}
    # Add any other relevant options from kwargs if they match ollama.chat options
    for key in ["num_ctx", "top_k", "top_p", "stop"]:  # Example options
        if key in kwargs:
            ollama_options[key] = kwargs[key]

    try:
        logger.debug(
            f"Ollama Request: model={model}, host={effective_host}, messages={json.dumps(full_messages, indent=2)}, tools={json.dumps(ollama_tools, indent=2)}, options={ollama_options}"
        )

        response = client.chat(
            model=model,
            messages=full_messages,
            tools=ollama_tools if ollama_tools else None,  # Pass None if no valid tools
            options=ollama_options,
            # Pass keep_alive if present in kwargs? ollama library handles it differently
            # keep_alive=kwargs.get("keep_alive")
        )
        logger.debug(f"Ollama Response: {response}")
        message = response.get("message", {})
        content = message.get("content", "") or ""  # Ensure content is string or empty string
        tool_calls = message.get("tool_calls")

        num_ctx = get_num_ctx(client, model, kwargs)
        if response.prompt_eval_count + response.prompt_eval_count >= num_ctx:
            return None, f"Error: context length exceeded"

        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            # Process tool calls - assuming only one for this benchmark's purpose
            first_call = tool_calls[0]
            called_tool_name = first_call.function.name
            return called_tool_name, (
                content if content else f"Tool call requested: {called_tool_name}"
            )

        # No valid tool call found or processed
        return None, content

    except ollama.ResponseError as e:
        logger.error(f"Ollama API Response Error: Status Code: {e.status_code}, Error: {e.error}")
        return None, f"Error: Ollama API Error {e.status_code} - {e.error}"
    except Exception as e:
        # Catch potential connection errors, timeouts, etc.
        logger.error(
            f"Ollama invocation error (host={effective_host}, model={model}): {e}",
            exc_info=True,
        )
        return None, f"Error: Ollama invocation failed - {e}"
