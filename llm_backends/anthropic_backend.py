import anthropic
import os
import json
import logging

logger = logging.getLogger(__name__)

# Ensure API key is available
# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
# if not ANTHROPIC_API_KEY:
#     logger.warning("ANTHROPIC_API_KEY environment variable not set.")
#     # Or raise an error if it's strictly required:
#     # raise ValueError("ANTHROPIC_API_KEY environment variable must be set for Anthropic backend")


def format_tools_for_anthropic(tools_data: list[dict]) -> list[dict]:
    """Formats the generic tool definition for Anthropic API."""
    formatted_tools = []
    for tool in tools_data:
         if tool.get("type") == "function" and "function" in tool:
             # Anthropic expects 'name', 'description', 'input_schema' (which is 'parameters')
             func_def = tool["function"]
             formatted_tools.append(
                 {
                    "name": func_def.get("name"),
                    "description": func_def.get("description"),
                    "input_schema": func_def.get("parameters")
                 }
             )
    return formatted_tools


def invoke_anthropic(
    model: str,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
    temperature: float,
    api_key: str | None = None, # Allow override from config
    max_tokens: int = 1024,
    **kwargs # Accept other potential config args
) -> tuple[str | None, str | None]:
    """
    Invokes the Anthropic API and returns the called tool name and the raw response content.

    Returns:
        tuple[str | None, str | None]: (called_tool_name, raw_response_content or error_message)
    """
    try:
        # Initialize client here to potentially use key from config
        effective_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not effective_api_key:
             raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or provide it in the config.")

        client = anthropic.Anthropic(api_key=effective_api_key)

        anthropic_tools = format_tools_for_anthropic(tools)

        # Filter out None content for assistant tool_calls messages if needed by API
        valid_messages = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls") and msg.get("content") is None:
                 # Some APIs might prefer omitting content key entirely for tool calls
                 valid_messages.append({k:v for k,v in msg.items() if k != 'content'})
            elif msg.get("role") == "tool": # Anthropic uses 'tool' role for tool results
                 # Need to ensure 'tool_use_id' exists if adapting from other formats
                 if 'tool_use_id' in msg and 'content' in msg:
                     valid_messages.append({
                        "role": "tool",
                        "tool_use_id": msg["tool_use_id"],
                        "content": msg["content"] # Content here is the result FROM the tool
                     })
            else:
                valid_messages.append(msg)


        logger.debug(f"Anthropic Request: model={model}, system='{system_prompt}', messages={json.dumps(valid_messages, indent=2)}, tools={json.dumps(anthropic_tools, indent=2)}")

        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=valid_messages,
            tools=anthropic_tools if anthropic_tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.debug(f"Anthropic Response: {response.model_dump_json(indent=2)}")

        called_tool_name = None
        response_text = ""

        # Anthropic response structure processing
        if response.stop_reason == "tool_use":
            for content_block in response.content:
                if content_block.type == "tool_use":
                    called_tool_name = content_block.name
                    # Log arguments: content_block.input
                    # Assuming one tool call per turn for this eval
                    break
            response_text = f"Tool call requested: {called_tool_name}" # Placeholder text as primary response is tool call
        elif response.content and response.content[0].type == "text":
             response_text = response.content[0].text

        return called_tool_name, response_text

    except anthropic.APIError as e:
        logger.error(f"Anthropic API Error: {e.status_code} - {e.body}")
        return None, f"API Error: {e.message}"
    except Exception as e:
        logger.error(f"Anthropic Invocation Error: {e}")
        return None, f"Error: {e}"
