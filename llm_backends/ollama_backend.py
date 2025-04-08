import ollama
import json
import logging

logger = logging.getLogger(__name__)

def format_tools_for_ollama(tools_data: list[dict]) -> list[dict]:
    """Formats the generic tool definition for Ollama API."""
    formatted_tools = []
    for tool in tools_data:
        if tool.get("type") == "function" and "function" in tool:
            formatted_tools.append(
                {
                    "type": "function",
                    "function": tool["function"] # Ollama expects structure under 'function'
                 }
            )
    return formatted_tools

def invoke_ollama(
    model: str,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
    temperature: float,
    host: str | None = None,
    **kwargs # Accept other potential config args
) -> tuple[str | None, str | None]:
    """
    Invokes the Ollama API and returns the called tool name and the raw response content.

    Returns:
        tuple[str | None, str | None]: (called_tool_name, raw_response_content or error_message)
    """
    client_args = {}
    if host:
        client_args['host'] = host

    ollama_tools = format_tools_for_ollama(tools)
    full_messages = []
    if system_prompt:
         full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    try:
        logger.debug(f"Ollama Request: model={model}, messages={json.dumps(full_messages, indent=2)}, tools={json.dumps(ollama_tools, indent=2)}")
        response: ollama.ChatResponse = ollama.chat(
            model=model,
            messages=full_messages,
            tools=ollama_tools if ollama_tools else None,
            options={"temperature": temperature},
            **client_args
        )
        logger.debug(f"Ollama Response: {response}")

        message = response.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")

        if tool_calls:
            # Assuming only one tool call for simplicity in this eval
            called_tool = tool_calls[0]["function"]["name"]
            # You might want to log arguments too: tool_calls[0]['function']['arguments']
            return called_tool, content or f"Tool call requested: {called_tool}"
        else:
            return None, content # No tool called

    except Exception as e:
        logger.error(f"Ollama API Error: {e}")
        return None, f"Error: {e}"
