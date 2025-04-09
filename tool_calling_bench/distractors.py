import random

# --- Distractor Message Sets ---

DISTRACTOR_SETS = {
    "simple_qa": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
    ],
    "simulated_tool_use": [
        {"role": "user", "content": "Can you search for recent AI news?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_call_1",
                    "function": {
                        "name": "builtin_web_search",
                        "arguments": {"query": "recent AI news"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_use_id": "tool_call_1",
            "content": '[{"title": "AI breakthrough announced", "snippet": "..."}]',
        },
        {
            "role": "assistant",
            "content": "I found an article about a recent AI breakthrough.",
        },
    ],
}


def get_distractor_messages(name: str) -> list[dict] | None:
    """Returns a predefined list of distractor messages in Ollama format."""
    return DISTRACTOR_SETS.get(name)


def get_random_distractor() -> tuple[str, list[dict]]:
    """Returns a random distractor set and its name."""
    name = random.choice(list(DISTRACTOR_SETS.keys()))
    return name, DISTRACTOR_SETS[name]
