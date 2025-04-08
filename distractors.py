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
            "content": None, # Anthropic specific for tool use request
            "tool_calls": [
                {
                    "id": "tool_call_1",
                    "type": "function",
                    "function": {
                        "name": "builtin_web_search",
                        "arguments": '{"query": "recent AI news"}',
                    },
                }
            ],
        },
         { # In Anthropic format, need tool_use role
            "role": "tool", # or "tool_use" / "function" depending on API/library
            "tool_use_id": "tool_call_1",
            "content": '[{"title": "AI breakthrough announced", "snippet": "..."}]', # Simulated tool result
        },
        {
            "role": "assistant",
            "content": "I found an article about a recent AI breakthrough."
        },
    ],
    "mixed_conversation": [
        {"role": "user", "content": "Hello there!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Tell me a short joke."},
        {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"},
        {"role": "user", "content": "Interesting. What about planets?"},
        {"role": "assistant", "content": "Planets are fascinating celestial bodies orbiting stars!"},
        ]
}

def get_distractor_messages(name: str) -> list[dict] | None:
    """Returns a predefined list of distractor messages."""
    return DISTRACTOR_SETS.get(name)

def get_random_distractor() -> tuple[str, list[dict]]:
    """Returns a random distractor set and its name."""
    name = random.choice(list(DISTRACTOR_SETS.keys()))
    return name, DISTRACTOR_SETS[name]
