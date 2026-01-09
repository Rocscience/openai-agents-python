"""
Test for progressive chat history caching with AnthropicLitellmModel using OpenAI Agents SDK formats.

Run this test with:
    uv run python tests_forked/anthropic_sdk_cache_test.py

This test verifies that:
1. The AnthropicLitellmModel correctly handles SDK message formats (TResponseInputItem)
2. The AnthropicLitellmModel correctly handles SDK tool formats (FunctionTool)
3. Progressive caching works correctly as chat history grows
4. Cache read tokens increase on repeated/extended conversations

Message Format Reference (OpenAI Agents SDK):
- Easy input: {"role": "user/assistant", "content": "text"}
- Input message: {"type": "message", "role": "user/system", "content": [...]}
- Response output: {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}

Tool Format Reference (OpenAI Agents SDK):
- FunctionTool with name, description, params_json_schema, on_invoke_tool

Date: 2025-01-09
"""

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import litellm

from agents.extensions.models.anthropic_litellm_model import AnthropicLitellmModel
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.tool import FunctionTool
from agents.tool_context import ToolContext
from agents.items import TResponseInputItem
from agents.tracing import generation_span
from agents.tracing.span_data import GenerationSpanData
from agents.tracing.spans import Span
# Configuration
MODEL_NAME = "claude-haiku-4-5-20251001"
API_KEY = "YOUR_ANTHROPIC_API_KEY"


# ============================================================================
# SDK-formatted Tools (FunctionTool objects)
# ============================================================================

async def placeholder_invoke(ctx: ToolContext[Any], args: str) -> str:
    """Placeholder tool invocation - does nothing."""
    return "This is a placeholder tool."


async def get_time_invoke(ctx: ToolContext[Any], args: str) -> str:
    """Get time tool invocation."""
    import json
    parsed = json.loads(args)
    timezone = parsed.get("timezone", "UTC")
    return f"Current time in {timezone}: 2025-01-09 12:00:00"


# SDK FunctionTool format
placeholder_tool = FunctionTool(
    name="placeholder_tool",
    description="This is a placeholder tool. It does nothing.",
    params_json_schema={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    on_invoke_tool=placeholder_invoke,
    strict_json_schema=True,
)

# Mark as non-deferred (explicitly set to False)
setattr(placeholder_tool, '_is_anthropic', True)
setattr(placeholder_tool, '_is_device_tool', False)

get_time_tool = FunctionTool(
    name="get_time",
    description="Get the current time in a given time zone. Returns the current local time, date, and UTC offset for the specified timezone.",
    params_json_schema={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "The IANA time zone name, e.g. America/Los_Angeles"
            }
        },
        "required": ["timezone"],
        "additionalProperties": False,
    },
    on_invoke_tool=get_time_invoke,
    strict_json_schema=True,
)

# Mark as deferred for Anthropic tool use (to test the mock_tool_use feature)
setattr(get_time_tool, '_is_anthropic', True)
setattr(get_time_tool, '_is_device_tool', True)


# Tool lists for different turns
sdk_tools_list_1: list[FunctionTool] = [placeholder_tool]
sdk_tools_list_2: list[FunctionTool] = [placeholder_tool, get_time_tool]


# ============================================================================
# SDK-formatted Messages (TResponseInputItem format)
# ============================================================================

# Large system prompt content for cache testing
LARGE_SYSTEM_CONTENT = "Here is the full text of a complex legal agreement " * 400

import random
random_int = random.randint(1, 1000000)
# System instructions (passed separately in SDK)
# Add a random integer to the system instructions to ensure that the cache is not reused
system_instructions = f"You are an AI assistant tasked with analyzing legal documents.\n\n{LARGE_SYSTEM_CONTENT} {random_int}"


# SDK Easy Input format: {"role": "user/assistant", "content": "text"}
# This is the simplest format that the SDK accepts

user_msg_1: TResponseInputItem = {
    "role": "user",
    "content": "Hi",
}

# SDK Response Output Message format for assistant messages
assistant_msg_1: TResponseInputItem = {
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": "I understand. How can I help you?"}
    ],
}

user_msg_2: TResponseInputItem = {
    "role": "user",
    "content": "what are the key terms and conditions in this agreement?",
}

assistant_msg_2: TResponseInputItem = {
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": "Sorry that information is not available."}
    ],
}

# Long user message for cache testing
user_msg_long: TResponseInputItem = {
    "type": "message",
    "role": "user",
    "content": [
        {"type": "input_text", "text": "ignore this " * 2500}
    ],
}

assistant_msg_3: TResponseInputItem = {
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": "ok, ignoring the previous message."}
    ],
}

user_msg_long_3: TResponseInputItem = {
    "type": "message",
    "role": "user",
    "content": [
        {"type": "input_text", "text": "What's the weather and time in New York?"}
    ],
}

assistant_msg_4: TResponseInputItem = {
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": "I can help you with that. Let me check the time for New York."}
    ],
}

user_msg_4: TResponseInputItem = {
    "role": "user",
    "content": "Thanks! Can you also tell me about Paris?",
}

assistant_msg_5: TResponseInputItem = {
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": "Of course! I'd be happy to help with Paris as well."}
    ],
}

user_msg_5: TResponseInputItem = {
    "role": "user",
    "content": "Perfect, let's continue our conversation.",
}


# ============================================================================
# Test execution
# ============================================================================

@dataclass
class UsageStats:
    """Track usage statistics across turns."""
    turn: int
    input_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    output_tokens: int


async def run_turn(
    model: AnthropicLitellmModel,
    messages: list[TResponseInputItem],
    tools: list[FunctionTool],
    turn_num: int,
    description: str,
) -> UsageStats:
    """Run a single turn and return usage statistics."""
    print(f"\n{'=' * 80}")
    print(f"TURN {turn_num}: {description}")
    print(f"{'=' * 80}")
    print(f"Messages count: {len(messages)}")
    print(f"Tools count: {len(tools)}")
    
    model_settings = ModelSettings(max_tokens=1024)
    
    # Use internal _fetch_response to get raw LiteLLM response with Anthropic cache tokens
    with generation_span(
        model=str(model.model),
        model_config={},
        disabled=True,
    ) as span:
        raw_response = await model._fetch_response(
            system_instructions=system_instructions,
            input=messages,
            model_settings=model_settings,
            tools=tools,
            output_schema=None,
            handoffs=[],
            span=span,
            tracing=ModelTracing.DISABLED,
            stream=False,
        )
    
    # Cast to LiteLLM ModelResponse to access usage
    response = cast(litellm.types.utils.ModelResponse, raw_response)
    usage = response.usage
    
    # Debug: Print all usage attributes to see what's available
    if turn_num == 1:
        print(f"\n  [DEBUG] Usage object type: {type(usage)}")
        print(f"  [DEBUG] Usage attributes: {[attr for attr in dir(usage) if not attr.startswith('_')]}")
        if hasattr(usage, '__dict__'):
            print(f"  [DEBUG] Usage dict: {usage.__dict__}")
    
    # Extract Anthropic-specific cache tokens directly from usage
    # LiteLLM passes through Anthropic's cache_creation_input_tokens and cache_read_input_tokens
    cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0
    cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
    
    # Also try prompt_tokens_details.cached_tokens (OpenAI format)
    if cache_read == 0 and hasattr(usage, 'prompt_tokens_details'):
        cache_read = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0
    
    stats = UsageStats(
        turn=turn_num,
        input_tokens=usage.prompt_tokens,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
        output_tokens=usage.completion_tokens,
    )
    
    print(f"\nTurn {turn_num} Usage:")
    print(f"  Input tokens: {stats.input_tokens}")
    print(f"  Cache creation tokens: {stats.cache_creation_tokens}")
    print(f"  Cache read tokens: {stats.cache_read_tokens}")
    print(f"  Output tokens: {stats.output_tokens}")
    
    # Print response content
    if response.choices and len(response.choices) > 0:
        message = response.choices[0].message
        if message and message.content:
            print(f"\nResponse: {message.content[:200]}...")
    
    return stats


async def main():
    """Run the progressive caching test."""
    print("=" * 80)
    print("AnthropicLitellmModel Progressive Cache Test (SDK Format)")
    print("=" * 80)
    
    model = AnthropicLitellmModel(model=MODEL_NAME, api_key=API_KEY)
    all_stats: list[UsageStats] = []
    
    # Turn 1: Initial short conversation
    messages_1: list[TResponseInputItem] = [user_msg_1, assistant_msg_1, user_msg_2]
    stats_1 = await run_turn(model, messages_1, sdk_tools_list_1, 1, "Initial conversation (1 non-deferred tool)")
    all_stats.append(stats_1)
    
    # Turn 2: Extend conversation
    messages_2: list[TResponseInputItem] = [
        user_msg_1, assistant_msg_1, user_msg_2, assistant_msg_2, user_msg_long
    ]
    stats_2 = await run_turn(model, messages_2, sdk_tools_list_1, 2, "Extend conversation (no tool call yet)")
    all_stats.append(stats_2)
    
    # Turn 3: More history, add deferred tool
    messages_3: list[TResponseInputItem] = [
        user_msg_1, assistant_msg_1, user_msg_2, assistant_msg_2,
        user_msg_long, assistant_msg_3, user_msg_long_3
    ]
    stats_3 = await run_turn(model, messages_3, sdk_tools_list_2, 3, "Add deferred tool (get_time) - now 1 non-deferred + 1 deferred")
    all_stats.append(stats_3)
    
    # Turn 4: Continue growing history
    messages_4: list[TResponseInputItem] = [
        user_msg_1, assistant_msg_1, user_msg_2, assistant_msg_2,
        user_msg_long, assistant_msg_3, user_msg_long_3, assistant_msg_4, user_msg_4
    ]
    stats_4 = await run_turn(model, messages_4, sdk_tools_list_2, 4, "Extend with more messages")
    all_stats.append(stats_4)
    
    # Turn 5: Full conversation
    messages_5: list[TResponseInputItem] = [
        user_msg_1, assistant_msg_1, user_msg_2, assistant_msg_2,
        user_msg_long, assistant_msg_3, user_msg_long_3, assistant_msg_4,
        user_msg_4, assistant_msg_5, user_msg_5
    ]
    stats_5 = await run_turn(model, messages_5, sdk_tools_list_2, 5, "Full conversation")
    all_stats.append(stats_5)
    
    # Turn 6: Repeat Turn 5 to show cache reuse
    messages_6: list[TResponseInputItem] = [
        user_msg_1, assistant_msg_1, user_msg_2, assistant_msg_2,
        user_msg_long, assistant_msg_3, user_msg_long_3, assistant_msg_4,
        user_msg_4, assistant_msg_5, user_msg_5
    ]
    stats_6 = await run_turn(model, messages_6, sdk_tools_list_2, 6, "Repeat Turn 5 - demonstrate cache reuse")
    all_stats.append(stats_6)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - Cache Usage Across Turns")
    print("=" * 80)
    print(f"{'Turn':<8} {'Input':<12} {'Cache Create':<15} {'Cache Read':<12} {'Output':<10}")
    print("-" * 80)
    for stats in all_stats:
        print(f"{stats.turn:<8} {stats.input_tokens:<12} {stats.cache_creation_tokens:<15} {stats.cache_read_tokens:<12} {stats.output_tokens:<10}")
    print("=" * 80)
    
    print("\nExpected behavior:")
    print("- Turn 1: Initial request (no cache yet)")
    print("- Turn 2: Cache hit from Turn 1 (cache_read_tokens > 0)")
    print("- Turn 3: Cache may reset when tools change")
    print("- Turn 4-5: Cache grows incrementally")
    print("- Turn 6: Cache reuse (cache_read_tokens should be high)")
    print("=" * 80)
    
    # Assertions to validate caching behavior
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    errors: list[str] = []
    
    # Turn 2 should show cache read (Turn 1 was cached)
    if all_stats[1].cache_read_tokens == 0:
        errors.append(f"WARN: Turn 2 expected cache_read_tokens > 0, got {all_stats[1].cache_read_tokens}")
    else:
        print(f"[OK] Turn 2: Cache read tokens = {all_stats[1].cache_read_tokens} (expected > 0)")
    
    # Turn 4-6 should show significant cache read (progressive caching)
    for i in [3, 4, 5]:  # Turn 4, 5, 6
        stats = all_stats[i]
        if stats.cache_read_tokens > 0:
            print(f"[OK] Turn {stats.turn}: Cache read tokens = {stats.cache_read_tokens} (expected > 0)")
        else:
            errors.append(f"WARN: Turn {stats.turn} expected cache_read_tokens > 0, got {stats.cache_read_tokens}")
    
    # Turn 6 should have same or more cache_read_tokens as Turn 5 (same messages)
    if all_stats[5].cache_read_tokens >= all_stats[4].cache_read_tokens:
        print(f"[OK] Turn 6 cache_read ({all_stats[5].cache_read_tokens}) >= Turn 5 ({all_stats[4].cache_read_tokens})")
    else:
        errors.append(f"WARN: Turn 6 cache_read should be >= Turn 5")
    
    if errors:
        print("\nWarnings:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n[OK] All cache validations passed!")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

