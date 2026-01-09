from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from copy import copy
from datetime import datetime
from pathlib import Path
from tkinter import TRUE
from typing import Any, Literal, cast, overload

from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

try:
    import litellm
except ImportError as _e:
    raise ImportError(
        "`litellm` is required to use the LitellmModel. You can install it via the optional "
        "dependency group: `pip install 'openai-agents[litellm]'`."
    ) from _e

from openai import AsyncStream, NotGiven, omit
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from openai.types.responses import Response

from ... import _debug
from ...agent_output import AgentOutputSchemaBase
from ...handoffs import Handoff
from ...items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ...logger import logger
from ...model_settings import ModelSettings
from ...models.chatcmpl_converter import Converter
from ...models.chatcmpl_helpers import HEADERS, HEADERS_OVERRIDE
from ...models.chatcmpl_stream_handler import ChatCmplStreamHandler
from ...models.fake_id import FAKE_RESPONSES_ID
from ...models.interface import Model, ModelTracing
from ...models.openai_responses import Converter as OpenAIResponsesConverter
from ...tool import Tool
from ...tracing import generation_span
from ...tracing.span_data import GenerationSpanData
from ...tracing.spans import Span
from ...usage import Usage
from ...util._json import _to_dump_compatible
from .litellm_model import LitellmConverter


def add_cache_control_to_last_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add cache_control to the last message in the conversation (mutates in place)."""
    if not messages:
        return messages

    last_msg = messages[-1]
    content = last_msg.get("content")

    # Handle string content.
    if isinstance(content, str):
        last_msg["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    # Handle list content.
    elif isinstance(content, list):
        # Add cache_control to the last text block.
        for j in range(len(content) - 1, -1, -1):
            if isinstance(content[j], dict) and content[j].get("type") == "text":
                content[j]["cache_control"] = {"type": "ephemeral"}
                break

    return messages


class AnthropicLitellmModel(Model):
    """This class enables using any model via LiteLLM. LiteLLM allows you to acess OpenAPI,
    Anthropic, Gemini, Mistral, and many other models.
    See supported models here: [litellm models](https://docs.litellm.ai/docs/providers).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,  # unused
        conversation_id: str | None = None,  # unused
        prompt: Any | None = None,
    ) -> ModelResponse:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=False,
                prompt=prompt,
            )

            message: litellm.types.utils.Message | None = None
            first_choice: litellm.types.utils.Choices | None = None
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if isinstance(choice, litellm.types.utils.Choices):
                    first_choice = choice
                    message = first_choice.message

            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Received model response")
            else:
                if message is not None:
                    logger.debug(
                        f"""LLM resp:\n{
                            json.dumps(message.model_dump(), indent=2, ensure_ascii=False)
                        }\n"""
                    )
                else:
                    finish_reason = first_choice.finish_reason if first_choice else "-"
                    logger.debug(f"LLM resp had no message. finish_reason: {finish_reason}")

            if hasattr(response, "usage"):
                response_usage = response.usage
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=response_usage.prompt_tokens,
                        output_tokens=response_usage.completion_tokens,
                        total_tokens=response_usage.total_tokens,
                        input_tokens_details=InputTokensDetails(
                            cached_tokens=getattr(
                                response_usage.prompt_tokens_details, "cached_tokens", 0
                            )
                            or 0
                        ),
                        output_tokens_details=OutputTokensDetails(
                            reasoning_tokens=getattr(
                                response_usage.completion_tokens_details, "reasoning_tokens", 0
                            )
                            or 0
                        ),
                    )
                    if response.usage
                    else Usage()
                )
            else:
                usage = Usage()
                logger.warning("No usage information returned from Litellm")

            if tracing.include_data():
                span_generation.span_data.output = (
                    [message.model_dump()] if message is not None else []
                )
            span_generation.span_data.usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

            # Build provider_data for provider specific fields
            provider_data: dict[str, Any] = {"model": self.model}
            if message is not None and hasattr(response, "id"):
                provider_data["response_id"] = response.id

            items = (
                Converter.message_to_output_items(
                    LitellmConverter.convert_message_to_openai(message, model=self.model),
                    provider_data=provider_data,
                )
                if message is not None
                else []
            )

            return ModelResponse(
                output=items,
                usage=usage,
                response_id=None,
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,  # unused
        conversation_id: str | None = None,  # unused
        prompt: Any | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response, stream = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=True,
                prompt=prompt,
            )

            final_response: Response | None = None
            async for chunk in ChatCmplStreamHandler.handle_stream(
                response, stream, model=self.model
            ):
                yield chunk

                if chunk.type == "response.completed":
                    final_response = chunk.response

            if tracing.include_data() and final_response:
                span_generation.span_data.output = [final_response.model_dump()]

            if final_response and final_response.usage:
                span_generation.span_data.usage = {
                    "input_tokens": final_response.usage.input_tokens,
                    "output_tokens": final_response.usage.output_tokens,
                }

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[True],
        prompt: Any | None = None,
    ) -> tuple[Response, AsyncStream[ChatCompletionChunk]]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[False],
        prompt: Any | None = None,
    ) -> litellm.types.utils.ModelResponse: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: Any | None = None,
    ) -> litellm.types.utils.ModelResponse | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        # Preserve reasoning messages for tool calls when reasoning is on
        # This is needed for models like Claude 4 Sonnet/Opus which support interleaved thinking
        preserve_thinking_blocks = (
            model_settings.reasoning is not None and model_settings.reasoning.effort is not None
        )

        converted_messages = Converter.items_to_messages(
            input,
            preserve_thinking_blocks=preserve_thinking_blocks,
            preserve_tool_output_all_content=True,
            model=self.model,
        )

        # Fix for interleaved thinking bug: reorder messages to ensure tool_use comes before tool_result  # noqa: E501
        if "anthropic" in self.model.lower() or "claude" in self.model.lower():
            converted_messages = self._fix_tool_message_ordering(converted_messages)

        # Convert Google's extra_content to litellm's provider_specific_fields format
        if "gemini" in self.model.lower():
            converted_messages = self._convert_gemini_extra_content_to_provider_specific_fields(
                converted_messages
            )

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        converted_messages = _to_dump_compatible(converted_messages)

        if tracing.include_data():
            span.span_data.input = converted_messages

        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else None
        )
        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = Converter.convert_response_format(output_schema)

        # Defered tools for caching patch
        deferred_tools = []
        for tool in tools:
            #This will basically default the tool to load deferred, so we don't risk breaking cache
            is_anthropic = getattr(tool, "_is_anthropic", TRUE)
            is_device_tool = getattr(tool, "_is_device_tool", TRUE)
            if is_anthropic and is_device_tool:
                deferred_tools.append(tool.name)

        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))

        converted_tools = _to_dump_compatible(converted_tools)

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            messages_json = json.dumps(
                converted_messages,
                indent=2,
                ensure_ascii=False,
            )
            tools_json = json.dumps(
                converted_tools,
                indent=2,
                ensure_ascii=False,
            )
            logger.debug(
                f"Calling Litellm model: {self.model}\n"
                f"{messages_json}\n"
                f"Tools:\n{tools_json}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
            )

        # Build reasoning_effort - use dict only when summary is present (OpenAI feature)
        # Otherwise pass string for backward compatibility with all providers
        reasoning_effort: dict[str, Any] | str | None = None
        if model_settings.reasoning:
            if model_settings.reasoning.summary is not None:
                # Dict format when summary is needed (OpenAI only)
                reasoning_effort = {
                    "effort": model_settings.reasoning.effort,
                    "summary": model_settings.reasoning.summary,
                }
            elif model_settings.reasoning.effort is not None:
                # String format for compatibility with all providers
                reasoning_effort = model_settings.reasoning.effort

        # Enable developers to pass non-OpenAI compatible reasoning_effort data like "none"
        # Priority order:
        #  1. model_settings.reasoning (effort + summary)
        #  2. model_settings.extra_body["reasoning_effort"]
        #  3. model_settings.extra_args["reasoning_effort"]
        if (
            reasoning_effort is None  # Unset in model_settings
            and isinstance(model_settings.extra_body, dict)
            and "reasoning_effort" in model_settings.extra_body
        ):
            reasoning_effort = model_settings.extra_body["reasoning_effort"]
        if (
            reasoning_effort is None  # Unset in both model_settings and model_settings.extra_body
            and model_settings.extra_args
            and "reasoning_effort" in model_settings.extra_args
        ):
            reasoning_effort = model_settings.extra_args["reasoning_effort"]

        stream_options = None
        if stream and model_settings.include_usage is not None:
            stream_options = {"include_usage": model_settings.include_usage}

        extra_kwargs = {}
        if model_settings.extra_query:
            extra_kwargs["extra_query"] = copy(model_settings.extra_query)
        if model_settings.metadata:
            extra_kwargs["metadata"] = copy(model_settings.metadata)
        if model_settings.extra_body and isinstance(model_settings.extra_body, dict):
            extra_kwargs.update(model_settings.extra_body)

        # Add kwargs from model_settings.extra_args, filtering out None values
        if model_settings.extra_args:
            extra_kwargs.update(model_settings.extra_args)

        # Prevent duplicate reasoning_effort kwargs when it was promoted to a top-level argument.
        extra_kwargs.pop("reasoning_effort", None)

        # Insert patch here
        # print(f"Converted messages: {converted_messages}")
        for tool in converted_tools:
            tool_name = tool.get("function", {}).get("name") if "function" in tool else tool.get("name")
            if tool_name in deferred_tools:
                # Add is_deferred to the function dict for OpenAI format
                if "function" in tool:
                    tool["function"]["defer_loading"] = True
                else:
                    tool["defer_loading"] = True

        tool_reference_list = []
        for tool_name in deferred_tools:
            tool_reference_list.append({"type": "tool_reference", "tool_name": tool_name})

        mock_tool_use_msg = {
            "role": "assistant",
            "content": [
                # server_tool_use: the assistant's search request
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_ph",
                    "name": "tool_search_tool_regex",
                    "input": {"query": ".*time.*"},
                },
                # tool_search_tool_result: the search result
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "srvtoolu_ph",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": tool_reference_list,
                    },
                },
            ],
        }

        # Add cache control to the last message
        # Type ignore: converted_messages is already converted to dicts, add_cache_control mutates
        converted_messages_dicts: list[dict[str, Any]] = cast(
            list[dict[str, Any]], converted_messages
        )
        converted_messages_dicts = add_cache_control_to_last_message(converted_messages_dicts)
        converted_messages_dicts = converted_messages_dicts + [mock_tool_use_msg]

        # Log converted messages and tools to a file for debugging
        # TODO: Remove this after debugging
        self._log_request_to_file(converted_messages_dicts, converted_tools)

        ret = await litellm.acompletion(
            model=self.model,
            headers={"anthropic-beta": "advanced-tool-use-2025-11-20"},
            messages=converted_messages_dicts,
            tools=converted_tools or None,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            frequency_penalty=model_settings.frequency_penalty,
            presence_penalty=model_settings.presence_penalty,
            max_tokens=model_settings.max_tokens,
            tool_choice=self._remove_not_given(tool_choice),
            response_format=self._remove_not_given(response_format),
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            stream_options=stream_options,
            reasoning_effort=reasoning_effort,
            top_logprobs=model_settings.top_logprobs,
            extra_headers=self._merge_headers(model_settings),
            api_key=self.api_key,
            base_url=self.base_url,
            **extra_kwargs,
        )

        if isinstance(ret, litellm.types.utils.ModelResponse):
            return ret

        responses_tool_choice = OpenAIResponsesConverter.convert_tool_choice(
            model_settings.tool_choice
        )
        if responses_tool_choice is None or responses_tool_choice is omit:
            responses_tool_choice = "auto"

        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model=self.model,
            object="response",
            output=[],
            tool_choice=responses_tool_choice,  # type: ignore[arg-type]
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls or False,
            reasoning=model_settings.reasoning,
        )
        return response, ret

    def _convert_gemini_extra_content_to_provider_specific_fields(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert Gemini model's extra_content format to provider_specific_fields format for litellm.

        Transforms tool calls from internal format:
            extra_content={"google": {"thought_signature": "..."}}
        To litellm format:
            provider_specific_fields={"thought_signature": "..."}

        Only processes tool_calls that appear after the last user message.
        See: https://ai.google.dev/gemini-api/docs/thought-signatures
        """

        # Find the index of the last user message
        last_user_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                last_user_index = i
                break

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue

            # Only process assistant messages that come after the last user message
            # If no user message found (last_user_index == -1), process all messages
            if last_user_index != -1 and i <= last_user_index:
                continue

            # Check if this is an assistant message with tool calls
            if message.get("role") == "assistant" and message.get("tool_calls"):
                tool_calls = message.get("tool_calls", [])

                for tool_call in tool_calls:  # type: ignore[attr-defined]
                    if not isinstance(tool_call, dict):
                        continue

                    # Default to skip validator, overridden if valid thought signature exists
                    tool_call["provider_specific_fields"] = {
                        "thought_signature": "skip_thought_signature_validator"
                    }

                    # Override with actual thought signature if extra_content exists
                    if "extra_content" in tool_call:
                        extra_content = tool_call.pop("extra_content")
                        if isinstance(extra_content, dict):
                            # Extract google-specific fields
                            google_fields = extra_content.get("google")
                            if google_fields and isinstance(google_fields, dict):
                                thought_sig = google_fields.get("thought_signature")
                                if thought_sig:
                                    tool_call["provider_specific_fields"] = {
                                        "thought_signature": thought_sig
                                    }

        return messages

    def _fix_tool_message_ordering(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """
        Fix the ordering of tool messages to ensure tool_use messages come before tool_result messages.

        This addresses the interleaved thinking bug where conversation histories may contain
        tool results before their corresponding tool calls, causing Anthropic API to reject the request.
        """  # noqa: E501
        if not messages:
            return messages

        # Collect all tool calls and tool results
        tool_call_messages = {}  # tool_id -> (index, message)
        tool_result_messages = {}  # tool_id -> (index, message)
        other_messages = []  # (index, message) for non-tool messages

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                other_messages.append((i, message))
                continue

            role = message.get("role")

            if role == "assistant" and message.get("tool_calls"):
                # Extract tool calls from this assistant message
                tool_calls = message.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_id = tool_call.get("id")
                            if tool_id:
                                # Create a separate assistant message for each tool call
                                single_tool_msg = cast(dict[str, Any], message.copy())
                                single_tool_msg["tool_calls"] = [tool_call]
                                tool_call_messages[tool_id] = (
                                    i,
                                    cast(ChatCompletionMessageParam, single_tool_msg),
                                )

            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id:
                    tool_result_messages[tool_call_id] = (i, message)
                else:
                    other_messages.append((i, message))
            else:
                other_messages.append((i, message))

        # First, identify which tool results will be paired to avoid duplicates
        paired_tool_result_indices = set()
        for tool_id in tool_call_messages:
            if tool_id in tool_result_messages:
                tool_result_idx, _ = tool_result_messages[tool_id]
                paired_tool_result_indices.add(tool_result_idx)

        # Create the fixed message sequence
        fixed_messages: list[ChatCompletionMessageParam] = []
        used_indices = set()

        # Add messages in their original order, but ensure tool_use → tool_result pairing
        for i, original_message in enumerate(messages):
            if i in used_indices:
                continue

            if not isinstance(original_message, dict):
                fixed_messages.append(original_message)
                used_indices.add(i)
                continue

            role = original_message.get("role")

            if role == "assistant" and original_message.get("tool_calls"):
                # Process each tool call in this assistant message
                tool_calls = original_message.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_id = tool_call.get("id")
                            if (
                                tool_id
                                and tool_id in tool_call_messages
                                and tool_id in tool_result_messages
                            ):
                                # Add tool_use → tool_result pair
                                _, tool_call_msg = tool_call_messages[tool_id]
                                tool_result_idx, tool_result_msg = tool_result_messages[tool_id]

                                fixed_messages.append(tool_call_msg)
                                fixed_messages.append(tool_result_msg)

                                # Mark both as used
                                used_indices.add(tool_call_messages[tool_id][0])
                                used_indices.add(tool_result_idx)
                            elif tool_id and tool_id in tool_call_messages:
                                # Tool call without result - add just the tool call
                                _, tool_call_msg = tool_call_messages[tool_id]
                                fixed_messages.append(tool_call_msg)
                                used_indices.add(tool_call_messages[tool_id][0])

                used_indices.add(i)  # Mark original multi-tool message as used

            elif role == "tool":
                # Only preserve unmatched tool results to avoid duplicates
                if i not in paired_tool_result_indices:
                    fixed_messages.append(original_message)
                used_indices.add(i)

            else:
                # Regular message - add it normally
                fixed_messages.append(original_message)
                used_indices.add(i)

        return fixed_messages

    def _log_request_to_file(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> None:
        """Log the converted messages and tools to a JSON file for debugging.

        Creates a logs directory if it doesn't exist and writes each request
        to a timestamped file.
        """
        try:
            # Create logs directory in the current working directory
            log_dir = Path("logs/anthropic_litellm")
            log_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file = log_dir / f"request_{timestamp}.json"

            # Prepare log data
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "messages": messages,
                "tools": tools,
            }

            # Write to file
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Logged request to {log_file}")
        except Exception as e:
            logger.warning(f"Failed to log request to file: {e}")

    def _remove_not_given(self, value: Any) -> Any:
        if value is omit or isinstance(value, NotGiven):
            return None
        return value

    def _merge_headers(self, model_settings: ModelSettings):
        return {**HEADERS, **(model_settings.extra_headers or {}), **(HEADERS_OVERRIDE.get() or {})}
