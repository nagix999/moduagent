from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, create_model

from moduagent.tools.base import (
    ToolExecutionContext,
    ToolSchema,
    _await_if_needed,
    _run_sync_in_daemon,
)


Function = Callable[..., Any]
F = TypeVar("F", bound=Function)


class FunctionTool:
    """Adapt a typed Python function to the Tool contract."""

    def __init__(
        self,
        function: Function,
        *,
        name: str | None = None,
        description: str | None = None,
        input_model: type[BaseModel] | None = None,
        args_schema: type[BaseModel] | None = None,
        idempotent: bool = False,
        timeout_seconds: float | None = None,
        max_result_bytes: int | None = None,
    ) -> None:
        if input_model is not None and args_schema is not None:
            raise ValueError("use either input_model or args_schema, not both")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_result_bytes is not None and max_result_bytes < 1:
            raise ValueError("max_result_bytes must be at least 1")

        self.function = function
        self.name = name or function.__name__
        if not self.name.strip():
            raise ValueError("tool name cannot be empty")
        self.description = description or inspect.getdoc(function) or self.name
        self.idempotent = idempotent
        self.timeout_seconds = timeout_seconds
        self.max_result_bytes = max_result_bytes
        self._context_parameter: str | None = None
        explicit_model = input_model or args_schema
        if explicit_model is not None:
            if not isinstance(explicit_model, type) or not issubclass(
                explicit_model, BaseModel
            ):
                raise TypeError("input_model must be a Pydantic BaseModel class")
            self._context_parameter = self._find_context_parameter(function)
            self.input_model = explicit_model
        else:
            self.input_model = self._argument_model(function)
        self.args_schema = self.input_model
        self._schema = ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )

    @staticmethod
    def _find_context_parameter(function: Function) -> str | None:
        try:
            hints = get_type_hints(function, include_extras=True)
        except (NameError, TypeError):
            hints = dict(getattr(function, "__annotations__", {}))
        matches = [
            parameter.name
            for parameter in inspect.signature(function).parameters.values()
            if hints.get(parameter.name, parameter.annotation) is ToolExecutionContext
        ]
        if len(matches) > 1:
            raise TypeError("a tool may accept only one ToolExecutionContext")
        return matches[0] if matches else None

    def _argument_model(self, function: Function) -> type[BaseModel]:
        try:
            hints = get_type_hints(function, include_extras=True)
        except (NameError, TypeError):
            hints = dict(getattr(function, "__annotations__", {}))

        fields: dict[str, tuple[Any, Any]] = {}
        for parameter in inspect.signature(function).parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError("*args and **kwargs are not supported")
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError("positional-only tool arguments are not supported")

            annotation = hints.get(parameter.name, parameter.annotation)
            if annotation is inspect.Signature.empty:
                annotation = Any
            if annotation is ToolExecutionContext:
                if self._context_parameter is not None:
                    raise TypeError("a tool may accept only one ToolExecutionContext")
                self._context_parameter = parameter.name
                continue
            default = (
                ...
                if parameter.default is inspect.Signature.empty
                else parameter.default
            )
            fields[parameter.name] = (annotation, default)

        return create_model(
            f"{function.__name__.title()}Arguments",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = self.input_model.model_validate(dict(arguments))
        return {name: getattr(validated, name) for name in type(validated).model_fields}

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any:
        return await self.invoke_validated(self.validate_arguments(arguments), context)

    async def invoke_validated(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any:
        values = dict(arguments)
        if self._context_parameter is not None:
            values[self._context_parameter] = (
                context if context is not None else ToolExecutionContext()
            )

        if inspect.iscoroutinefunction(self.function):
            return await self.function(**values)
        result = await _run_sync_in_daemon(lambda: self.function(**values))
        return await _await_if_needed(result)


@overload
def function_tool(function: F, /) -> FunctionTool: ...


@overload
def function_tool(
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
    args_schema: type[BaseModel] | None = None,
    idempotent: bool = False,
    timeout_seconds: float | None = None,
    max_result_bytes: int | None = None,
) -> Callable[[F], FunctionTool]: ...


def function_tool(
    function: F | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
    args_schema: type[BaseModel] | None = None,
    idempotent: bool = False,
    timeout_seconds: float | None = None,
    max_result_bytes: int | None = None,
) -> FunctionTool | Callable[[F], FunctionTool]:
    """Create a FunctionTool, usable both directly and as a decorator."""

    def decorate(target: F) -> FunctionTool:
        return FunctionTool(
            target,
            name=name,
            description=description,
            input_model=input_model,
            args_schema=args_schema,
            idempotent=idempotent,
            timeout_seconds=timeout_seconds,
            max_result_bytes=max_result_bytes,
        )

    if function is None:
        return decorate
    return decorate(function)
