from __future__ import annotations

from collections.abc import Iterable, Iterator

from moduagent.tools.base import Tool, ToolSchema


class ToolRegistry:
    """Ordered registry of uniquely named tools."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._frozen = False
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if self._frozen:
            raise RuntimeError("tool registry is frozen")
        name = str(tool.name)
        if not name.strip():
            raise ValueError("tool name cannot be empty")
        if name in self._tools and not replace:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool
        return tool

    def unregister(self, name: str) -> Tool:
        if self._frozen:
            raise RuntimeError("tool registry is frozen")
        try:
            return self._tools.pop(name)
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return tool

    def schemas(self, names: Iterable[str] | None = None) -> tuple[ToolSchema, ...]:
        """Return schemas in registration order, optionally restricted by name."""

        if names is None:
            return tuple(tool.schema for tool in self._tools.values())
        selected = frozenset(str(name) for name in names)
        unknown = selected.difference(self._tools)
        if unknown:
            missing = ", ".join(sorted(unknown))
            raise KeyError(f"unknown tools: {missing}")
        return tuple(
            tool.schema for name, tool in self._tools.items() if name in selected
        )

    @property
    def schema_list(self) -> tuple[ToolSchema, ...]:
        return self.schemas()

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> ToolRegistry:
        """Seal the composition against post-validation Tool replacement."""

        self._frozen = True
        return self

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self.require(name)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
