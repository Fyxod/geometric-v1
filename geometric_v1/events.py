from __future__ import annotations

from typing import Any, Callable


EventCallback = Callable[[dict[str, Any]], None]
StopCallback = Callable[[], bool]


def emit_event(event_callback: EventCallback | None, event_type: str, **payload: Any) -> None:
    if event_callback is None:
        return
    event_callback({"type": event_type, **payload})


def is_stop_requested(stop_requested: StopCallback | None) -> bool:
    return bool(stop_requested and stop_requested())


def with_event_context(
    event_callback: EventCallback | None,
    **context: Any,
) -> EventCallback | None:
    if event_callback is None:
        return None

    def emit(event: dict[str, Any]) -> None:
        merged = dict(context)
        merged.update(event)
        event_callback(merged)

    return emit
