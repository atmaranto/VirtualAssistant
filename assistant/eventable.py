from typing import Callable, Any, Dict, List

class Eventable:
    def __init__(self) -> None:
        self._event_handlers: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, event_name: str, handler: Callable[..., Any]) -> None:
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                handler(*args, **kwargs)