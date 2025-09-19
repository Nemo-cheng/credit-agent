"""事件系统模块
处理模块间通信的事件总线
"""
from typing import Dict, Any, List, Callable
from enum import Enum, auto
from dataclasses import dataclass

class EventType(Enum):
    DATA_COLLECTED = auto()
    DATA_ANALYZED = auto()
    RISK_ANALYZED = auto()
    REPORT_GENERATED = auto()
    FEEDBACK_RECEIVED = auto()

@dataclass
class Event:
    type: EventType
    data: Any
    source: str

class EventBus:
    """事件总线，处理模块间通信"""
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
    
    def subscribe(self, event_type: EventType, callback: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def publish(self, event: Event):
        for callback in self._subscribers.get(event.type, []):
            callback(event)