from enum import IntEnum, auto


class NodeCommandStatus(IntEnum):
    EXCLUDED = auto()
    UNREQUESTED = auto()
    SEND = auto()
    STARTED = auto()
    WAIT_FOR_RESPONSE = auto()
    FINISHED = auto()
    FAILED = auto()
