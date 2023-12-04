from enum import IntEnum, auto


class NodeCommandStatus(IntEnum):
    EXCLUDED = auto()
    UNREQUESTED = auto()
    SEND = auto()
    STARTED = auto()
    WAIT_FOR_RESPONSE = auto()
    FINISHED = auto()
    FAILED = auto()


class CommandExecutionStatus(IntEnum):
    EXCLUDED = 1
    STARTED = 4
    FINISHED = 6
    FAILED = 7


class CommandDistributionStatus(IntEnum):
    EXCLUDED = 1
    UNREQUESTED = 2
    SEND = 3
    STARTED = 4
    WAIT_FOR_RESPONSE = 5
    FINISHED = 6
    FAILED = 7
