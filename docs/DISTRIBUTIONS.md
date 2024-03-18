# Distribution

A Distribution is a class that manages the distribution of commands to the clients. 

```python
class InstructionStatus(IntEnum):
    CREATED = auto()  # The Instruction object has been created
    BOOTING = auto()  # The Instruction object is initializing
    EXECUTION = auto()  # The Instruction object is executing
    EXECUTION_FINISHED = auto()  # The Instruction object has finished executing
    COMPLETED = auto()  # The Instruction object has completed its execution cycle
```

```python	
class CommandDistributionStatus(IntEnum):
    EXCLUDED = 1
    UNREQUESTED = 2
    SEND = 3
    STARTED = 4
    WAIT_FOR_RESPONSE = 5
    FINISHED = 6
    FAILED = 7
```

## Example

```python
ClosedDistribution(    
    command=PrintResourceKeysCommand(),
    selector=AllClientsSelector(),
)
```
