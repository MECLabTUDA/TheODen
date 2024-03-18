# Transferable(s): Python -> JSON and JSON -> Python interface

## Example

```python   
from theoden import Transferables
from theoden.operations import TrainRoundCommand

command = TrainRoundCommand(communication_round=1, num_steps=10)
cmd_dict = command.dict()
print(cmd_dict)
# {'datatype': 'TrainRoundCommand', 'data': {'communication_round': {'datatype': 'int', 'value': 1}, 'num_steps': {'datatype': 'int', 'value': 10}}}

cmd = TrainRoundCommand.init_from_dict(cmd_dict)
print(cmd)
# <theoden.operations.commands.action.train.TrainRoundCommand object at 0x7fbbd4db7400>
print(cmd.dict())
# {'datatype': 'TrainRoundCommand', 'data': {'communication_round': {'datatype': 'int', 'value': 1}, 'num_steps': {'datatype': 'int', 'value': 10}}}

cmd = Transferables().to_object(cmd_dict)
print(cmd)
# <theoden.operations.commands.action.train.TrainRoundCommand object at 0x7fbbd4db71c0>
print(cmd.dict())
# {'datatype': 'TrainRoundCommand', 'data': {'communication_round': {'datatype': 'int', 'value': 1}, 'num_steps': {'datatype': 'int', 'value': 10}}}
```

## Create Transferables

```python
from theoden import Transferable, Transferables


class MyNewBaseTransferable(Transferable, is_base_type=True):
    def __init__(self, name:str, age:int):
        self.name = name
        self.age = age
:
class AnotherTransferable(MyNewBaseTransferable):
    def __init__(self, name:str, age:int, another_value:int):
        super().__init__(name, age)
        self.another_value = another_value
```

| Parameter               |      type  - default      | Function                                                                                                                  |
| ----------------------- | :-----------------------: | ------------------------------------------------------------------------------------------------------------------------- |
| is_base_type            |      boolean - False      | If True, this class will be a new base class. A base class should be a new type of Class e.g. there is Command base type. |
| base_type               | type[Transferable] - None | Define the base type class                                                                                                |
| implements              | type[Transferable] - None | If a transferable gets implemented by other classes, these classes are created instead                                    |
| return_super_class_dict |      boolean - False      | Return the dictionary of the super class. Useful for Sequentioal Meta Commands                                            |
