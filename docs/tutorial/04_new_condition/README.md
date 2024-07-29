# New condition

As shown in [new_condition.py](./new_condition.py) you can create a new condition by creating a new class that inherits from `Condition` and implementing the `resolved` method.

Import it on the server and you can use it. Here we wait until the last digit ot the current time is 5.
