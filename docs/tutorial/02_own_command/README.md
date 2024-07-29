# Create own commands

This tutorial will show you how to create your own commands.

## Prerequisites

Look at [Cifar10 Tutorial](../01_cifar10_run) to understand the basic concepts.

## Create a basic command with only an execute method

The minimum requirement for a command is to have an execute method.
Create a class that inherits from Command and implement the execute method.
Have a look at the [new_command.py](./new_command.py) file.

A command can return a response that is sent to the server but you an also return None. In this example return the number of model parameters as a metric.
The command can access the resource manager of the client using `self.client_rm`.
 
For a more detailed explanation of the command class, have a look at the [COMMANDS](../../COMMANDS.md) documentation.

Start server and client like in the [Cifar10 Tutorial](../01_cifar10_run).



## Create a more complex command with additional methods

In this example we count the number of model parameters and add them up on the server. herefore we need more command methods.