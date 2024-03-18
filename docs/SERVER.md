# Server

The server holds all information regarding the operations that will be carried out during one training run.

In general the server has only two major functions: [process_server_request](../theoden/topology/server.py) and [process_status_update](../theoden/topology/server.py). 

How the status updates and [Serverrequests](./SERVERREQUESTS.md) can be sent to the server and how the server responds is handled by the communication interface of the server. 

## Communication Interface

### Rest Interface

The rest interface opens a rest interface that can be used to send server requests and status updates to the server. The server will respond with the appropriate response.
It can be secured with OAuth2 and HTTPS. If you plan to do local testing, you should use this interface.

### RabbitMQ Interface

The RabbitMQ interface connects to a RabbitMQ server and listens to a specific queue. It will process the messages and respond with the appropriate response. This interface is useful if the clients cannot directly address the server but a rabbitmq instance as a proxy.

## Operations Manager

The operation manager hold the different Instructions and Conditions for the training. As instructions there are `Distributions` and `Actions`.
[Distributions](./DISTRIBUTIONS.md) are used to distribute [Commands](./COMMANDS.md) to the clients while [Actions](./ACTIONS.md) are functionalities that are being executed on the server.

The operation manager will infer the commands fo thr requesting clients and will also handle the status updates from the clients. It will start actions on the server and temporarily stops, if a [Condition](./CONDITIONS.md) is not met.

The code for the operation manager can be found [here](../theoden/topology/manager.py).
## Watcher Pool
To monitor the training process, the server has a pool of watchers. Watchers can react to specific events that trigger notifications.
A server has some basic watchers but the user can easily implement and add own watchers that fit to their needs. Use the watcher parameter in the `start_server` function to add a watcher to the server.

More information about the watcher can be found [here](./WATCHER.md).

## Usage

Please refer to the [Cifar10 tutorial](./tutorial/01_cifar10_run/) to create a Cifar10 server and run 3 clients.
The start methods can be found [here](../theoden/start.py). The server class can be found [here](../theoden/topology/server.py).