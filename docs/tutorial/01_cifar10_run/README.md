# Create a Cifar10 demo server and run 3 clients

This tutorial will show you how to create a Cifar10 server and run 3 clients.

## Prerequisites

Before you start, make sure you have installed the required packages and also installed theoden.

```bash
pip install git+https://github.com/MECLabTUDA/TheODen.git
```

## Step 1: Create a Cifar10 server

First, we need to create a Cifar10 server. Navigate in this folder and run the following command:

```bash
python server.py
```

This will start a server that will wait for clients to connect.
The instructions define what the clients and the server should do during a run.

First we set the dataset partitions, the model, the optimizer and other necessary training resources.
Then will initialize the global model by taking one of the clients as the global model. Finally, we train for multiple federated rounds.

Hava close look at the [server.py](./server.py) file to understand the server implementation.
Also, look at other documents in the [docs](../../) folder to understand the server and client implementation.

## Step 2: Create Cifar10 clients

Now, we need to create 3 clients. Navigate in this folder and run the following command in 3 different terminals:

```bash
python client.py
```

This will start a client that will connect to the server and train the model.

Hava close look at the [client.py](./client.py) file to understand the client implementation.
Also, look at other documents in the [docs](../../) folder to understand the server and client implementation.