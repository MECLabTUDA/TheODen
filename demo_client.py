from theoden.topology.node import Node
from theoden.common import GlobalContext

# set up global context. This is a singleton object that is used to store global variables like paths to datasets.
GlobalContext().load_from_yaml("demo_context.yaml")
# create a node with a ping interval of 0.5 seconds
node = Node(ping_interval=0.5)
# start the node
node.start()
