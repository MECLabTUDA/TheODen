from theoden import start_client

#### Import the new command ########
from new_command import (
    NumberOfModelParametersCommand,
    CalculateSumOfModelParametersCommand,
)

import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_client(global_context="../tutorial_context.yaml")
