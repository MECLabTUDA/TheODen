from theoden import start_server

from theoden.operations import *
from theoden.models import TimmModel

#### Import the new command #####
from new_command import (
    NumberOfModelParametersCommand,
    CalculateSumOfModelParametersCommand,
)


if __name__ == "__main__":
    instructions = [
        ClosedDistribution(
            SequentialCommand(
                [
                    InitModelCommand(
                        TimmModel(
                            "resnet18", num_classes=10, pretrained=False, in_chans=3
                        )
                    ),
                    CalculateSumOfModelParametersCommand(),
                ]
            )
        ),
    ]

    start_server(
        run_name="02_own_command",
        instructions=instructions,
        permanent_conditions=[RequireNumberOfClientsCondition(2)],
        exit_on_finish=True,
        global_context="../tutorial_context.yaml",
    )
