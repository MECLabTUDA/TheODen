from theoden import start_server

from theoden.operations import *

if __name__ == "__main__":
    instructions = [
        ClosedDistribution(
            SequentialCommand([PrintResourceKeysCommand(), ABCInitModelCommand()])
        ),
    ]

    start_server(
        run_name="03_overwrite_commands",
        instructions=instructions,
        permanent_conditions=[RequireNumberOfClientsCondition(2)],
        exit_on_finish=True,
        global_context="../tutorial_context.yaml",
    )
