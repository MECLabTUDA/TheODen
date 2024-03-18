from theoden import start_server

from theoden.operations import *

from new_action import NewAction, NewActionThatExitsAfterwardsAction

if __name__ == "__main__":
    instructions = [
        NewAction(),
        ClosedDistribution(PrintResourceKeysCommand()),
    ]

    start_server(
        run_name="05_new_action",
        instructions=instructions,
        permanent_conditions=[RequireNumberOfClientsCondition(2)],
        exit_on_finish=True,
        global_context="../tutorial_context.yaml",
    )
