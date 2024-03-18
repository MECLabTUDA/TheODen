from theoden import start_server

from theoden.operations import *

from new_condition import LastDigitOfSecondsIsNumberCondition

if __name__ == "__main__":
    instructions = [
        LastDigitOfSecondsIsNumberCondition(5),
        ClosedDistribution(PrintResourceKeysCommand()),
    ]

    start_server(
        run_name="04_new_condition",
        instructions=instructions,
        permanent_conditions=[RequireNumberOfClientsCondition(2)],
        exit_on_finish=True,
        global_context="../tutorial_context.yaml",
    )
