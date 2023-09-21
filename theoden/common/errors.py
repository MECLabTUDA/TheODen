class UnauthorizedError(Exception):
    def __init__(self, message="Unauthorized"):
        self.message = message


class ForbiddenCommandError(Exception):
    def __init__(self, message="Forbidden"):
        self.message = message


class NotFoundError(Exception):
    def __init__(self, message="Not Found"):
        self.message = message


class ConflictError(Exception):
    def __init__(self, message="Conflict"):
        self.message = message


class RequestDeniedError(Exception):
    def __init__(self, message="Request Denied"):
        self.message = message


class InvalidRequestError(Exception):
    def __init__(self, message="Invalid Request"):
        self.message = message


class TopologyError(Exception):
    def __init__(self, message="Topology Error"):
        self.message = message


class NotImplementedAbstractCommandError(Exception):
    def __init__(self, message="Not Implemented Abstract Command"):
        self.message = message


class ServerRequestError(Exception):
    def __init__(self, message="Server Request Error"):
        self.message = message
