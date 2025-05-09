class UnauthorizedError(Exception):
    def __init__(self, message="Unauthorized"):
        self.message = message


class ForbiddenOperationError(Exception):
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


class AggregationError(Exception):
    def __init__(self, message="Aggregation Error"):
        self.message = message


class NoCommandException(Exception):
    def __init__(self, message="No Command Exception"):
        self.message = message


class TooManyCommandsExecutingException(Exception):
    def __init__(self, message="Too Many Commands Excuting Exception"):
        self.message = message


class ClientConfigurationError(Exception):
    def __init__(self, message="Client Configuration Error"):
        self.message = message
