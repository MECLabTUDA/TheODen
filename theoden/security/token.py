from datetime import datetime, timedelta
from jose import JWTError, jwt, ExpiredSignatureError
from secrets import token_hex

from ..common import UnauthorizedError


# create a secret key for signing the JWT
SECRET_KEY = token_hex(32)

# Define the algorithm used to sign the JWT
ALGORITHM = "HS256"

# Define the expiration time of the JWT (in minutes)
ACCESS_TOKEN_EXPIRE_DAYS = 3


# Define a function to create a new JWT access token
def create_access_token(
    data: dict, delta: int = ACCESS_TOKEN_EXPIRE_DAYS, unit: str = "days"
):
    """Create a new JWT access token.

    Args:
        data (dict): The data to encode in the token.
        delta (int, optional): The time delta for the token to expire. Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.
        unit (str, optional): The unit of the time delta. Defaults to "minutes".

    Returns:
        str: The encoded JWT access token.
    """
    to_encode = data.copy()
    now = datetime.utcnow()
    expire = datetime.utcnow() + timedelta(**{unit: delta})
    if delta == 0:
        expire = datetime.utcnow() + timedelta(**{"weeks": 10000})
    to_encode.update({"exp": expire, "iat": now})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> str:
    """Decode a JWT access token.

    Args:
        token (str): The JWT access token to decode.

    Returns:
        str: The value encoded in the token.

    Raises:
        UnauthorizedError: If the token is invalid.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        value = payload.get("sub")
        if value is None:
            raise UnauthorizedError("Invalid authentication credentials")
    except ExpiredSignatureError:
        raise UnauthorizedError("Token has expired")
    except JWTError:
        raise UnauthorizedError("Invalid authentication credentials")
    return value
