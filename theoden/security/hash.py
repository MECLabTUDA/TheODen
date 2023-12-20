from passlib.context import CryptContext

# Define a password context to securely hash passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_value(value: str | bytes) -> str:
    """Hash a value.

    Args:
        value (str | bytes): The value to hash.

    Returns:
        str: The hashed value.
    """
    return pwd_context.hash(value)


def verify_hash(plain: str | bytes, hashed: str | bytes) -> bool:
    """Verify a hashed value against a plain value.

    Args:
        plain (str | bytes): The plain value to verify.
        hashed (str | bytes): The hashed value to verify against.

    Returns:
        bool: True if the hashed value matches the plain value.
    """
    return pwd_context.verify(plain, hashed)
