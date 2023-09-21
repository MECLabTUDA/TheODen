import pytest
from pathlib import Path
import sys, os

sys.path.insert(0, Path(os.path.dirname(__file__)).parent.parent.as_posix())

from theoden.resources import ResourceRegister


def test_register_resource():
    # Arrange
    resource_register = ResourceRegister()
    key = "my_resource"
    resource = "my_resource_value"

    # Act
    registered_resource = resource_register.sr(key, resource)

    # Assert
    assert key in resource_register
    assert resource_register[key] == resource
    assert registered_resource == resource


def test_register_subresource():
    # Arrange
    resource_register = ResourceRegister()
    key = "subregister:subresource"
    resource = "subresource_value"

    # Act
    registered_resource = resource_register.sr(key, resource)

    # Assert
    assert "subregister" in resource_register
    assert "subresource" in resource_register["subregister"]
    assert resource_register["subregister"]["subresource"] == resource
    assert registered_resource == resource
    assert isinstance(resource_register["subregister"], ResourceRegister)


def test_register_subresource_twice_with_overwrite():
    # Arrange
    resource_register = ResourceRegister()
    key = "subregister:subresource"
    resource = "subresource_value"
    resource2 = "new_subresource_value"

    # Act
    registered_resource = resource_register.sr(key, resource)
    registered_resource2 = resource_register.sr(key, resource2)

    # Assert
    assert "subregister" in resource_register
    assert "subresource" in resource_register["subregister"]
    assert resource_register["subregister"]["subresource"] == resource2
    assert registered_resource2 == resource2


def test_register_subresource_twice_without_overwrite():
    # Arrange
    resource_register = ResourceRegister()
    key = "subregister:subresource"
    resource = "subresource_value"
    resource2 = "new_subresource_value"

    # Act
    registered_resource = resource_register.sr(key, resource)
    with pytest.raises(KeyError):
        registered_resource2 = resource_register.sr(key, resource2, overwrite=False)

    # Assert
    assert "subregister" in resource_register
    assert "subresource" in resource_register["subregister"]
    assert resource_register["subregister"]["subresource"] == resource
    assert registered_resource == resource


def test_assert_resource_type():
    # Arrange
    resource_register = ResourceRegister()
    key = "my_resource"
    resource = 42

    # Act
    with pytest.raises(AssertionError):
        resource_register.sr(key, resource, assert_type=str)

    # Assert
    assert key not in resource_register


def test_assert_resource_type_hint():
    # Arrange
    resource_register = ResourceRegister()
    key = "my_resource"
    key2 = "my_resource2"
    resource = [42, 123]

    # Act
    with pytest.raises(AssertionError):
        resource_register.sr(key, resource, assert_type=list[str])
    resource_register.sr(key2, resource, assert_type=list[int])

    # Assert
    assert key not in resource_register
    assert key2 in resource_register


def test_get_resource():
    # Arrange
    resource_register = ResourceRegister()
    key = "my_resource"
    resource = "my_resource_value"
    resource_register.sr(key, resource)

    # Act
    registered_resource = resource_register.gr(key)

    # Assert
    assert registered_resource == resource


def test_get_subresource():
    # Arrange
    resource_register = ResourceRegister()
    key = "subregister:subresource"
    resource = "subresource_value"
    resource_register.sr(key, resource)

    # Act
    registered_resource = resource_register.gr(key)

    # Assert
    assert registered_resource == resource


def test_get_resource_with_type():
    # Arrange
    resource_register = ResourceRegister()
    key = "my_resource"
    resource = "my_resource_value"
    resource_register.sr(key, resource)

    # Act
    with pytest.raises(AssertionError):
        registered_resource = resource_register.gr(key, int)
    registered_resource = resource_register.gr(key, str)

    # Assert
    assert registered_resource == resource


def test_get_resource_with_default():
    # Arrange
    resource_register = ResourceRegister()
    key = "my_resource"
    resource = "my_resource_value"
    resource_register.sr(key, resource)

    # Act
    registered_resource = resource_register.gr(key, default="default")
    default_resource = resource_register.gr("not_existing", default="default")

    # Assert
    assert registered_resource == resource
    assert default_resource == "default"
