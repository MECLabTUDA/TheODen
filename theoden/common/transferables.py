from __future__ import annotations

from docstring_parser import parse

from dataclasses import dataclass
import yaml
from functools import partial
import inspect
from typing import TypeVar, Generic, Type, Literal, get_origin, get_args, Union

from .singleton import SingletonMeta
from .utils import are_classes_from_same_package, hash_dict

T = TypeVar("T")


class Transferable:
    _original_init: callable
    name: str = __name__
    metadata: None = None
    implemented: type[Transferable] | None = None
    base_type: type[Transferable] | None = None
    return_super_class_dict: bool = False

    def __init_subclass__(
        cls: type[Transferable],
        is_base_type: bool = False,
        base_type: type[Transferable] | None = None,
        implements: type[Transferable] | None = None,
        return_super_class_dict: bool = False,
        **kwargs,
    ):
        """Include the class in the list of transferables. Also overwrite the __init__ method to make the class transferable.

        Args:
            is_base_type (bool, optional): Whether the class is a base type. Defaults to False.
            base_type (Transferable, optional): The base type of the class. Defaults to None.
            implements (Transferable, optional): The interface that the class implements. Defaults to None.
            build (bool, optional): Whether to build the class. Defaults to True.
            return_super_class_dict (bool, optional): Whether to return the dict of the super class. Defaults to False.
            **kwargs: Additional keyword arguments.
        """

        super().__init_subclass__()

        # add the class to the list of transferables
        Transferables().add_transferable(
            cls,
            is_base_type=is_base_type,
            base_type=base_type,
            implements=implements,
            return_super_class_dict=return_super_class_dict,
            **kwargs,
        )

        """
        The class will be modified to be transferable. This means that the class will be gain a dict 
        function to convert the init parameters into a dict such that the class can be reconstructed from
        the dict. Furthermore, the class will gain a init_from_dict function to reconstruct the class from a 
        dict. This is necessary for the transfer of the class over the network.
        """

        # save the original __init__ method
        _original_init = cls.__init__

        # create a new __init__ method
        def _init(
            self: Transferable,
            *args,
            _additional_kwarg_keys: list[str] | None = None,
            **obj_kwargs,
        ):
            """Overwrite the __init__ method to make the class transferable by saving the init parameters.

            Args:
                *args: The positional arguments.
                _additional_kwarg_keys (list[str], optional): Additional keyword arguments. Defaults to None.
                **obj_kwargs: The keyword arguments.

            Raises:
                TypeError: If the class is not initialized with the correct arguments.
            """

            is_super_class = hasattr(self, "_dict_params")

            _dict_params = getattr(self, "_dict_params", None)
            _datatype = getattr(self, "datatype", type(self).__name__)

            if not is_super_class:
                self._super_initialized = not cls.return_super_class_dict

            # only for the class __init__ method and not for the super class __init__ method
            if not is_super_class or not self._super_initialized:
                args_as_kwargs = {}

                # check if or_init has __code__ attribute
                if hasattr(_original_init, "__code__"):
                    # zop variable names and non keyword args
                    zipped_args = zip(
                        _original_init.__code__.co_varnames[1 : len(args) + 1],
                        args,
                    )
                    # convert to dict
                    args_as_kwargs = dict(zipped_args)

                # remove _additional_kwarg_keys from kwargs
                local_kwargs = {
                    key: arg
                    for key, arg in obj_kwargs.items()
                    if _additional_kwarg_keys is None
                    or key not in _additional_kwarg_keys
                }
                # create the _init_params attribute by merging the kwargs and the args
                _dict_params = local_kwargs | args_as_kwargs
                _datatype = cls.__name__

                if is_super_class:
                    self._super_initialized = True

            self._dict_params = _dict_params
            self.datatype = _datatype
            _original_init(self, *args, **obj_kwargs)

        cls.__init__ = _init

    @staticmethod
    def make_transferable(
        cls: type,
        is_base_type=False,
        base_type: Transferable | None = None,
        implements: type[Transferable] | None = None,
        return_super_class_dict: bool = False,
        **kwargs,
    ) -> type[Transferable]:
        """Make a class transferable.

        Args:
            cls (type): class to make transferable
            is_base_type (bool, optional): whether the class is a base type. Defaults to False.
            base_type (Transferable | None, optional): base type of the class. Defaults to None.
            implements (type[Transferable] | None, optional): other transferable the class implements. Defaults to None.
            return_super_class_dict (bool, optional): whether to return the dict of the super class. Defaults to False.
            **kwargs: additional kwargs to pass to the init function

        Returns:
            Transferable: transferable class
        """

        # this will raise an error if a class only inherits from object, this it will not be transferable
        if cls.__bases__ == (object,):
            return cls

        cls.__bases__ += (Transferable,)
        cls.__init_subclass__(
            is_base_type=is_base_type,
            base_type=base_type,
            implements=implements,
            return_super_class_dict=return_super_class_dict,
            **kwargs,
        )

        return cls

    @classmethod
    def init_from_dict(
        cls, obj_dict: dict[str, any], **additional_kwargs
    ) -> Transferable:
        """Initialize the object from a dict.

        Args:
            obj_dict (dict[str, any]): dict representation of the object with the init parameters
            additional_kwargs (dict[str, any]): additional kwargs to pass to the init function

        Returns:
            Transferable: initialized object
        """

        return cls(
            **{
                key: dict_to_object(arg)
                for key, arg in obj_dict["data"].items()
                if type_check(cls, arg)
            },
            **additional_kwargs,
            _additional_kwarg_keys=additional_kwargs.keys(),
        )

    @classmethod
    def init_from_yaml(cls, file_path: str, **additional_kwargs) -> Transferable:
        """Initialize the object from a yaml file.

        Args:
            file_path (str): path to the yaml file
            additional_kwargs (dict[str, any]): additional kwargs to pass to the init function

        Returns:
            Transferable: initialized object
        """

        with open(file_path, "r") as f:
            return cls.init_from_dict(
                yaml.load(f, Loader=yaml.FullLoader), **additional_kwargs
            )

    def dict(self, exclude_keys: list[str] | None = None) -> dict[str, any]:
        """Returns a dict representation of the object with the init parameters.

        Args:
            exclude_keys (list[str] | None, optional): keys to exclude from the dict. Defaults to None.

        Returns:
            dict[str, any]: dict representation of the object with the init parameters
        """
        return {
            "datatype": self.datatype,
            "data": {
                k: object_to_dict(v)
                for k, v in self._dict_params.items()
                if not exclude_keys or k not in exclude_keys
            },
        }

    def yaml(self, file_path: str, exclude_keys: list[str] | None = None) -> None:
        """Saves the object as a yaml file.

        Args:
            file_path (str): path to the yaml file
        """
        with open(file_path, "w") as f:
            yaml.dump(self.dict(exclude_keys=exclude_keys), f)

    def init_after_deserialization(self) -> Transferable:
        """Initialize the object if it has not been initialized yet.

        This method should remain empty as it is being defined by the Transferable class

        Returns:
            Transferable: initialized object
        """
        return self

    def initialization_hash(self, exclude_keys: list[str] | None = None) -> str:
        """Returns a hash of the initialization parameters of the object.

        Args:
            exclude_keys (list[str] | None, optional): keys to exclude from the hash. Defaults to None.

        Returns:
            str: hash of the initialization parameters of the object
        """

        return hash_dict(self.dict(exclude_keys=exclude_keys))

    def add_initialization_parameter(
        self, _overwrite: bool = False, **kwargs
    ) -> Transferable:
        for key in kwargs:
            if key in self._dict_params and not _overwrite:
                raise KeyError(f"Key {key} already in _init_params")
            self._dict_params[key] = kwargs[key]
        return self

    def remove_initialization_parameter(self, *keys) -> Transferable:
        for key in keys:
            if key in self._dict_params:
                del self._dict_params[key]
        return self

    @classmethod
    def info(cls, include_metadata: bool = True) -> dict:
        """Returns a dict with information about the class.

        Args:
            include_metadata (bool, optional): whether to include the metadata in the dict. Defaults to True.

        Returns:
            dict: dict with information about the class
        """

        return {
            "name": cls.name,
            "datatype": cls,
            "implemented": cls.implemented,
            "base_type": cls.base_type,
        } | ({"metadata": cls.metadata} if include_metadata else {})


@dataclass
class RegisteredObject:
    """A registered object.

    Args:
        name (str): name of the object
        datatype (type): type of the object
        metadata (any): metadata of the object
        base_type (type | None, optional): base type of the object. Defaults to None.
        implemented (type | None, optional): implemented type of the object. Defaults to None.
    """

    name: str
    datatype: type[Transferable]
    metadata: any
    base_type: type[Transferable] | None = None
    implemented: type[Transferable] | None = None


class Transferables(metaclass=SingletonMeta):
    _transferables: dict[str, type[Transferable]] = {}
    _allow_overwrite = False

    def add_transferable(
        self,
        cls: type[Transferable],
        is_base_type=False,
        base_type: Transferable | None = None,
        implements: type[Transferable] | None = None,
        return_super_class_dict: bool = False,
        **kwargs,
    ) -> None:
        """Add a transferable class to the transferables.

        Args:
            cls (type[Transferable]): class to add
            is_base_type (bool, optional): whether the class is a base type. Defaults to False.
            base_type (Transferable | None, optional): base type of the class. Defaults to None.
            implements (type[Transferable] | None, optional): other transferable the class implements. Defaults to None.
            **kwargs: additional kwargs to pass to the init function

        Raises:
            KeyError: if the transferable key already exists and overwriting is not allowed
            ValueError: if the class is external and the base type is not specified or if the class is external and the base type is not a base type
        """

        # Check if the transferable key already exists, overwriting is not allowed and the class does not implement a class.
        if cls.__name__ in self and not self._allow_overwrite and not implements:
            raise KeyError(
                f"The key {cls.__name__} already exists and overwriting is not allowed."
            )

        # If the class implements another class, add the implemented class to the object's implementation.
        if implements:
            self[implements].implemented = cls
        else:
            """
            This part will insert the class into the transferables. There it can be accessed by the class name.
            Different parameters ike an alternative implementation and the base type are saved as well as the
            class metadata.
            """

            # check ig the class is from this package or external
            is_internal = are_classes_from_same_package(type(self), cls)

            # if the class is external, check if the base type is specified
            if not is_internal and not is_base_type and base_type is None:
                raise ValueError(
                    f"The transferable class `{cls.__name__}` has no Theoden base type. Please specify the base type."
                )

            # find base type of the class e.g. for an Augmentation this would be the Augmentation class
            if is_base_type:
                if base_type is not None:
                    raise ValueError(
                        f"The transferable class `{cls.__name__}` is specified as a base type and has a base type specified. "
                        "Please specify only one."
                    )
                base_type_ = None
            else:
                if base_type is None:
                    # check if the base type is specified based on the class hierarchy
                    base_types = self.base_types
                    for class_ in cls.__mro__:
                        if class_ in base_types:
                            base_type_ = class_
                            break
                    else:
                        raise ValueError(
                            f"The transferable class `{cls.__name__}` has no Theoden base type. Please specify the base type."
                        )
                else:
                    base_type_ = base_type

            # insert the class into the transferables
            cls.name = cls.__name__
            cls.base_type = base_type_
            cls.implemented = implements
            cls.return_super_class_dict = return_super_class_dict
            cls.metadata = ClassArgumentMetadata(name=cls.__name__, type_=cls, **kwargs)
            self[cls] = cls

    def __getitem__(self, item: type | str) -> type[Transferable]:
        """Get a transferable by its name or type.

        Args:
            item (type | str): name or type of the transferable

        Returns:
            Transferable: transferable
        """
        return self._transferables[item.__name__ if isinstance(item, type) else item]

    def __setitem__(self, key: type | str, value: type[Transferable]):
        """Set a transferable by its name or type.

        Args:
            key (type | str): name or type of the transferable
            value (Transferable): transferable
        """
        self._transferables[key.__name__ if isinstance(key, type) else key] = value

    def __contains__(self, item: type | str):
        """Check if a transferable is registered.

        Args:
            item (type | str): name or type of the transferable

        Returns:
            bool: whether the transferable is registered
        """
        return (
            item.__name__ if isinstance(item, type) else item
        ) in self._transferables

    @property
    def base_types(self) -> list[type]:
        """Returns a list of all base types.

        Returns:
            list[type]: list of all base types
        """
        return [v for v in self._transferables.values() if v.base_type is None]

    def to_object(
        self,
        json_data: dict,
        base_type: type[T] | str | None = None,
        **additional_kwargs: any,
    ) -> T | Transferable:
        """Converts a dict to an object.

        Args:
            json_data (dict): dict to convert
            base_type (str | None, optional): base type of the object. Defaults to None.
            **additional_kwargs (any): additional keyword arguments to pass to the object's init method

        Raises:
            TypeError: if the base type does not match the object's base type
        """
        base_type_ = self[json_data["datatype"]].base_type
        if base_type_ is None:
            # if it is a base type, the base type is the class itself
            base_type_ = self[json_data["datatype"]]

        if base_type is not None and base_type_ not in Transferables():
            raise ValueError(f"Base type {base_type_} is not registered.")

        if base_type is not None and (
            base_type_.__name__ != base_type
            if isinstance(base_type, str)
            else base_type_ != base_type
        ):
            raise TypeError(
                f"Expected object of with base_type '{base_type.__name__ if isinstance(base_type, type) else base_type}', got '{base_type_.__name__}'."
            )

        datatype_string: str = json_data["datatype"]
        if datatype_string not in Transferables():
            raise ValueError(f"Datatype {datatype_string} is not registered.")

        datatype = Transferables()[datatype_string]

        to_be_created = datatype if not datatype.implemented else datatype.implemented

        return to_be_created.init_from_dict(
            json_data, **additional_kwargs
        ).init_after_deserialization()

    def overview(
        self,
        group_by_base: bool = True,
        of_type: type | str | None = None,
        include_metadata: bool = False,
    ):
        """Returns an overview of all transferables.

        Args:
            group_by_base (bool, optional): If True, the transferables are grouped by their base type. Defaults to True.
            of_type (type | str | None, optional): If specified, only transferables of the specified type are returned. Defaults to None.
            include_metadata (bool, optional): If True, the metadata of the transferables is included. Defaults to False.

        Returns:
            dict: The overview of the transferables.
        """

        if not group_by_base:
            return {
                k: v.info(include_metadata=include_metadata)
                for k, v in self._transferables.items()
                if of_type is None or v.base_type == of_type
            }

        groups: dict[str, list[type[Transferable]]] = {}
        for obj in self._transferables.values():
            base_type_ = obj.base_type.__name__ if obj.base_type is not None else "BASE"
            if of_type is None or base_type_ == (
                of_type.__name__ if isinstance(of_type, type) else of_type
            ):
                if base_type_ not in groups:
                    groups[base_type_] = []
                groups[base_type_].append(obj.info(include_metadata=include_metadata))
        return groups


class ClassArgumentMetadata(dict):
    def __init__(
        self,
        name: str,
        *,
        type_: type[Transferable],
        _ignore_params: tuple[str] = ("self", "kwargs"),
        **kwargs,
    ) -> None:
        """Creates a new ClassArgumentMetadata object.

        Args:
            name (str): name of the class
            type_ (type): type of the class
            _ignore_params (tuple[str], optional): parameters to ignore. Defaults to ("self", "kwargs").
        """

        super().__init__(name=name)
        self.update(kwargs)
        self._ignore_params = _ignore_params
        self._parse_init_types_and_docstring(type_)

    @classmethod
    def getClassAndIgnoreParams(cls, ignore_params: list[str]) -> callable:
        return partial(cls, _ignore_params=ignore_params)

    def _parse_init_types_and_docstring(self, type_: type[Transferable]) -> None:
        """Parses the init method of the class and extracts the types of the arguments and the docstring.

        Args:
            type_ (Transferable): the class to parse
        """

        # Get the signature of the __init__ method of the dataset class
        signature = inspect.signature(type_.__init__)
        required_args = {}
        optional_args = {}

        if inspect.getdoc(type_.__init__):
            parsed_docstring = parse(inspect.getdoc(type_.__init__))
            self["description"] = parsed_docstring.short_description
            self["long_description"] = parsed_docstring.long_description
        else:
            parsed_docstring = None

        for param_name, param in signature.parameters.items():
            if param_name in self._ignore_params:
                continue

            if param.default == inspect.Parameter.empty:
                required_args[param_name] = type_to_dict(param.annotation)
            else:
                optional_args[param_name] = type_to_dict(param.annotation)

            # find the parameter in the docstring
            if parsed_docstring:
                for param in parsed_docstring.params:
                    if param.arg_name == param_name:
                        if param_name in required_args:
                            required_args[param_name]["description"] = param.description
                        elif param_name in optional_args:
                            optional_args[param_name]["description"] = param.description
                        break

        self["required_args"] = required_args
        self["optional_args"] = optional_args


def object_to_dict(obj) -> dict[str, any]:
    """Converts an object to a dictionary.

    Args:
        obj (Any): The object to be converted to a dictionary.

    Returns:
        Dict[str, Any]: A dictionary representation of the object.

    Raises:
        TypeError: If the object is not of a supported type.

    """
    _dict = {"datatype": type(obj).__name__}

    # For each supported data type, add the corresponding value to the dictionary
    if isinstance(obj, (int, float, str, bool)):
        _dict["value"] = obj
    elif obj is None:
        _dict["value"] = "none"
    elif isinstance(obj, (list, tuple)):
        _dict["value"] = [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        _dict["value"] = {key: object_to_dict(value) for key, value in obj.items()}
    elif type(obj) is type and obj in Transferables():
        _dict["value"] = obj.__name__
        _dict["datatype"] = "registered_type"
    # If the object is of a registered type, call the 'dict' method and add the result to the dictionary
    elif type(obj) in Transferables():
        _json = obj.dict()
        _dict["datatype"] = _json["datatype"]
        _dict["value"] = _json

    # If the object is not of a supported type, raise an exception
    else:
        raise TypeError(
            f"{type(obj).__name__} is not a registered class or in [int, float, str, bool, None, list, tuple, dict]"
        )

    return _dict


def dict_to_object(
    object_dict: dict,
) -> int | float | str | bool | None | list | tuple | dict | Transferable:
    """
    This function takes a dictionary as input, where the keys represent variable names, and the
    values contain both the value and the data type of the variable. The function then converts
    the dictionary to an object with the correct types.

    Args:
        object_dict (dict): The dictionary to be converted to an object.

    Returns:
        object: An object with the correct types.

    Raises:
        TypeError: If the data type specified in the dictionary is not supported.
    """

    # Check if input is a dictionary
    if isinstance(object_dict, dict):
        # Check if dictionary contains both "value" and "type" keys
        if "value" in object_dict and "datatype" in object_dict:
            # Retrieve the value and type from the dictionary
            value = object_dict["value"]
            type_name = object_dict["datatype"]

            # Check the data type specified in the dictionary and convert to the correct type
            if type_name == "int":
                return int(value)
            elif type_name == "float":
                return float(value)
            elif type_name == "str":
                return str(value)
            elif type_name == "bool":
                return bool(value)
            elif type_name == "NoneType":
                return None
            elif type_name == "list":
                return [dict_to_object(item) for item in value]
            elif type_name == "tuple":
                return tuple([dict_to_object(item) for item in value])
            elif type_name == "dict":
                return {key: dict_to_object(value) for key, value in value.items()}
            elif type_name == "registered_type":
                return Transferables()[value]
            # Check if the data type is registered and convert using the registered type
            elif type_name in Transferables():
                return Transferables().to_object(value).init_after_deserialization()
            else:
                # Raise TypeError if the data type is not supported
                raise TypeError("Unsupported data type: {}".format(type_name))
        else:
            raise TypeError("Not a valid json type")


def type_to_dict(type_: type) -> dict:
    """
    Convert a Python type to a dictionary representation.

    Args:
        type_: The type to convert.

    Returns:
        A dictionary representing the input type.

    """
    if type_ == bool:
        return {"type": "bool"}
    elif type_ == str:
        return {"type": "str"}
    elif type_ == int:
        return {"type": "int"}
    elif type_ == float:
        return {"type": "float"}
    elif type_ == any:
        return {"type": "any"}
    elif get_origin(type_) is None:
        return {"type": "none"}
    elif get_origin(type_) is Literal:
        # For a Literal type, convert the arguments to a list of dictionaries
        return {
            "type": "literal",
            "args": [arg for arg in get_args(type_)],
        }
    elif get_origin(type_) is Union:
        # For a Literal type, convert the arguments to a list of dictionaries
        return {
            "type": "union",
            "args": [type_to_dict(arg) for arg in get_args(type_)],
        }
    elif get_origin(type_) is Range:
        # For a Range type, create a dictionary with the base type and optional lower and upper limits
        _dict = {
            "type": "range",
            "base_type": type_to_dict(get_args(type_)[0]),
        }
        if get_origin(get_args(type_)[1]) is Literal:
            _dict["lower"] = get_args(type_)[1].__args__[0]
        if get_origin(get_args(type_)[2]) is Literal:
            _dict["upper"] = get_args(type_)[2].__args__[0]
        return _dict
    elif get_origin(type_) is list:
        # For a list type, convert the arguments to a list of dictionaries
        return {"type": "list", "args": [type_to_dict(arg) for arg in get_args(type_)]}
    elif get_origin(type_) is tuple:
        # For a tuple type, convert the arguments to a list of dictionaries
        return {"type": "tuple", "args": [type_to_dict(arg) for arg in get_args(type_)]}
    else:
        # For all other types, return a dictionary with the type name
        return {"type": str(get_origin(type_))}


T = TypeVar("T", Type, None)
U = TypeVar("U", int, float)
V = TypeVar("V", int, float)


class Range(Generic[T, U, V]):
    def __init__(self, type_: Type[T], lower: U, upper: V):
        self.type_ = type_
        self.lower = lower
        self.upper = upper


def type_check(cls, data) -> bool:
    if True:
        return True
    else:
        raise TypeError("Type must match type hints")
