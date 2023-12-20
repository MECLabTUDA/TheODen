import io
import tempfile

import numpy as np
import torch

from ...common import Transferable


class StateLoader(Transferable, is_base_type=True):
    @staticmethod
    def save(model_dict: dict) -> bytes:
        """Saves the model dictionary to bytes.

        Args:
            model_dict (dict): The model dictionary.

        Returns:
            bytes: The model dictionary as bytes.
        """
        pass

    @staticmethod
    def load(model_bytes: bytes) -> dict:
        """Loads the model dictionary from bytes.

        Args:
            model_bytes (bytes): The model dictionary as bytes.

        Returns:
            dict: The model dictionary.
        """
        pass


class NumpyStateLoader(StateLoader):
    @staticmethod
    def save(model_dict: dict) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, model_dict)
        return buffer.getvalue()

    @staticmethod
    def load(model_bytes: bytes) -> dict:
        return np.load(io.BytesIO(model_bytes), allow_pickle=True).item()


class TorchStateLoader(StateLoader):
    @staticmethod
    def save(model_dict: dict) -> bytes:
        buffer = io.BytesIO()
        torch.save(model_dict, buffer)
        return buffer.getvalue()

    @staticmethod
    def load(model_bytes: bytes) -> dict:
        return torch.load(io.BytesIO(model_bytes))


class TensorflowStateLoader(StateLoader):
    @staticmethod
    def save(model_dict: dict) -> bytes:
        import tensorflow as tf

        buffer = io.BytesIO()
        # Save the model using the SavedModel format
        tf.saved_model.save(model_dict, buffer)
        return buffer.getvalue()

    @staticmethod
    def load(model_bytes: bytes):
        import pickle

        import tensorflow as tf

        buffer = io.BytesIO(model_bytes)
        # Deserialize the bytes into a dictionary
        loaded_model_dict = pickle.loads(buffer.read())
        return loaded_model_dict


# Function to suppress TensorFlow warnings
def suppress_tf_warnings():
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TensorflowLiteV1StateLoader(StateLoader):
    @staticmethod
    def save(model_dict: dict) -> bytes:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import os

        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model_checkpoint.ckpt")

            # Create a list of tensor names and a list of tensor values
            tensor_names = list(model_dict.keys())
            tensor_values = [tf.constant(value) for value in model_dict.values()]

            # Use tf.raw_ops.Save to save tensors to the checkpoint file
            tf.raw_ops.Save(
                filename=checkpoint_path,
                tensor_names=tensor_names,
                data=tensor_values,
                name="save",
            )

            # Read the contents of the temporary checkpoint file into bytes
            with open(checkpoint_path, "rb") as temp_file:
                checkpoint_bytes = temp_file.read()

        return checkpoint_bytes

    @staticmethod
    def load(model_bytes: bytes):
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(model_bytes)
            temp_file_path = temp_file.name

            # Create a regular Python dictionary to store variable names and their values
            variables_dict = {}

            # List all variable names in the checkpoint using TensorFlow 1.x compatible API
            with tf.compat.v1.Session() as sess:
                reader = tf.compat.v1.train.NewCheckpointReader(temp_file_path)
                var_names = reader.get_variable_to_shape_map().keys()

                for var_name in var_names:
                    variables_dict[var_name] = np.array(reader.get_tensor(var_name))

        return variables_dict
