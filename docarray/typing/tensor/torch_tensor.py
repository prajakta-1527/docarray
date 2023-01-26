from copy import copy
from typing import TYPE_CHECKING, Any, Dict, Generic, Type, TypeVar, Union, cast

import numpy as np
import torch  # type: ignore

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from pydantic.fields import ModelField
    from pydantic import BaseConfig
    import numpy as np
    from docarray.proto import NdArrayProto
    from docarray.computation.torch_backend import TorchCompBackend

from docarray.base_document.base_node import BaseNode

T = TypeVar('T', bound='TorchTensor')
ShapeT = TypeVar('ShapeT')

torch_base: type = type(torch.Tensor)
node_base: type = type(BaseNode)


# the mypy error suppression below should not be necessary anymore once the following
# is released in mypy: https://github.com/python/mypy/pull/14135
class metaTorchAndNode(
    AbstractTensor.__parametrized_meta__,  # type: ignore
    torch_base,  # type: ignore
    node_base,  # type: ignore
):  # type: ignore
    pass


@_register_proto(proto_type_name='torch_tensor')
class TorchTensor(
    torch.Tensor, AbstractTensor, Generic[ShapeT], metaclass=metaTorchAndNode
):
    # Subclassing torch.Tensor following the advice from here:
    # https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
    """
    Subclass of torch.Tensor, intended for use in a Document.
    This enables (de)serialization from/to protobuf and json, data validation,
    and coersion from compatible types like numpy.ndarray.

    This type can also be used in a parametrized way,
    specifying the shape of the tensor.

    EXAMPLE USAGE

    .. code-block:: python

        from docarray import BaseDocument
        from docarray.typing import TorchTensor
        import torch


        class MyDoc(BaseDocument):
            tensor: TorchTensor
            image_tensor: TorchTensor[3, 224, 224]
            square_crop: TorchTensor[3, 'x', 'x']


        # create a document with tensors
        doc = MyDoc(
            tensor=torch.zeros(128),
            image_tensor=torch.zeros(3, 224, 224),
            square_crop=torch.zeros(3, 64, 64),
        )

        # automatic shape conversion
        doc = MyDoc(
            tensor=torch.zeros(128),
            image_tensor=torch.zeros(224, 224, 3),  # will reshape to (3, 224, 224)
            square_crop=torch.zeros(3, 128, 128),
        )

        # !! The following will raise an error due to shape mismatch !!
        doc = MyDoc(
            tensor=torch.zeros(128),
            image_tensor=torch.zeros(224, 224),  # this will fail validation
            square_crop=torch.zeros(3, 128, 64),  # this will also fail validation
        )
    """

    __parametrized_meta__ = metaTorchAndNode

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, TorchTensor):
            return cast(T, value)
        elif isinstance(value, torch.Tensor):
            return cls._docarray_from_native(value)

        else:
            try:
                arr: torch.Tensor = torch.tensor(value)
                return cls._docarray_from_native(arr)
            except Exception:
                pass  # handled below
        raise ValueError(f'Expected a torch.Tensor compatible type, got {type(value)}')

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        # this is needed to dump to json
        field_schema.update(type='string', format='tensor')

    def _docarray_to_json_compatible(self) -> np.ndarray:
        """
        Convert torchTensor into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.numpy()  ## might need to  check device later

    def unwrap(self) -> torch.Tensor:
        """
        Return the original torch.Tensor without any memory copy.

        The original view rest intact and is still a Document TorchTensor
        but the return object is a pure torch Tensor but both object share
        the same memory layout.

        EXAMPLE USAGE
        .. code-block:: python
            from docarray.typing import TorchTensor
            import torch

            t = TorchTensor.validate(torch.zeros(3, 224, 224), None, None)
            # here t is a docarray TorchTensor
            t2 = t.unwrap()
            # here t2 is a pure torch.Tensor but t1 is still a Docarray TorchTensor
            # But both share the same underlying memory


        :return: a torch Tensor
        """
        value = copy(self)  # as unintuitive as it sounds, this
        # does not do any relevant memory copying, just shallow
        # reference to the torch data
        value.__class__ = torch.Tensor  # type: ignore
        return value

    @classmethod
    def _docarray_from_native(cls: Type[T], value: torch.Tensor) -> T:
        """Create a TorchTensor from a native torch.Tensor

        :param value: the native torch.Tensor
        :return: a TorchTensor
        """
        if cls.__unparametrizedcls__:  # This is not None if the tensor is parametrized
            value.__class__ = cls.__unparametrizedcls__
        else:
            value.__class__ = cls
        return cast(T, value)

    @classmethod
    def from_ndarray(cls: Type[T], value: np.ndarray) -> T:
        """Create a TorchTensor from a numpy array

        :param value: the numpy array
        :return: a TorchTensor
        """
        return cls._docarray_from_native(torch.from_numpy(value))

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'NdArrayProto') -> 'T':
        """
        read ndarray from a proto msg
        :param pb_msg:
        :return: a torch tensor
        """
        source = pb_msg.dense
        if source.buffer:
            x = np.frombuffer(bytearray(source.buffer), dtype=source.dtype)
            return cls.from_ndarray(x.reshape(source.shape))
        elif len(source.shape) > 0:
            return cls.from_ndarray(np.zeros(source.shape))
        else:
            raise ValueError(f'proto message {pb_msg} cannot be cast to a TorchTensor')

    def to_protobuf(self) -> 'NdArrayProto':
        """
        transform self into a NdArrayProto protobuf message
        """
        from docarray.proto import NdArrayProto

        nd_proto = NdArrayProto()

        value_np = self.detach().cpu().numpy()
        nd_proto.dense.buffer = value_np.tobytes()
        nd_proto.dense.ClearField('shape')
        nd_proto.dense.shape.extend(list(value_np.shape))
        nd_proto.dense.dtype = value_np.dtype.str

        return nd_proto

    @staticmethod
    def get_comp_backend() -> 'TorchCompBackend':
        """Return the computational backend of the tensor"""
        from docarray.computation.torch_backend import TorchCompBackend

        return TorchCompBackend()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # this tells torch to treat all of our custom tensors just like
        # torch.Tensor's. Otherwise, torch will complain that it doesn't
        # know how to handle our custom tensor type.
        docarray_torch_tensors = TorchTensor.__subclasses__()
        types_ = tuple(
            torch.Tensor if t in docarray_torch_tensors else t for t in types
        )
        return super().__torch_function__(func, types_, args, kwargs)