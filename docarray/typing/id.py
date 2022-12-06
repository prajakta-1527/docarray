from typing import Type, TypeVar, Union
from uuid import UUID
from pydantic import BaseConfig, parse_obj_as
from pydantic.fields import ModelField

from docarray.proto import NodeProto
from docarray.typing.abstract_type import AbstractType

T = TypeVar('T', bound='ID')


class ID(str, AbstractType):
    """
    Represent an unique ID
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, int, UUID],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:

        try:
            id: str = str(value)
            return cls(id)
        except Exception:
            raise ValueError(f'Expected a str, int or UUID, got {type(value)}')

    def _to_node_protobuf(self) -> NodeProto:
        """Convert an ID into a NodeProto message. This function should
        be called when the self is nested into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(id=self)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'str') -> T:
        """
        read ndarray from a proto msg
        :param pb_msg:
        :return: a string
        """
        return parse_obj_as(cls, pb_msg)