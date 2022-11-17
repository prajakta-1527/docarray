from abc import abstractmethod
from typing import Iterable, Type

from docarray.document import BaseDocument


class AbstractDocumentArray(Iterable):

    document_type: Type[BaseDocument]

    @abstractmethod
    def __init__(self, docs: Iterable[BaseDocument]):
        ...

    @abstractmethod
    def __class_getitem__(
        cls, item: Type[BaseDocument]
    ) -> Type['AbstractDocumentArray']:
        ...