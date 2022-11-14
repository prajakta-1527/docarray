from abc import abstractmethod
from typing import Iterable, Type

from docarray.document import BaseDocument
from docarray.document.abstract_document import AbstractDocument


class AbstractDocumentArray(Iterable):

    document_type: Type[BaseDocument]

    @abstractmethod
    def __init__(self, docs: Iterable[AbstractDocument]):
        ...