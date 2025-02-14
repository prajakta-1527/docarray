from __future__ import annotations

from typing import Any, Dict, Optional

from docarray import BaseDoc, DocList
from docarray.typing import AnyEmbedding, AnyTensor


class LegacyDocument(BaseDoc):
    """
    This Document is the LegacyDocument. It follows the same schema as in DocList v1.
    It can be useful to start migrating a codebase from v1 to v2.

    Nevertheless, the API is not totally compatible with DocAray v1 `Document`.
    Indeed, none of the method associated with `Document` are present. Only the schema
    of the data is similar.

    ```python
    from docarray import DocList
    from docarray.documents.legacy import LegacyDocument
    import numpy as np

    doc = LegacyDocument(text='hello')
    doc.url = 'http://myimg.png'
    doc.tensor = np.zeros((3, 224, 224))
    doc.embedding = np.zeros((100, 1))

    doc.tags['price'] = 10

    doc.chunks = DocList[Document]([Document() for _ in range(10)])

    doc.chunks = DocList[Document]([Document() for _ in range(10)])
    ```

    """

    tensor: Optional[AnyTensor]
    chunks: Optional[DocList[LegacyDocument]]
    matches: Optional[DocList[LegacyDocument]]
    blob: Optional[bytes]
    text: Optional[str]
    url: Optional[str]
    embedding: Optional[AnyEmbedding]
    tags: Dict[str, Any] = dict()
    scores: Optional[Dict[str, Any]]
