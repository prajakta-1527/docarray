import importlib
from typing import Callable, List, NamedTuple, Optional, Type, Union

import torch  # TODO(johannes) this breaks the optional import of torch

from docarray import Document, DocumentArray
from docarray.typing import Tensor
from docarray.typing.tensor import type_to_framework

# but will be fixed once we have a computational backend


class FindResult(NamedTuple):
    documents: DocumentArray
    scores: Tensor


def find(
    index: DocumentArray,
    query: Union[Tensor, Document],
    embedding_field: str = 'embedding',
    metric: str = 'cosine_sim',
    limit: int = 10,
    device: Optional[str] = None,
    descending: Optional[bool] = None,
) -> FindResult:
    """
    Find the closest Documents in the index to the query.
    Supports PyTorch and NumPy embeddings.

    .. note::
        This utility function is likely to be removed once
        Document Stores are available.
        At that point, and in-memory Document Store will serve the same purpose
        by exposing a .find() method.

    .. note::
        This is a simple implementation that assumes the same embedding field name for
        both query and index, does not support nested search, and does not support
        hybrid (multi-vector) search. These shortcoming will be addressed in future
        versions.

    EXAMPLE USAGE

    .. code-block:: python

        from docarray import DocumentArray, Document
        from docarray.typing import TorchTensor
        from docarray.utility.find import find


        class MyDocument(Document):
            embedding: TorchTensor


        index = DocumentArray[MyDocument](
            [MyDocument(embedding=torch.rand(128)) for _ in range(100)]
        )

        # use Document as query
        query = MyDocument(embedding=torch.rand(128))
        top_matches, scores = find(
            index=index,
            query=query,
            embedding_field='tensor',
            metric='cosine_sim',
        )

        # use tensor as query
        query = torch.rand(128)
        top_matches, scores = find(
            index=index,
            query=query,
            embedding_field='tensor',
            metric='cosine_sim',
        )

    :param index: the index of Documents to search in
    :param query: the query to search for
    :param embedding_field: the tensor-like field in the index to use
        for the similarity computation
    :param metric: the distance metric to use for the similarity computation.
        Can be one of the following strings:
        'cosine_sim' for cosine similarity, 'euclidean_dist' for euclidean distance,
        'sqeuclidean_dist' for squared euclidean distance
    :param limit: return the top `limit` results
    :param device: the computational device to use,
        can be either `cpu` or a `cuda` device.
    :param descending: sort the results in descending order.
        Per default, this is chosen based on the `metric` argument.
    :return: A named tuple of the form (DocumentArray, Tensor),
        where the first element contains the closes matches for the query,
        and the second element contains the corresponding scores.
    """
    query = _extract_embedding_single(query, embedding_field)
    return find_batched(
        index=index,
        query=query,
        embedding_field=embedding_field,
        metric=metric,
        limit=limit,
        device=device,
        descending=descending,
    )[0]


def find_batched(
    index: DocumentArray,
    query: Union[Tensor, DocumentArray],
    embedding_field: str = 'embedding',
    metric: str = 'cosine_sim',
    limit: int = 10,
    device: Optional[str] = None,
    descending: Optional[bool] = None,
) -> List[FindResult]:
    """
    Find the closest Documents in the index to the queries.
    Supports PyTorch and NumPy embeddings.

    .. note::
        This utility function is likely to be removed once
        Document Stores are available.
        At that point, and in-memory Document Store will serve the same purpose
        by exposing a .find() method.

    .. note::
        This is a simple implementation that assumes the same embedding field name for
        both query and index, does not support nested search, and does not support
        hybrid (multi-vector) search. These shortcoming will be addressed in future
        versions.

        EXAMPLE USAGE

    .. code-block:: python

        from docarray import DocumentArray, Document
        from docarray.typing import TorchTensor
        from docarray.utility.find import find


        class MyDocument(Document):
            embedding: TorchTensor


        index = DocumentArray[MyDocument](
            [MyDocument(embedding=torch.rand(128)) for _ in range(100)]
        )

        # use DocumentArray as query
        query = DocumentArray[MyDocument](
            [MyDocument(embedding=torch.rand(128)) for _ in range(3)]
        )
        results = find(
            index=index,
            query=query,
            embedding_field='tensor',
            metric='cosine_sim',
        )
        top_matches, scores = results[0]

        # use tensor as query
        query = torch.rand(3, 128)
        results, scores = find(
            index=index,
            query=query,
            embedding_field='tensor',
            metric='cosine_sim',
        )
        top_matches, scores = results[0]

    :param index: the index of Documents to search in
    :param query: the query to search for
    :param embedding_field: the tensor-like field in the index to use
        for the similarity computation
    :param metric: the distance metric to use for the similarity computation.
        Can be one of the following strings:
        'cosine_sim' for cosine similarity, 'euclidean_dist' for euclidean distance,
        'sqeuclidean_dist' for squared euclidean distance
    :param limit: return the top `limit` results
    :param device: the computational device to use,
        can be either `cpu` or a `cuda` device.
    :param descending: sort the results in descending order.
        Per default, this is chosen based on the `metric` argument.
    :return: a list of named tuples of the form (DocumentArray, Tensor),
        where the first element contains the closes matches for each query,
        and the second element contains the corresponding scores.
    """
    if descending is None:
        descending = metric.endswith('_sim')  # similarity metrics are descending

    embedding_type = _da_attr_type(index, embedding_field)

    # get framework-specific distance and top_k function
    metric_fn = _get_metric_fn(embedding_type, metric)
    top_k_fn = _get_topk_fn(embedding_type)

    # extract embeddings from query and index
    index_embeddings = _extraxt_embeddings(index, embedding_field, embedding_type)
    query_embeddings = _extraxt_embeddings(query, embedding_field, embedding_type)

    # compute distances and return top results
    dists = metric_fn(query_embeddings, index_embeddings, device=device)
    top_scores, top_indices = top_k_fn(
        dists, k=limit, device=device, descending=descending
    )

    index_doc_type = index.document_type
    results = []
    for indices_per_query, scores_per_query in zip(top_indices, top_scores):
        docs_per_query = DocumentArray[index_doc_type]([])  # type: ignore
        for idx in indices_per_query:  # workaround until #930 is fixed
            docs_per_query.append(index[idx])
        results.append(FindResult(scores=scores_per_query, documents=docs_per_query))
    return results


def _extract_embedding_single(
    data: Union[DocumentArray, Document, Tensor],
    embedding_field: str,
) -> Tensor:
    """Extract the embeddings from a single query,
    and return it in a batched representation.

    :param data: the data
    :param embedding_field: the embedding field
    :param embedding_type: type of the embedding: torch.Tensor, numpy.ndarray etc.
    :return: the embeddings
    """
    if isinstance(data, Document):
        emb = getattr(data, embedding_field)
    else:  # treat data as tensor
        emb = data
    if len(emb.shape) == 1:
        # TODO(johannes) solve this with computational backend,
        #  this is ugly hack for now
        if isinstance(emb, torch.Tensor):
            emb = emb.unsqueeze(0)
        else:
            import numpy as np

            if isinstance(emb, np.ndarray):
                emb = np.expand_dims(emb, axis=0)
    return emb


def _extraxt_embeddings(
    data: Union[DocumentArray, Document, Tensor],
    embedding_field: str,
    embedding_type: Type,
) -> Tensor:
    """Extract the embeddings from the data.

    :param data: the data
    :param embedding_field: the embedding field
    :param embedding_type: type of the embedding: torch.Tensor, numpy.ndarray etc.
    :return: the embeddings
    """
    # TODO(johannes) put docarray stack in the computational backend
    if isinstance(data, DocumentArray):
        emb = getattr(data, embedding_field)
        if not data.is_stacked():
            emb = embedding_type.__docarray_stack__(emb)
    elif isinstance(data, Document):
        emb = getattr(data, embedding_field)
    else:  # treat data as tensor
        emb = data

    if len(emb.shape) == 1:
        # TODO(johannes) solve this with computational backend,
        #  this is ugly hack for now
        if isinstance(emb, torch.Tensor):
            emb = emb.unsqueeze(0)
        else:
            import numpy as np

            if isinstance(emb, np.ndarray):
                emb = np.expand_dims(emb, axis=0)
    return emb


def _da_attr_type(da: DocumentArray, attr: str) -> Type:
    """Get the type of the attribute according to the Document type
    (schema) of the DocumentArray.

    :param da: the DocumentArray
    :param attr: the attribute name
    :return: the type of the attribute
    """
    return da.document_type.__fields__[attr].type_


def _get_topk_fn(embedding_type: Type) -> Callable:
    """Dynamically import the distance function from the framework-specific module.
    This will go away once we have a computational backend.

    :param embedding_type: the type of the embedding
    :param distance_name: the name of the distance function
    :return: the framework-specific distance function
    """
    framework = type_to_framework[embedding_type]
    return getattr(
        importlib.import_module(f'docarray.utility.helper.{framework}'),
        'top_k',
    )


def _get_metric_fn(embedding_type: Type, metric: Union[str, Callable]) -> Callable:
    """Dynamically import the distance function from the framework-specific module.
    This will go away once we have a proper computational backend.

    :param embedding_type: the type of the embedding
    :param metric: the name of the metric, or the metric itself
    :return: the framework-specific metric
    """
    if callable(metric):
        return metric
    framework = type_to_framework[embedding_type]
    return getattr(
        importlib.import_module(f'docarray.utility.math.metrics.{framework}'),
        f'{metric}',
    )