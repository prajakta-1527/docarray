__all__ = ['find', 'find_batched']

from typing import Any, Dict, List, NamedTuple, Optional, Type, Union, cast

from typing_inspect import is_union_type

from multiprocessing.pool import Pool, ThreadPool

from docarray.array.any_array import AnyDocArray
from docarray.array.doc_list.doc_list import DocList
from docarray.array.doc_vec.doc_vec import DocVec
from docarray.base_doc import BaseDoc
from docarray.helper import _get_field_type_by_access_path
from docarray.typing import AnyTensor
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils.map import _map_docs_batched_multiarg
from docarray.documents import TextDoc


class FindResult(NamedTuple):
    documents: DocList
    scores: AnyTensor


class _FindResult(NamedTuple):
    documents: Union[DocList, List[Dict[str, Any]]]
    scores: AnyTensor


class FindResultBatched(NamedTuple):
    documents: List[DocList]
    scores: List[AnyTensor]


class _FindResultBatched(NamedTuple):
    documents: Union[List[DocList], List[List[Dict[str, Any]]]]
    scores: List[AnyTensor]


def find(
    index: AnyDocArray,
    query: Union[AnyTensor, BaseDoc],
    search_field: str = '',
    metric: str = 'cosine_sim',
    limit: int = 10,
    device: Optional[str] = None,
    descending: Optional[bool] = None,
) -> FindResult:
    """
    Find the closest Documents in the index to the query.
    Supports PyTorch and NumPy embeddings.

    !!! note
        This is a simple implementation of exact search. If you need to do advance
        search using approximate nearest neighbours search or hybrid search or
        multi vector search please take a look at the [`BaseDoc`][docarray.base_doc.doc.BaseDoc].

    ---

    ```python
    from docarray import DocList, BaseDoc
    from docarray.typing import TorchTensor
    from docarray.utils.find import find
    import torch

    class MyDocument(BaseDoc):
        embedding: TorchTensor


    index = DocList[MyDocument]([MyDocument(embedding=torch.rand(128)) for _ in range(100)])

    # use Document as query
    query = MyDocument(embedding=torch.rand(128))
    top_matches, scores = find(
        index=index,
        query=query,
        search_field='embedding',
        metric='cosine_sim',
    )


    # use tensor as query
    query = torch.rand(128)
    top_matches, scores = find(
        index=index,
        query=query,
        search_field='embedding',
        metric='cosine_sim',
    )
    ```

    :param index: the index of Documents to search in
    :param query: the query to search for
    :param search_field: the tensor-like field in the index to use
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
    :return: A named tuple of the form (DocList, AnyTensor),
        where the first element contains the closes matches for the query,
        and the second element contains the corresponding scores.
    """
    query = _extract_embedding_single(query, search_field)
    docs, scores = find_batched(
        index=index,
        query=query,
        search_field=search_field,
        metric=metric,
        limit=limit,
        device=device,
        descending=descending,
    )
    return FindResult(documents=docs[0], scores=scores[0])


def get_result(
    query_embeds,
    index_embeddings,
    device,
    comp_backend,
    search_field,
    embedding_type,
    metric_fn,
    limit,
    descending,
):
    q_embed = query_embeds
    dists = metric_fn(q_embed, index_embeddings, device=device)
    top_scores, top_indices = comp_backend.Retrieval.top_k(
        dists, k=limit, device=device, descending=descending
    )
    return top_indices, top_scores


def find_batched(
    index: AnyDocArray,
    query: Union[AnyTensor, DocList],
    batch_size: Optional[int] = None,
    search_field: str = '',
    metric: str = 'cosine_sim',
    limit: int = 10,
    device: Optional[str] = None,
    descending: Optional[bool] = None,
    shuffle: bool = False,
    backend: str = 'thread',
    num_worker: Optional[int] = None,
    pool: Optional[Union[Pool, ThreadPool]] = None,
    show_progress: bool = False,
) -> FindResultBatched:
    """
    Find the closest Documents in the index to the queries.
    Supports PyTorch and NumPy embeddings.

    !!! note
        This is a simple implementation of exact search. If you need to do advance
        search using approximate nearest neighbours search or hybrid search or
        multi vector search please take a look at the [`BaseDoc`][docarray.base_doc.doc.BaseDoc]


    ---

    ```python
    from docarray import DocList, BaseDoc
    from docarray.typing import TorchTensor
    from docarray.utils.find import find_batched
    import torch

    class MyDocument(BaseDoc):
        embedding: TorchTensor


    index = DocList[MyDocument]([MyDocument(embedding=torch.rand(128)) for _ in range(100)])

    # use DocList as query

    query = DocList[MyDocument]([MyDocument(embedding=torch.rand(128)) for _ in range(10)])
    results = find_batched(
        index=index,
        query=query,
        batch_size=5,
        search_field='embedding',
        metric='cosine_sim',
    )
    top_matches, scores = docs[0], scores[0]

    # use tensor as query
    query = torch.rand(10, 128)
    docs, scores = find_batched(
        index=index,
        query=query,
        batch_size=5,
        search_field='embedding',
        metric='cosine_sim',
    )
    top_matches, scores = docs[0], scores[0]


    :param index: the index of Documents to search in
    :param query: the query to search for
    :param search_field: the tensor-like field in the index to use
        for the similarity computation
    :param batch_size: Size of each generated batch (except the last one, which might
        be smaller).
    :param metric: the distance metric to use for the similarity computation.
        Can be one of the following strings:
        'cosine_sim' for cosine similarity, 'euclidean_dist' for euclidean distance,
        'sqeuclidean_dist' for squared euclidean distance
    :param limit: return the top `limit` results
    :param device: the computational device to use,
        can be either `cpu` or a `cuda` device.
    :param descending: sort the results in descending order.
        Per default, this is chosen based on the `metric` argument.
    :param shuffle: If set, shuffle the Documents before dividing into minibatches.
    :param backend: `thread` for multithreading and `process` for multiprocessing.
        Defaults to `thread`.
    :param num_worker: the number of parallel workers. If not given, then the number of CPUs
        in the system will be used.
    :param show_progress: show a progress bar
    :param pool: use an existing/external pool. If given, `backend` is ignored and you will
        be responsible for closing the pool.
    :return: A named tuple of the form (DocList, AnyTensor),
        where the first element contains the closest matches for each query,
        and the second element contains the corresponding scores.
    """

    if descending is None:
        descending = metric.endswith('_sim')  # similarity metrics are descending

    embedding_type = _da_attr_type(index, search_field)
    comp_backend = embedding_type.get_comp_backend()
    # extract embeddings from index
    index_embeddings = _extract_embeddings(index, search_field, embedding_type)
    metric_fn = getattr(comp_backend.Metrics, metric)

    func_args = {
        'index_embeddings': index_embeddings,
        'device': device,
        'comp_backend': comp_backend,
        'search_field': search_field,
        'embedding_type': embedding_type,
        'metric_fn': metric_fn,
        'limit': limit,
        'descending': descending,
    }

    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError(
                f'`batch_size` must be larger than 0, receiving {batch_size}'
            )
        else:
            batch_size = int(batch_size)
    else:
        query_embs = _extract_embeddings(query, search_field, embedding_type)
        top_indices, top_scores = get_result(query_embs, **func_args)
        batched_docs: List[DocList] = []
        scores = []
        for _, (indices_per_query, scores_per_query) in enumerate(
            zip(top_indices, top_scores)
        ):
            docs_per_query: DocList = DocList([])
            for idx in indices_per_query:  # workaround until #930 is fixed
                docs_per_query.append(index[idx])
            batched_docs.append(DocList(docs_per_query))
            scores.append(scores_per_query)
        return FindResultBatched(documents=batched_docs, scores=scores)

    if not (isinstance(query, DocList) or isinstance(query, (DocVec, BaseDoc))):
        query = cast(AnyTensor, query)
        d = DocList[TextDoc](TextDoc() for _ in range(comp_backend.shape(query)[0]))
        d.embedding = query  # type: ignore
        query = d

    it = _map_docs_batched_multiarg(
        docs=query,
        func=get_result,
        batch_size=batch_size,
        backend=backend,
        num_worker=num_worker,
        shuffle=shuffle,
        pool=pool,
        show_progress=show_progress,
        func_args=func_args,
    )
    retrieved_docs: List[DocList] = []
    docs_scores: List[AnyTensor] = []
    for indices_per_query, scores_per_query in it:
        for idx, scores in zip(indices_per_query, scores_per_query):
            per_query_docs: DocList = DocList([])
            for id in idx:  # workaround until #930 is fixed
                per_query_docs.append(index[id])
            retrieved_docs.append(DocList(per_query_docs))
            docs_scores.append(scores)  # type: ignore
    return FindResultBatched(documents=retrieved_docs, scores=docs_scores)


def _extract_embedding_single(
    data: Union[DocList, BaseDoc, AnyTensor],
    search_field: str,
) -> AnyTensor:
    """Extract the embeddings from a single query,
    and return it in a batched representation.

    :param data: the data
    :param search_field: the embedding field
    :param embedding_type: type of the embedding: torch.Tensor, numpy.ndarray etc.
    :return: the embeddings
    """
    if isinstance(data, BaseDoc):
        emb = next(AnyDocArray._traverse(data, search_field))
    else:  # treat data as tensor
        emb = data
    if len(emb.shape) == 1:
        # all currently supported frameworks provide `.reshape()`. Onc this is not true
        # anymore, we need to add a `.reshape()` method to the computational backend
        emb = emb.reshape(1, -1)
    return emb


def _extract_embeddings(
    data: Union[AnyDocArray, BaseDoc, AnyTensor],
    search_field: str,
    embedding_type: Type,
) -> AnyTensor:
    """Extract the embeddings from the data.

    :param data: the data
    :param search_field: the embedding field
    :param embedding_type: type of the embedding: torch.Tensor, numpy.ndarray etc.
    :return: the embeddings
    """
    emb: AnyTensor
    if isinstance(data, DocList):
        emb_list = list(AnyDocArray._traverse(data, search_field))
        emb = embedding_type._docarray_stack(emb_list)
    elif isinstance(data, (DocVec, BaseDoc)):
        emb = next(AnyDocArray._traverse(data, search_field))
    else:  # treat data as tensor
        emb = cast(AnyTensor, data)

    if len(emb.shape) == 1:
        emb = emb.get_comp_backend().reshape(array=emb, shape=(1, -1))
    return emb


def _da_attr_type(docs: AnyDocArray, access_path: str) -> Type[AnyTensor]:
    """Get the type of the attribute according to the Document type
    (schema) of the DocList.

    :param docs: the DocList
    :param access_path: the "__"-separated access path
    :return: the type of the attribute
    """
    field_type: Optional[Type] = _get_field_type_by_access_path(
        docs.doc_type, access_path
    )
    if field_type is None:
        raise ValueError(f"Access path is not valid: {access_path}")

    if is_union_type(field_type):
        # determine type based on the fist element
        field_type = type(next(AnyDocArray._traverse(docs[0], access_path)))

    if not issubclass(field_type, AbstractTensor):
        raise ValueError(
            f'attribute {access_path} is not a tensor-like type, '
            f'but {field_type.__class__.__name__}'
        )

    return cast(Type[AnyTensor], field_type)
