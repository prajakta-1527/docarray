import numpy as np
import pytest
import torch

from docarray import Document, DocumentArray
from docarray.typing import NdArray, TorchTensor
from docarray.utility import find, find_batched


class TorchDoc(Document):
    tensor: TorchTensor


class NdDoc(Document):
    tensor: NdArray


@pytest.fixture()
def random_torch_query():
    return TorchDoc(tensor=torch.rand(128))


@pytest.fixture()
def random_torch_batch_query():
    return DocumentArray[TorchDoc]([TorchDoc(tensor=torch.rand(128)) for _ in range(5)])


@pytest.fixture()
def random_nd_query():
    return NdDoc(tensor=np.random.rand(128))


@pytest.fixture()
def random_nd_batch_query():
    return DocumentArray[NdDoc]([NdDoc(tensor=np.random.rand(128)) for _ in range(5)])


@pytest.fixture()
def random_torch_index():
    return DocumentArray[TorchDoc](TorchDoc(tensor=torch.rand(128)) for _ in range(10))


@pytest.fixture()
def random_nd_index():
    return DocumentArray[NdDoc](NdDoc(tensor=np.random.rand(128)) for _ in range(10))


@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_torch(random_torch_query, random_torch_index, metric):
    top_k, scores = find(
        random_torch_index,
        random_torch_query,
        embedding_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    if metric.endswith('_dist'):
        assert (torch.stack(sorted(scores)) == scores).all()
    else:
        assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


def test_find_torch_tensor_query(random_torch_query, random_torch_index):
    query = random_torch_query.tensor
    top_k, scores = find(
        random_torch_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


def test_find_torch_stacked(random_torch_query, random_torch_index):
    random_torch_index.stack()
    top_k, scores = find(
        random_torch_index,
        random_torch_query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (torch.stack(sorted(scores, reverse=True)) == scores).all()


@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_np(random_nd_query, random_nd_index, metric):
    top_k, scores = find(
        random_nd_index,
        random_nd_query,
        embedding_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    if metric.endswith('_dist'):
        assert (sorted(scores) == scores).all()
    else:
        assert (sorted(scores, reverse=True) == scores).all()


def test_find_np_tensor_query(random_nd_query, random_nd_index):
    query = random_nd_query.tensor
    top_k, scores = find(
        random_nd_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (sorted(scores, reverse=True) == scores).all()


def test_find_np_stacked(random_nd_query, random_nd_index):
    random_nd_index.stack()
    top_k, scores = find(
        random_nd_index,
        random_nd_query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(top_k) == 7
    assert len(scores) == 7
    assert (sorted(scores, reverse=True) == scores).all()


@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_batched_torch(random_torch_batch_query, random_torch_index, metric):
    results = find_batched(
        random_torch_index,
        random_torch_batch_query,
        embedding_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(results) == len(random_torch_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        if metric.endswith('_dist'):
            assert (torch.stack(sorted(sc)) == sc).all()
        else:
            assert (torch.stack(sorted(sc, reverse=True)) == sc).all()


def test_find_batched_torch_tensor_query(random_torch_batch_query, random_torch_index):
    query = torch.stack(random_torch_batch_query.tensor)
    results = find_batched(
        random_torch_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(results) == len(random_torch_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        assert (torch.stack(sorted(sc, reverse=True)) == sc).all()


@pytest.mark.parametrize('stack_what', ['index', 'query', 'both'])
def test_find_batched_torch_stacked(
    random_torch_batch_query, random_torch_index, stack_what
):
    if stack_what in ('index', 'both'):
        random_torch_index.stack()
    if stack_what in ('query', 'both'):
        random_torch_batch_query.stack()

    results = find_batched(
        random_torch_index,
        random_torch_batch_query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(results) == len(random_torch_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        assert (torch.stack(sorted(sc, reverse=True)) == sc).all()


@pytest.mark.parametrize('metric', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
def test_find_batched_np(random_nd_batch_query, random_nd_index, metric):
    results = find_batched(
        random_nd_index,
        random_nd_batch_query,
        embedding_field='tensor',
        limit=7,
        metric=metric,
    )
    assert len(results) == len(random_nd_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        if metric.endswith('_dist'):
            assert (sorted(sc) == sc).all()
        else:
            assert (sorted(sc, reverse=True) == sc).all()


def test_find_batched_np_tensor_query(random_nd_batch_query, random_nd_index):
    query = np.stack(random_nd_batch_query.tensor)
    results = find_batched(
        random_nd_index,
        query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(results) == len(random_nd_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        assert (sorted(sc, reverse=True) == sc).all()


@pytest.mark.parametrize('stack_what', ['index', 'query', 'both'])
def test_find_batched_np_stacked(random_nd_batch_query, random_nd_index, stack_what):
    if stack_what in ('index', 'both'):
        random_nd_index.stack()
    if stack_what in ('query', 'both'):
        random_nd_batch_query.stack()
    results = find_batched(
        random_nd_index,
        random_nd_batch_query,
        embedding_field='tensor',
        limit=7,
        metric='cosine_sim',
    )
    assert len(results) == len(random_nd_batch_query)
    for top_k, scores in results:
        assert len(top_k) == 7
        assert len(scores) == 7
    for sc in [scores for _, scores in results]:
        assert (sorted(sc, reverse=True) == sc).all()