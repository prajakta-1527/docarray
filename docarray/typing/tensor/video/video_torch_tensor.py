from typing import TYPE_CHECKING, Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.torch_tensor import TorchTensor, metaTorchAndNode
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin

T = TypeVar('T', bound='VideoTorchTensor')

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


@_register_proto(proto_type_name='video_torch_tensor')
class VideoTorchTensor(TorchTensor, VideoTensorMixin, metaclass=metaTorchAndNode):
    """
    Subclass of [`TorchTensor`][docarray.typing.TorchTensor], to represent a video tensor.
    Adds video-specific features to the tensor.

    ---

    ```python
    from typing import Optional

    import torch

    from docarray import BaseDoc
    from docarray.typing import VideoTorchTensor, VideoUrl


    class MyVideoDoc(BaseDoc):
        title: str
        url: Optional[VideoUrl]
        video_tensor: Optional[VideoTorchTensor]


    doc_1 = MyVideoDoc(
        title='my_first_video_doc',
        video_tensor=torch.randn(size=(100, 224, 224, 3)),
    )
    # doc_1.video_tensor.save(file_path='file_1.mp4')

    doc_2 = MyVideoDoc(
        title='my_second_video_doc',
        url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true',
    )

    doc_2.video_tensor = doc_2.url.load().video
    # doc_2.video_tensor.save(file_path='file_2.wav')
    ```

    ---

    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        tensor = super().validate(value=value, field=field, config=config)
        return cls.validate_shape(value=tensor)
