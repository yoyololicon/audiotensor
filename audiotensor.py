import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Union, Iterable
from functools import reduce
from itertools import chain


def linear_upsample(x: Tensor, hop_length: int) -> Tensor:
    return F.interpolate(
        x.reshape(-1, 1, x.size(-1)),
        (x.size(-1) - 1) * hop_length + 1,
        mode="linear",
        align_corners=True,
    ).view(*x.shape[:-1], -1)


def check_hop_length(func):
    def wrapper(self, *args, **kwargs):
        if self.hop_length < 0:
            raise ValueError(
                "Cannot call {} on an AudioTensor with hop_length < 0".format(
                    func.__name__
                )
            )
        return func(self, *args, **kwargs)

    return wrapper


class AudioTensor(Tensor):
    def __new__(
        cls,
        x,
        *args,
        **kwargs,
    ):
        return super().__new__(cls, torch.as_tensor(x))

    def __init__(
        self,
        data: Union[Tensor, np.ndarray],
        hop_length: int = 1,
    ):
        super().__init__()
        self.hop_length = hop_length if data.ndim > 1 else -1

    def __repr__(self):
        return f"Hop-length: {self.hop_length}\n" + super().__repr__()

    @check_hop_length
    def set_hop_length(self, hop_length: int):
        if hop_length > self.hop_length:
            assert hop_length % self.hop_length == 0
            return self.increase_hop_length(hop_length // self.hop_length)
        elif hop_length < self.hop_length:
            assert self.hop_length % hop_length == 0
            return self.reduce_hop_length(self.hop_length // hop_length)
        return self

    @check_hop_length
    def increase_hop_length(self, factor: int):
        assert factor > 0, "factor must be positive"
        if factor == 1 or self.ndim < 2:
            return self

        data = self[:, ::factor].clone()
        data.hop_length = self.hop_length * factor
        return data

    @check_hop_length
    def reduce_hop_length(self, factor: int = None):
        if factor is None:
            factor = self.hop_length
        else:
            assert self.hop_length % factor == 0 and factor <= self.hop_length

        if factor == 1 or self.ndim < 2:
            return self

        self_copy = self.clone()
        # swap the time dimension to the last
        if self.ndim > 2:
            self_copy = self_copy.transpose(1, -1)
        expand_self_copy = linear_upsample(self_copy, factor)

        # swap the time dimension back
        if self.ndim > 2:
            expand_self_copy = expand_self_copy.transpose(1, -1)

        expand_self_copy.hop_length = self.hop_length // factor

        return expand_self_copy

    def as_tensor(self):
        t = self.clone()
        t.hop_length = -1
        return t

    @property
    @check_hop_length
    def steps(self):
        if self.ndim < 2:
            return 1
        return self.size(1)

    @check_hop_length
    def truncate(self, steps: int):
        if steps >= self.steps:
            return self
        return self.narrow(1, 0, steps)

    def new_tensor(self, data: Tensor):
        return AudioTensor(data, hop_length=self.hop_length)

    @check_hop_length
    def unfold(self, size: int, step: int = 1):
        assert self.ndim == 2
        data = super().unfold(1, size, step)
        data.hop_length = self.hop_length * step
        return data

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        mask = tuple(
            map(lambda t: isinstance(t, AudioTensor) and t.hop_length > 0, args)
        )

        if sum(mask) > 1:
            audio_tensors, regular_tensors = reduce(
                lambda x, y: (x[0] + (y[1],), x[1]) if y[0] else (x[0], x[1] + (y[1],)),
                zip(mask, args),
                ((), ()),
            )

            audio_tensors = AudioTensor.broadcasting(*audio_tensors)
            min_steps = min(a.steps for a in audio_tensors)
            audio_tensors = tuple(a.truncate(min_steps) for a in audio_tensors)
            broadcasted_args, *_ = reduce(
                lambda x, is_audio: (x[0] + x[1][:1], x[1][1:], x[2])
                if is_audio
                else (x[0] + x[2][:1], x[1], x[2][1:]),
                mask,
                ((), audio_tensors, regular_tensors),
            )
            args = broadcasted_args

        def get_all_hop_lengths(ys, xs):
            if len(xs) == 0:
                return ys
            x, *xs = xs
            if isinstance(x, AudioTensor):
                return get_all_hop_lengths(ys + [x.hop_length], xs)
            elif isinstance(x, Iterable):
                return get_all_hop_lengths(ys, tuple(chain(x, xs)))
            else:
                return get_all_hop_lengths(ys, xs)

        hop_lengths = get_all_hop_lengths([], args)

        if not all(h == hop_lengths[0] for h in hop_lengths):
            out_hop_length = -1
        else:
            out_hop_length = hop_lengths[0]

        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, AudioTensor):
            ret.hop_length = out_hop_length
        elif isinstance(ret, tuple) and all(isinstance(r, AudioTensor) for r in ret):
            ret = tuple(r.set_hop_length(out_hop_length) for r in ret)
        return ret

    @classmethod
    def broadcasting(cls, *tensors):
        assert len(tensors) > 0
        # check hop lengths are divisible by each other
        hop_lengths = tuple(t.hop_length for t in tensors)
        minimum_hop_length = min(hop_lengths)
        assert all(
            h % minimum_hop_length == 0 for h in hop_lengths
        ), "All hop lengths must be divisible by each other"
        ret = tuple(
            t.reduce_hop_length(t.hop_length // minimum_hop_length)
            if t.hop_length > minimum_hop_length
            else t
            for t in tensors
        )
        max_ndim = max(t.ndim for t in ret)
        ret = tuple(
            t[(slice(None),) * t.ndim + (None,) * (max_ndim - t.ndim)]
            if t.ndim < max_ndim
            else t
            for t in ret
        )
        return ret
