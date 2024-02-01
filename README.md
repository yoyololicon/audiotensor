# AudioTensor

- Wrap your tensor as `AudioTensor(tensor, hop_length)` and use it as a regular tensor.
- For `ndim > 1`, the shape is `(B, T, ...)`, where `B` is the batch size and `T` is the number of frames.
- For `ndim == 1`, the shape is `(B,)` and `hop_length = -1`.
- Tensors with different `hop_length` will be automatically linearly resampled to the maximum common divisor of `hop_length`s when performing operations.
- `set_hop_length(hop_length)` will resample the tensor to the new `hop_length`.
- `reduce_hop_length()` will upsample the tensor to `hop_length = 1`.
- `steps` will return the size of `T` in the tensor.
- `truncate(steps)` will truncate the tensor to the given `steps`.
- `unfold(size, step)` return a view of the tensor with a sliding window of size `size` and step `step` along the `T` dimension.