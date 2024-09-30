from typing import Tuple
import numpy as np

from luma.interface.typing import TensorLike
from luma.neural.base import Layer


__all__ = "_Slice"


class _Slice(Layer):
    def __init__(self, slice_str: str) -> None:
        super().__init__()
        self.slices = self._parse_slice_str(slice_str)
        self.input_shape = None

    def _parse_slice_part(self, part: str) -> slice | int:
        part = part.strip()
        if part == ":":
            return slice(None)
        elif ":" in part:
            tokens = part.split(":")
            if len(tokens) > 3:
                raise ValueError(f"Invalid slice part: '{part}'")
            tokens = [
                int(token) if token.strip("-").isdigit() else None for token in tokens
            ]
            while len(tokens) < 3:
                tokens.append(None)
            return slice(*tokens)
        else:
            return int(part)

    def _parse_slice_str(self, slice_str: str) -> Tuple[slice | int]:
        parts = slice_str.split(",")
        slices = []
        for part in parts:
            if part.strip() == "":
                continue
            slices.append(self._parse_slice_part(part))

        return tuple(slices)

    def _complete_slices(self, input_shape: Tuple[int]) -> Tuple[slice | int]:
        completed = list(self.slices)
        if len(completed) < len(input_shape):
            completed += [slice(None)] * (len(input_shape) - len(completed))

        elif len(completed) > len(input_shape):
            raise ValueError(
                "Number of slicing arguments exceeds number of input dimensions."
            )

        return tuple(completed)

    def _compute_out_shape(
        self, input_shape: Tuple[int], completed_slices: Tuple[slice | int]
    ) -> Tuple[int]:
        output_shape = []
        for dim, sl in zip(input_shape, completed_slices):
            if isinstance(sl, slice):
                start, stop, step = sl.indices(dim)
                length = max(
                    0, (stop - start + (step - 1 if step > 0 else step + 1)) // step
                )
                output_shape.append(length)

            elif isinstance(sl, int):
                continue

        return tuple(output_shape)

    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        _ = is_train
        self.input_shape = X.shape

        completed_slices = self._complete_slices(self.input_shape)
        self.completed_slices = completed_slices

        out = X[completed_slices]
        return out

    def backward(self, d_out: TensorLike) -> TensorLike:
        if self.input_shape is None:
            raise ValueError("Must perform forward pass before backward.")

        grad_input = np.zeros(self.input_shape)
        grad_input[self.completed_slices] = d_out

        return grad_input

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        completed_slices = self._complete_slices(in_shape)
        return self._compute_out_shape(in_shape, completed_slices)
