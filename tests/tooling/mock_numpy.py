"""Minimal numpy-like mock used explicitly by tests when numpy is unavailable."""

from __future__ import annotations


def _deep_copy(value):
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _is_2d(data):
    return isinstance(data, list) and data and isinstance(data[0], list)


class _FakeArray:
    def __init__(self, data, shape_override=None):
        self.data = _deep_copy(data)
        self._shape_override = shape_override

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @property
    def shape(self):
        if self._shape_override is not None:
            return self._shape_override
        if _is_2d(self.data):
            return (len(self.data), len(self.data[0]))
        if isinstance(self.data, list):
            return (len(self.data),)
        return ()

    @property
    def T(self):
        if not _is_2d(self.data):
            return _FakeArray(self.data)
        rows = len(self.data)
        cols = len(self.data[0])
        return _FakeArray([[self.data[r][c] for r in range(rows)] for c in range(cols)])

    def reshape(self, rows, cols):
        flat = self._flatten(self.data)
        if rows * cols != len(flat):
            raise ValueError("cannot reshape array")
        out = []
        idx = 0
        for _ in range(rows):
            end = idx + cols
            out.append(flat[idx:end])
            idx += cols
        return _FakeArray(out)

    def _flatten(self, value):
        if isinstance(value, list):
            out = []
            for entry in value:
                out.extend(self._flatten(entry))
            return out
        return [value]

    def _apply_scalar(self, scalar, fn):
        if _is_2d(self.data):
            return _FakeArray([[fn(v, scalar) for v in row] for row in self.data])
        if isinstance(self.data, list):
            return _FakeArray([fn(v, scalar) for v in self.data])
        return _FakeArray(fn(self.data, scalar))

    def __add__(self, other):
        return self._apply_scalar(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self._apply_scalar(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._apply_scalar(other, lambda a, b: a / b)

    def __iadd__(self, other):
        updated = (self + other).data
        self.data = updated
        return self

    def __eq__(self, other):
        if isinstance(other, _FakeArray):
            return self.data == other.data
        if _is_2d(self.data):
            return _FakeArray([[v == other for v in row] for row in self.data])
        if isinstance(self.data, list):
            return _FakeArray([v == other for v in self.data])
        return self.data == other

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key

            if isinstance(row_key, slice):
                rows = self.data[row_key]
            elif isinstance(row_key, list):
                rows = [self.data[i] for i in row_key]
            else:
                rows = [self.data[row_key]]

            if isinstance(col_key, slice):
                return _FakeArray([row[col_key] for row in rows])

            values = [row[col_key] for row in rows]
            if isinstance(row_key, int):
                return values[0]
            return _FakeArray(values)

        if isinstance(key, list):
            return _FakeArray([self.data[i] for i in key])

        if isinstance(key, slice):
            return _FakeArray(self.data[key])

        return self.data[key]

    def _assign_slice_values(self, idxs, value):
        if isinstance(value, list) and len(value) == len(idxs):
            for idx, item in zip(idxs, value):
                self.data[idx] = _deep_copy(item)
        else:
            for idx in idxs:
                self.data[idx] = value

    def _assign_slice_columns(self, idxs, col_key, value):
        if isinstance(value, list) and len(value) == len(idxs):
            for idx, item in zip(idxs, value):
                self.data[idx][col_key] = _deep_copy(item)
        else:
            for idx in idxs:
                self.data[idx][col_key] = value

    def __setitem__(self, key, value):
        if isinstance(value, _FakeArray):
            value = value.data

        if isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(row_key, slice) and isinstance(col_key, int):
                idxs = list(range(*row_key.indices(len(self.data))))
                self._assign_slice_columns(idxs, col_key, value)
                return

        if isinstance(key, slice):
            idxs = list(range(*key.indices(len(self.data))))
            self._assign_slice_values(idxs, value)
            return

        self.data[key] = value

    def tolist(self):
        return _deep_copy(self.data)


class _Maximum:
    @staticmethod
    def accumulate(values):
        out = []
        current = None
        for value in values:
            current = value if current is None else max(current, value)
            out.append(current)
        return out


class _MockNumpy:
    float32 = "float32"
    maximum = _Maximum()

    def array(self, value, dtype=None):
        del dtype
        if isinstance(value, _FakeArray):
            return _FakeArray(value.data)
        if hasattr(value, "tolist") and not isinstance(value, list):
            value = value.tolist()
        return _FakeArray(value)

    def arange(self, stop, dtype=None):
        del dtype
        return _FakeArray(list(range(stop)))

    def zeros(self, shape, dtype=None):
        del dtype
        if isinstance(shape, tuple) and len(shape) == 2:
            if shape[0] == 0:
                return _FakeArray([], shape_override=shape)
            return _FakeArray([[0 for _ in range(shape[1])] for _ in range(shape[0])])
        if isinstance(shape, tuple):
            return _FakeArray([0 for _ in range(shape[0])])
        return _FakeArray([0 for _ in range(shape)])

    def ones(self, shape, dtype=None):
        del dtype
        if isinstance(shape, tuple) and len(shape) == 2:
            if shape[0] == 0:
                return _FakeArray([], shape_override=shape)
            return _FakeArray([[1 for _ in range(shape[1])] for _ in range(shape[0])])
        if isinstance(shape, tuple):
            return _FakeArray([1 for _ in range(shape[0])])
        return _FakeArray([1 for _ in range(shape)])

    def tile(self, value, reps):
        arr = self.array(value)
        if not isinstance(reps, tuple) or len(reps) != 2:
            return arr
        col_repeat = reps[1]
        if _is_2d(arr.data):
            return _FakeArray([row * col_repeat for row in arr.data])
        return _FakeArray(arr.data * col_repeat)

    def roll(self, value, shift, axis=0):
        arr = self.array(value)
        if axis != 0 or len(arr.data) == 0:
            return arr
        shift = shift % len(arr.data)
        return _FakeArray(arr.data[-shift:] + arr.data[:-shift])

    def all(self, value):
        if isinstance(value, _FakeArray):
            return self.all(value.data)
        if hasattr(value, "all") and callable(value.all):
            return bool(value.all())
        if hasattr(value, "tolist") and not isinstance(value, list):
            return self.all(value.tolist())
        if isinstance(value, list):
            return all(self.all(v) for v in value)
        return bool(value)

    def array_equal(self, left, right):
        left_data = left.data if isinstance(left, _FakeArray) else left
        right_data = right.data if isinstance(right, _FakeArray) else right
        if hasattr(left_data, "tolist") and not isinstance(left_data, list):
            left_data = left_data.tolist()
        if hasattr(right_data, "tolist") and not isinstance(right_data, list):
            right_data = right_data.tolist()
        return left_data == right_data

    def mean(self, value, axis=None):
        arr = self.array(value).data
        if axis == 1:
            return [sum(row) / len(row) if row else 0 for row in arr]
        flat = []
        for row in arr:
            if isinstance(row, list):
                flat.extend(row)
            else:
                flat.append(row)
        return sum(flat) / len(flat) if flat else 0

    def argmax(self, values):
        return max(range(len(values)), key=lambda i: values[i])

    def zeros_like(self, value):
        arr = self.array(value)
        if _is_2d(arr.data):
            return _FakeArray([[0 for _ in row] for row in arr.data])
        return _FakeArray([0 for _ in arr.data])

    def linspace(self, start, stop, num):
        if num <= 1:
            return _FakeArray([start])
        step = (stop - start) / (num - 1)
        return _FakeArray([start + i * step for i in range(num)])

    def vstack(self, arrays):
        rows = []
        for arr in arrays:
            arr_obj = self.array(arr)
            rows.extend(arr_obj.data)
        return _FakeArray(rows)

    def argsort(self, values):
        return sorted(range(len(values)), key=lambda i: values[i])

    def unique(self, values, return_index=False):
        seen = {}
        uniq = []
        idxs = []
        for idx, value in enumerate(values):
            if value not in seen:
                seen[value] = idx
                uniq.append(value)
                idxs.append(idx)
        if return_index:
            return uniq, idxs
        return uniq


mock_numpy = _MockNumpy()
