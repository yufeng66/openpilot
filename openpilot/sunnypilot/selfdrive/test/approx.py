"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np


class approx:
  """Tolerant equality for asserts, standing in for pytest.approx.

  openpilot dropped pytest for a unittest runner (#38471), but the fork's control
  tests compare a lot of floats and float lists. Same tolerance rule pytest used:
  a value matches when it is within max(rtol * |expected|, atol), elementwise for
  sequences. Named rtol/atol rather than pytest's rel/abs because `abs` would
  shadow a builtin.
  """

  def __init__(self, expected, rtol: float = 1e-6, atol: float = 1e-12):
    self.expected = expected
    self.rtol = rtol
    self.atol = atol

  def __eq__(self, other) -> bool:
    expected = np.asarray(self.expected, dtype=float)
    actual = np.asarray(other, dtype=float)
    if expected.shape != actual.shape:
      return False
    tolerance = np.maximum(self.rtol * np.abs(expected), self.atol)
    return bool(np.all(np.abs(actual - expected) <= tolerance))

  def __ne__(self, other) -> bool:
    return not self.__eq__(other)

  def __hash__(self):
    return hash(("approx", self.rtol, self.atol))

  def __repr__(self) -> str:
    return f"approx({self.expected!r}, rtol={self.rtol}, atol={self.atol})"
