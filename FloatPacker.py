# Copyright 2024 Alexandre De Zotti. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class FloatPacker:
    # pack & unpack float16 with 15 bits (4 bits exponent), differences are where exponent values start
    def __init__(self, max_exponent: int = -12, debug: bool = False):
        if((max_exponent < 0) or (max_exponent > 30)):
            raise ValueError(f'exponent max should be within 0 to 30')
        self.exponent_max = max_exponent
        self.exponent_min = self.exponent_max - 14
        self.smallest_positive_value = 2.**self.exponent_min
        self.largest_value = (2**self.exponent_max) * (
                                                          1 + 1/2 + 1/4 + 1/8
                                                          + 1/16 + 1/32 + 1/64
                                                          + 1/128 + 1/256 + 1/512 + 1/1024
                                                      )
        self.exponent_biais = self.exponent_max + 1
        self.debug = debug
        self._print_ct = 0
    def pack(self, float_value: int) -> int:
        exp = (float_value & 0b0111110000000000) >> 10
        exp -= self.exponent_biais
        exp &= 0b1111
        sign = ((float_value & 0b1000000000000000) >> 15) & 1
        packed_value = (float_value & 0b000001111111111) | (exp << 10) | (sign << 14)
        if(self.debug):
            if(self._print_ct < 24):
                print(f'fv={float_value:016b}')
                print(f's={sign:04b} - exp={exp:05b}')
                print(f'cv={packed_value:016b}')
                self._print_ct += 1
        return packed_value
    def unpack(self, packed_value: int) -> int:
        exp = (packed_value & 0b011110000000000) >> 10
        exp += self.exponent_biais
        sign = (packed_value & 0b100000000000000) >> 14
        float_value = (packed_value & 0b000001111111111) | (exp << 10) | (sign << 15)
        return float_value
    def normalise_numbers(self, x: torch.Tensor,) -> torch.Tensor:
        x[torch.abs(x) < self.smallest_positive_value] = self.smallest_positive_value
        x[x > self.largest_value] = self.largest_value
        x[x < - self.largest_value] = - self.largest_value
        return x

