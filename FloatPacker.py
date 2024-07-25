import torch


class FloatPacker:
    # pack & unpack float16 with 15 bits (4 bits exponent), differences are where exponent values start
    def __init__(self, max_exponent: int = -12, debug: bool = False):
        if((max_exponent < 0) or (max_exponent > 30)):
            raise ValueError(f'exponent max should be within 0 to 30')
        self.exponent_max = max_exponent
        self.exponent_min = self.exponent_max - 14
        self.smallest_positive_value = (1 << self.exponent_min)
        self.largest_value = (1 << self.exponent_max) * (
                                                          1 + 1/2 + 1/4 + 1/8
                                                          + 1/16 + 1/32 + 1/64
                                                          + 1/128 + 1/256 + 1/512 + 1/1024
                                                        )
        self.exponent_biais = self.exponent_max + 1
        self.debug = debug
    def pack(self, float_value: int) -> int:
        exp = (float_value & 0b0111110000000000) >> 10
        exp -= self.exponent_biais
        exp &= 0b1111
        sign = ((float_value & 0b1000000000000000) >> 15) & 1
        packed_value = (float_value & 0b000001111111111) | (exp << 10) | (sign << 14)
        if(self.debug):
            print(f'fv={float_value:016b}')
            print(f's={sign:04b} - exp={exp:05b}')
            print(f'cv={packed_value:016b}')
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

