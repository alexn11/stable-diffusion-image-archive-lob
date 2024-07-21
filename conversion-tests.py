import numpy as np

from key_to_embedding import convert_bin_key_to_float_array
from key_to_embedding import pack_float_array_into_binary_key, unpack_binary_key_into_binary_float_array
from key_to_embedding import convert_float_to_15_bits, convert_exponent_to_5_bits

x = np.array([1.,2.,3.], dtype=np.float16)
x = np.array([1.123456,2.987654,3.10101], dtype=np.float16)

packed = pack_float_array_into_binary_key(x)

unpacked = unpack_binary_key_into_binary_float_array(packed, data_size=len(x)*2)
y = convert_bin_key_to_float_array(unpacked)
