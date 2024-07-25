import numpy as np
import torch

from model_constants import prompt_embeddings_exponent_max
from FloatPacker import FloatPacker
from key_to_embedding import convert_bin_key_to_float_array
from key_to_embedding import pack_float_array_into_binary_key, unpack_binary_key_into_binary_float_array
from key_to_embedding import convert_float_to_15_bits, convert_exponent_to_5_bits
from key_to_embedding import generate_random_key_base64, convert_key_to_binary


def pack_and_unpack(x: np.ndarray):
    packed = pack_float_array_into_binary_key(x, dbg=True)
    unpacked = unpack_binary_key_into_binary_float_array(packed, data_size=len(x)*2, dbg=True)
    y = convert_bin_key_to_float_array(unpacked, endian='<', dbg=True)
    return y, unpacked, packed

def show_binary_data(data: bytes):
    for d in data:
        print(f'{d:08b}')

def check_pack_and_unpack(x: np.ndarray):
    y, _, packed = pack_and_unpack(x)
    print(f'packed=')
    show_binary_data(packed)
    try:
        assert(np.allclose(x, y)) #, atol=0.0125))
    except AssertionError:
        print('FAILED')
        print(f'x={x}')
        print(f'y={y}')
        #assert(np.allclose(x[:-1], y[:-1], atol=0.01))
        raise


x = np.array([1.,2.,3., -1., 12., 0.25], dtype=np.float16)
check_pack_and_unpack(x)

x = np.array([1.123456,2.987654,3.10101], dtype=np.float16)
y, unpacked, packed = pack_and_unpack(x)
for i in packed:
    print(f'{i:08b}')
assert(np.allclose(x, y, atol=0.01))

x = np.array([11.,12.,13., -11., 2., 9.25, ], dtype=np.float16)
check_pack_and_unpack(x)

x_data = [-1.363, 1.349, 0.01707, -0.00911, -0.779, 1.878, -2.041, -0.5615, 1.403, 1.253, -0.1927, -1.7, 1.548, 2.084, -0.852, 1.37, -0.1356, 1.527, -0.01339, -0.0344, 1.615, 0.6553, 0.1733, 1.49, 2.451, -0.3435, -0.628, -0.00301, 1.206, -0.508, 1.806, 1.119, -2.068, -1.759, 2.193, -2.074, 0., -0.2406, -0.4497, ]
normaliser = FloatPacker(max_exponent=prompt_embeddings_exponent_max, debug=True)
x = normaliser.normalise_numbers(torch.tensor(x_data, dtype=torch.float16)).detach().cpu().numpy()
print(f'🔦️ orig - norm = {np.max(np.abs(x-np.array(x_data, dtype=np.float16)))}')
check_pack_and_unpack(x)

for array_len in range(19):
    print(f'🪰️🪰️ array len = {array_len}')
    data_size = array_len * 2
    generated_key = generate_random_key_base64(array_len)
    print(f'generated key="{generated_key}" - {data_size} -> {len(generated_key)}')
    key_bin = convert_key_to_binary(generated_key, nb_bits_target=array_len * 15, dbg=True)
    print(f'bin key len={len(key_bin)}')
    unpacked = unpack_binary_key_into_binary_float_array(key_bin, data_size=data_size, dbg=True)
    print(f'unpacked: {len(unpacked)}')
    print(unpacked)
    y = convert_bin_key_to_float_array(unpacked, endian='<', dbg=True)
    print(y)
    assert(len(y) == array_len)

print('packing unpacking random arrays')
for array_len in range(1, 19):
    print(f'🪨️🪨️ arr len={array_len}')
    x = np.random.random_sample(size=array_len).astype(np.float16)
    check_pack_and_unpack(x)


print('✅️ passed')

