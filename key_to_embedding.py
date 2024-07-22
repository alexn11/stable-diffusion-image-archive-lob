import base64
import random
import struct

import numpy as np

# key things

base64_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

def generate_random_key_base64(nb_shorts_target: int = 151552, dbg=False) -> str:
    #def generate_random_base64(nb_bytes: int = 118268 + 83200) -> str:
    # 110880 == 77 * 768 * 15 / 8
    nb_bits = nb_shorts_target * 15
    length = nb_bits // 6
    nb_extra_bits = nb_bits % 6
    base64_key = ''.join([ random.choice(base64_characters) for i in range(length) ])
    match(nb_extra_bits):
        case 0:
            extra_bits = ''
        case 2:
            extra_bits = random.choice(base64_characters[:4])
        case 3: # only possible case (with 0) in the current form
            extra_bits = random.choice(base64_characters[:8])
        case 4:
            extra_bits = random.choice(base64_characters[:16])
        case _:
            print(f'nb extra bits:{nb_extra_bits} ??')
    base64_key += extra_bits
    match(len(base64_key) % 4):
        case 0:
            padding = ''
        case 1:
            padding = base64_characters[0] * 3
        case 2:
            padding = base64_characters[0] * 2
        case 3:
            padding = base64_characters[0]
    base64_key += padding
    return base64_key

def convert_key_to_binary(key: str, nb_bits_target: int | None = None, dbg=False) -> bytes:
    try:
        bin_key = base64.b64decode(key)
    except:
        try:
            bin_key = base64.b64decode(key + '=')
        except:
            try:
                bin_key = base64.b64decode(key + '==')
            except:
                try:
                    bin_key = base64.b64decode(key + '===')
                except:
                    raise
    if(nb_bits_target != None):
        nb_bytes = nb_bits_target // 8
        if((nb_bits_target % 8) > 0):
            nb_bytes += 1
        bin_key = bin_key[:nb_bytes]
    # https://stackoverflow.com/questions/73089007/invalid-base64-encoded-string-number-of-data-characters-13-cannot-be-1-more-t
    return bin_key


#

def convert_exponent_to_5_bits(datum: int):
    exp = (datum & 0b011110000000000) >> 10
    exp += 3
    sign = (datum & 0b100000000000000) >> 14
    datum = (datum & 0b000001111111111) | (exp << 10) | (sign << 15)
    return datum

def convert_float_to_15_bits(float_value: int):
    exp = (float_value & 0b0111110000000000) >> 10
    exp -= 3
    exp &= 0b1111
    sign = ((float_value & 0b1000000000000000) >> 15) & 1
    datum = (float_value & 0b000001111111111) | (exp << 10) | (sign << 14)
    #print(f'fv={float_value:016b}')
    #print(f's={sign:04b} - exp={exp:05b}')
    #print(f'cv={datum:016b}')
    return datum

def unpack_binary_key_into_binary_float_array(key_bin: bytes,
                                              data_size=(77*768)*2+4*52*80*2,
                                              dbg=False) -> bytes:
    unpacked_bytes = bytearray(data_size * [0])
    #
    extra_bits = 0
    nb_extra_bits = 0
    is_8_bits = True
    unpacked_i = 0
    key_byte_i = 0
    for byte in key_bin:
        #dbg_key = '8' if(is_8_bits) else '7'
        #print(f'({dbg_key}): input={byte:08b}')
        unpacked_byte = ((byte << nb_extra_bits) | extra_bits) & 0xff
        try:
            unpacked_bytes[unpacked_i] = unpacked_byte
            #print(f'({dbg_key}-T) unpacked byte: {unpacked_bytes[unpacked_i]:08b} - extra = {extra_bits} - {"8" if(is_8_bits) else "7"} bits')
        except IndexError:
            print(f'key byte:{key_byte_i} ({len(key_bin)}) - unpacked: {unpacked_i} ({data_size})')
            raise
        unpacked_i += 1
        if(unpacked_i == data_size):
            print(f'reached padding at {key_byte_i} ({len(key_bin)})')
            break
        if(is_8_bits):
            extra_bits_shift_len = 8 - nb_extra_bits
            extra_bits = (byte >> extra_bits_shift_len) % (1 << nb_extra_bits)
            #extra_bits_str = f'{extra_bits:08b}'[-nb_extra_bits:]
            #print('\n')
            #print(f'(8) extra={extra_bits_str} - nb={nb_extra_bits} - len={extra_bits_shift_len}')
            if(nb_extra_bits >= 7):
                unpacked_bytes[unpacked_i] = extra_bits
                #print('\n')
                #print(f'(7-) unpacked byte: {unpacked_bytes[unpacked_i]:08b} - extra = {extra_bits} - {"8" if(is_8_bits) else "7"} bits')
                unpacked_i += 1
                if(unpacked_i == data_size):
                    print(f'reached padding at {key_byte_i} ({len(key_bin)})')
                    break
                extra_bits = (extra_bits >> 7) & 1
                nb_extra_bits = nb_extra_bits - 7
                #print(f'(7+) extra={extra_bits:08b} - nb={nb_extra_bits} - len=1')
                is_8_bits = not is_8_bits
        else:
            unpacked_bytes[unpacked_i-1] &= 0x7f
            #print(f'(7) unpacked byte: {unpacked_bytes[unpacked_i-1]:08b} - extra = {extra_bits} - {"8" if(is_8_bits) else "7"} bits')
            nb_extra_bits = nb_extra_bits + 1
            extra_bits_shift_len = 8 - nb_extra_bits
            extra_bits = (byte >> extra_bits_shift_len) % (1 << nb_extra_bits)
            #extra_bits_str = f'{extra_bits:08b}'[-nb_extra_bits:]
            #print(f'(7) extra={extra_bits_str}  - nb={nb_extra_bits} - len={extra_bits_shift_len}')
        is_8_bits = not is_8_bits
        key_byte_i += 1
        if(unpacked_i == data_size):
            print(f'reached padding at {key_byte_i} ({len(key_bin)})')
            break
    #
    #print('unpacked bytes:')
    #for i in range(data_size):
    #    print(f'{unpacked_bytes[i]:08b}')
    #
    for value_i in range(data_size // 2):
        datum = (unpacked_bytes[2 * value_i + 1] << 8) | unpacked_bytes[2 * value_i]
        datum = convert_exponent_to_5_bits(datum)
        #print(f'datum={datum:016b}')
        unpacked_bytes[2 * value_i + 1] = (datum >> 8) & 0xff
        unpacked_bytes[2 * value_i] = datum & 0xff
    #
    return unpacked_bytes

def pack_float_array_into_binary_key(float16_array: np.ndarray, dbg=False) -> bytes:
    array_len = len(float16_array)
    #array_size_bytes = array_len * 2
    array_data = struct.unpack(f'{array_len}h', bytes(float16_array.data))
    packed_data_size_bits = array_len * 15
    packed_data_size_bytes = packed_data_size_bits // 8
    packed_data_nb_extra_bits = packed_data_size_bits % 8
    if(packed_data_nb_extra_bits > 0):
        packed_data_size_bytes += 1
    packed_data = bytearray(packed_data_size_bytes * [0])
    data_byte_i = 0
    datum_bit_i = 0
    current_datum = 0
    #print(f'current={current_datum:08b} - bit_i={datum_bit_i}')
    for array_value in array_data:
        #print(f'arrayv={array_value:016b}')
        packed_binary_value = convert_float_to_15_bits(array_value)
        packed_binary_value = (packed_binary_value & 0x7fff)
        #print(f'adding={packed_binary_value:015b}')
        # 1st bit
        shift_len = datum_bit_i
        mask_len = 8 - datum_bit_i
        mask = (1 << mask_len) - 1
        current_datum |= (packed_binary_value & mask) << shift_len
        packed_binary_value = (packed_binary_value >> mask_len)
        #print(f'current(0)={current_datum:08b}')
        packed_data[data_byte_i] = current_datum
        data_byte_i += 1
        # 2nd bit
        current_datum = packed_binary_value & 0xff
        #print(f'current(1)={current_datum:08b}')
        if(datum_bit_i != 0):
            packed_data[data_byte_i] = current_datum
            data_byte_i += 1
            packed_binary_value = (packed_binary_value >> 8) & 0xff
            current_datum = 0
            #print(f'current(2)={current_datum:08b}')
        else:
            current_datum &= 0x7f
            #print(f'current(1+)={current_datum:08b}')
        # 3rd bit
        if(datum_bit_i not in [0, 1]):
            mask_len = datum_bit_i - 1
            mask = (1 << mask_len) - 1
            current_datum = packed_binary_value & mask
            #print(f'current(2+)={current_datum:08b}')
        datum_bit_i = (datum_bit_i - 1) % 8
        #print(f'bit_i={datum_bit_i}')
    if(datum_bit_i > 0):
        #print(f'last extra={current_datum:08b}')
        packed_data[data_byte_i] = current_datum
        data_byte_i += 1
    return packed_data


def convert_bin_key_to_float_array(data: bytes, endian='<', dbg=False) -> bytes:
    data_size = len(data)
    nb_floats = data_size // 2
    float_array = struct.unpack(f'{endian}{nb_floats}e', data)
    float_array = np.array(float_array, dtype=np.float16)
    return float_array

def compute_embedding_and_latents_from_key(key: str|None = None,
                                           key_file_path: str|None = None,
                                           prompt_embeddings_size=77*768,
                                           latents_size=4*52*80):
    if(key is None):
        with open(key_file_path, 'r') as key_file:
            key = key_file.read().strip()
    key_bin = convert_key_to_binary(key, nb_bits_target=(prompt_embeddings_size + latents_size) * 15)
    prompt_embeddings_bin_size = 2 * prompt_embeddings_size
    latents_bin_size = 2 * latents_size
    data_bin_size = prompt_embeddings_bin_size + latents_bin_size
    #print(f'key bin len={len(key_bin)} - data size={data_bin_size} ({prompt_embeddings_bin_size}+{latents_bin_size})')
    data =  unpack_binary_key_into_binary_float_array(key_bin, data_size=data_bin_size)
    prompt_embeddings = convert_bin_key_to_float_array(data[:prompt_embeddings_bin_size],)
    # specific values
    prompt_embeddings[19] = -28.078125
    prompt_embeddings[681] = 33.09375
    #
    assert(len(prompt_embeddings) == prompt_embeddings_size)
    latents = convert_bin_key_to_float_array(data[prompt_embeddings_bin_size:])
    assert(len(latents)==latents_size)
    return prompt_embeddings, latents
