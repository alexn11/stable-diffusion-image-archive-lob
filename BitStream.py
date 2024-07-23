from numpy import int8, int16, int32, int64
from numpy import array, ndarray

class BitStream:
    # starts with lowest digits
    def __init__(self, data: bytes, chunk_size_bits=1, data_size_bits=None):
        self.data = data
        self.set_data_size(data_size_bits)
        self.set_chunk_size(chunk_size_bits)
        self.set_index(0)
    def set_index(self, i: int = 0):
        self.index = i
        self.byte_i = i // 8
        self.bit_i = i % 8
    def increase_index(self, nb_bits: int):
        self.set_index(self.index + nb_bits)
    def set_data_size(self, data_size_bits):
        max_size = len(self.data) * 8
        if(data_size_bits is None):
            data_size_bits = max_size
        self.data_size_bits = data_size_bits
    def set_chunk_size(self, chunk_size_bits: int):
        # im only going to use it for nb bits <= 16 (namely 14, 15, 2)
        if(chunk_size_bits > 16):
            raise ValueError(f'chunk size is too large: {chunk_size_bits} (max 64 is bits)')
        self.chunk_size = chunk_size_bits
        if(chunk_size_bits <= 8):
            self.output_type = int8
        elif(chunk_size_bits <= 16):
            self.output_type = int16
        #elif(chunk_size_bits <= 32):
        #    self.output_type = int32
        #else:
        #    self.output_type = int64
    def get_chunk(self,):
        read_bits_count = 0
        chunk_value = 0
        remaining_to_read = self.chunk_size - read_bits_count
        while(remaining_to_read > 0):
            read_len = min(remaining_to_read, 8 - self.bit_i)
            read_mask = (1 << read_len) - 1
            chunk_value = chunk_value | (((self.data[self.byte_i] >> self.bit_i) & read_mask) << read_bits_count)
            self.increase_index(read_len)
            read_bits_count += read_len
            remaining_to_read -= read_len
        return chunk_value
    def get_chunks(self, nb_chunks=1, return_type='list') -> list | ndarray:
        chunks = [ self.get_chunk() for i in range(nb_chunks) ]
        if(return_type != 'list'):
            chunks = array(chunks, dtype=self.output_type)
        return chunks



