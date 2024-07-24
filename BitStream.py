from numpy import int8, int16, int32, int64
from numpy import array, ndarray

class BitStream:
    # starts with lowest digits
    def __init__(self, data: bytes | bytearray | None = None,
                 chunk_size_bits=1,
                 data_size_bits=None,
                 mode='r'):
        self.mode = mode
        if(self.mode == 'r'):
            self.can_write = False
        elif(self.mode in ['w', 'rw']): # i allow read in w mode because whats the point?
            self.can_write = True
        else:
            raise ValueError(f'unknown mode: "{mode}" (expects: r,w,rw)')
        self.set_data(data, data_size_bits=data_size_bits)
        self.set_chunk_size(chunk_size_bits)
        self.set_index(0)
    def set_data(self,
                 data: bytes | bytearray | None,
                 data_size_bits: int | None):
        if(data is None):
            if(self.mode in ['r','rw']):
                raise ValueError(f'stream read mode needs some data but data={None}')
            else:
                if(data_size_bits is None):
                    raise ValueError(f'either specify data or data_size_bits (both cant be None)')
                target_size_bytes = (data_size_bits + 7) // 8
                self.data = bytearray(target_size_bytes * [0])
        else:
            max_size_bits = len(data) * 8
            if(data_size_bits is None):
                data_size_bits = max_size_bits
            elif(data_size_bits > max_size_bits):
                raise ValueError(f'specified size ({data_size_bits}) greater than data size ({max_size_bits})')
            if(self.mode == 'r'):
                self.data = bytes(data)
            else:
                self.data = bytearray(data)
        self.data_size_bits = data_size_bits
    def set_index(self, i: int = 0):
        self.index = i
        self.byte_i = i // 8
        self.bit_i = i % 8
    def increase_index(self, nb_bits: int):
        self.set_index(self.index + nb_bits)
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
    def read_chunk(self,) -> int:
        return self.get_chunk()
    def read_chunks(self, nb_chunks=1, return_type='list') -> list | ndarray:
        return self.get_chunks(nb_chunks=nb_chunks, return_type=return_type)
    def write_chunk(self, chunk: int):
        if(not self.can_write):
            raise RuntimeError(f'stream is read only')
        remaining_bits = chunk
        nb_remaining_to_write = self.chunk_size 
        while(nb_remaining_to_write > 0):
            write_len = min(nb_remaining_to_write, 8 - self.bit_i)
            write_mask = (1 << write_len) - 1
            store_mask = ((1 << self.bit_i) - 1)
            if(write_len + self.bit_i < 8):
                higher_bits_position = write_len + self.bit_i 
                nb_higher_bits = (8 - higher_bits_position)
                store_mask |= ((1 << nb_higher_bits) - 1) << higher_bits_position
            self.data[self.byte_i] &= store_mask
            self.data[self.byte_i] |= (remaining_bits & write_mask) << self.bit_i
            remaining_bits >>= write_len
            remaining_bits &= (1 << nb_remaining_to_write) - 1
            self.increase_index(write_len)
            nb_remaining_to_write -= write_len
    def write_chunks(self, chunks: list):
        for chunk in chunks:
            self.write_chunk(chunk)


