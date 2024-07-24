from BitStream import BitStream


stream = BitStream(chunk_size_bits=15, data_size_bits=45, mode='w')

assert(len(stream.data) == 6)

stream.write_chunk(0b110100100110010)
stream.write_chunk(0b111010111010111)
expected_data = [
    0b00110010,
    0b11101001,
    0b11101011,
    0b111010,
    0,
    0
]
assert(stream.byte_i == 3)
for i in range(4):
    assert(stream.data[i] == expected_data[i])

stream.set_chunk_size(2)
stream.write_chunks([0b01,0b11,0b01,0b11,0b00,0b10,])
expected_data = [
    0b00110010,
    0b11101001,
    0b11101011,
    0b01111010,
    0b00110111,
    0b10,
]
assert(stream.byte_i == 5)
for i in range(6):
    assert(stream.data[i] == expected_data[i])


stream.set_index()
stream.set_chunk_size(3)
stream.write_chunks([0,0b111,0b011])
expected_data = [
    0b11111000,
    0b11101000,
    0b11101011,
    0b01111010,
    0b00110111,
    0b10,
]

for d in stream.data:
    print(f'{d:08b}')

for i in range(6):
    assert(stream.data[i] == expected_data[i])




print('😏️ passed')