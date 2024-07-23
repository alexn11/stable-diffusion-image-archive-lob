from BitStream import BitStream


x = [ 0b01010101, 0b11110000, 0b00110011, 0b11100111, 0b00111100, 0b00000101 ]
bit_stream = BitStream(bytes(x), chunk_size_bits=15, data_size_bits=45)
values = bit_stream.get_chunks(3)
for v in values:
    print(f'{v:015b}')
expected_values = [
    0b111000001010101,
    0b100111001100111,
    0b001010011110011
]
assert(len(values) == 3)
assert(all([values[i] == expected_values[i] for i in range(3)]))

bit_stream = BitStream(bytes(x), chunk_size_bits=14, data_size_bits=42)
values = bit_stream.get_chunks(3)
for v in values:
    print(f'{v:014b}')
expected_values = [
    0b11000001010101,
    0b01110011001111,
    0b01001111001110
]
assert(len(values) == 3)
assert(all([values[i] == expected_values[i] for i in range(3)]))

x = [ 0b01010101, 0b11110000, 0b00110011, 0b11100111, 0b00111100, 0b00000101 ]
bit_stream = BitStream(bytes(x), chunk_size_bits=2, data_size_bits=44)
values = bit_stream.get_chunks(22)
for v in values:
    print(f'{v:02b}')
expected_values = [
    0b01,0b01,0b01,0b01,
    0b00,0b00,0b11,0b11,
    0b11,0b00,0b11,0b00,
    0b11,0b01,0b10,0b11,
    0b00,0b11,0b11,0b00,
    0b01,0b01 
]
assert(len(values) == 22)
for i in range(22):
    assert(values[i] == expected_values[i])

# changing chunk size in the middle
x = [ 0b01010101, 0b11110000, 0b00110011, 0b11100111, 0b00111100, 0b00000101 ]
bit_stream = BitStream(bytes(x), chunk_size_bits=14, data_size_bits=42)
values = bit_stream.get_chunks(2)
for v in values:
    print(f'{v:014b}')
assert(len(values) == 2)
expected_values = [
    0b11000001010101,
    0b01110011001111,
]
assert(all([values[i] == expected_values[i] for i in range(2)]))
bit_stream.set_chunk_size(15)
value = bit_stream.get_chunk()
assert(value == 0b101001111001110)