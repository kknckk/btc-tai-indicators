def bits_to_difficulty(bits_hex):
    val = int(bits_hex, 16)
    exponent = val >> 24
    coefficient = val & 0xffffff
    target = coefficient * (256 ** (exponent - 3))
    max_target = 0xFFFF * (256 ** (0x1d - 3))
    return max_target / target

print(bits_to_difficulty("1a09ee5d"))
print(bits_to_difficulty("17034b17"))
