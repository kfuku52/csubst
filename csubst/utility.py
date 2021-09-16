import numpy

def rgb_to_hex(r,g,b):
    rgb = [r,g,b]
    for i in range(len(rgb)):
        assert rgb[i]<=1
        rgb[i] = int(numpy.round(rgb[i]*255, decimals=0))
    hex_color = '0x%02X%02X%02X' % (rgb[0],rgb[1],rgb[2])
    return hex_color