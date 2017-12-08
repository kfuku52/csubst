def get_chunks(input, nslots):
    if 'shape' in dir(input):
        chunks = [(input.shape[0] + i) // nslots for i in range(nslots)]
    else:
        chunks = [(len(input) + i) // nslots for i in range(nslots)]
    i = 0
    out_chunks = list()
    starts = list()
    for c in chunks:
        if 'shape' in dir(input):
            out_chunks.append(input[i:i + c, :])
        else:
            out_chunks.append(input[i:i + c])
        starts.append(i)
        i += c
    return out_chunks,starts
