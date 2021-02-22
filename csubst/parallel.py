def get_chunks(input, threads):
    if 'shape' in dir(input):
        chunks = [(input.shape[0] + i) // threads for i in range(threads)]
    else:
        chunks = [(len(input) + i) // threads for i in range(threads)]
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
