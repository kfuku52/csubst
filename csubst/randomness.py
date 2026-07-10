import hashlib

import numpy as np


DEFAULT_RANDOM_SEED = 1


def normalize_seed(value, default=DEFAULT_RANDOM_SEED, param_name="--random_seed"):
    if value is None:
        value = default
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} should be -1 or a non-negative integer.".format(param_name))
    if isinstance(value, (int, np.integer)):
        seed = int(value)
    elif isinstance(value, str):
        token = value.strip()
        digits = token[1:] if token.startswith("-") else token
        if (not digits.isdigit()) or (digits == ""):
            raise ValueError("{} should be -1 or a non-negative integer.".format(param_name))
        seed = int(token)
    else:
        raise ValueError("{} should be -1 or a non-negative integer.".format(param_name))
    if seed < -1:
        raise ValueError("{} should be -1 or >= 0.".format(param_name))
    return seed


def _component_words(components):
    payload = "\x1f".join(str(component) for component in components).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return np.frombuffer(digest, dtype="<u4").astype(np.uint32, copy=False).tolist()


def derive_seed(base_seed, *components):
    base_seed = normalize_seed(base_seed)
    if base_seed < 0:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    entropy = [int(base_seed)] + [int(word) for word in _component_words(components)]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])


def generator(base_seed, *components):
    return np.random.default_rng(derive_seed(base_seed, *components))


def configured_seed(g, key="random_seed", default=DEFAULT_RANDOM_SEED):
    if g is None:
        return normalize_seed(default)
    return normalize_seed(g.get(key, default), default=default, param_name="--" + str(key))


def next_seed(g, namespace, *components):
    base_seed = configured_seed(g)
    if g is None:
        return derive_seed(base_seed, namespace, 0, *components)
    counters = dict(g.get("_random_stream_counters", {}))
    namespace = str(namespace)
    counter = int(counters.get(namespace, 0))
    counters[namespace] = counter + 1
    g["_random_stream_counters"] = counters
    return derive_seed(base_seed, namespace, counter, *components)


def next_generator(g, namespace, *components):
    return np.random.default_rng(next_seed(g, namespace, *components))
