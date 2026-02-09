"""ETE compatibility helpers.

This module provides a small compatibility layer so CSUBST can run with either
ete4 (preferred) or ete3 (fallback for older environments).
"""

from __future__ import annotations

_ete4_import_error = None
_ete3_import_error = None
_backend = None

try:
    import ete4 as _ete_mod  # type: ignore[import-not-found]
    _backend = "ete4"
except Exception as exc:  # pragma: no cover - backend-specific
    _ete4_import_error = exc
    _ete_mod = None

if _backend is None:
    try:
        import ete3 as _ete_mod  # type: ignore[import-not-found]
        _backend = "ete3"
    except Exception as exc:  # pragma: no cover - backend-specific
        _ete3_import_error = exc

if _backend is None:  # pragma: no cover - backend-specific
    raise ImportError(
        "Failed to import both ete4 and ete3. "
        f"ete4 error={_ete4_import_error!r}, ete3 error={_ete3_import_error!r}"
    )


def backend_name():
    return _backend


def PhyloNode(source, format=1):
    if _backend == "ete4":
        return _ete_mod.Tree(source, parser=format)
    return _ete_mod.PhyloNode(source, format=format)


def is_root(node):
    value = getattr(node, "is_root")
    return value() if callable(value) else bool(value)


def is_leaf(node):
    value = getattr(node, "is_leaf")
    return value() if callable(value) else bool(value)


def get_leaf_names(node):
    if hasattr(node, "leaf_names"):
        return list(node.leaf_names())
    return node.get_leaf_names()


def get_leaves(node):
    if hasattr(node, "leaves"):
        return list(node.leaves())
    return node.get_leaves()


def get_children(node):
    if hasattr(node, "children"):
        return list(node.children)
    return node.get_children()


def get_sisters(node):
    if hasattr(node, "sisters"):
        return list(node.sisters())
    return node.get_sisters()


def get_descendants(node):
    if hasattr(node, "descendants"):
        return list(node.descendants())
    return node.get_descendants()


def iter_leaves(node):
    if hasattr(node, "leaves"):
        return node.leaves()
    return node.iter_leaves()


def iter_ancestors(node):
    if hasattr(node, "ancestors"):
        return node.ancestors()
    return node.iter_ancestors()


def get_common_ancestor(tree, targets):
    if hasattr(tree, "common_ancestor"):
        return tree.common_ancestor(targets)
    return tree.get_common_ancestor(targets)


def get_distance(node, target, topology_only=False):
    if _backend == "ete4":
        return node.get_distance(node, target, topological=topology_only)
    return node.get_distance(target=target, topology_only=topology_only)


def get_tree_root(tree):
    if hasattr(tree, "root"):
        return tree.root
    return tree.get_tree_root()


def add_features(node, **kwargs):
    if hasattr(node, "add_props"):
        node.add_props(**kwargs)
        return None
    return node.add_features(**kwargs)


def _read_fasta(path):
    seq_dict = {}
    current_name = None
    current_alias = None
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_name = line[1:]
                current_alias = current_name.split(" ", 1)[0]
                seq_dict[current_name] = ""
                if (current_alias != current_name) and (current_alias not in seq_dict):
                    seq_dict[current_alias] = ""
                continue
            if current_name is None:
                raise ValueError(f"Invalid FASTA format: sequence line before header in {path}")
            seq_dict[current_name] += line
            if (current_alias is not None) and (current_alias != current_name):
                seq_dict[current_alias] += line
    return seq_dict


def set_prop(node, key, value):
    if (_backend == "ete4") and hasattr(node, "add_props"):
        node.add_props(**{key: value})
        return value
    setattr(node, key, value)
    return value


def get_prop(node, key, default=None):
    if (_backend == "ete4") and hasattr(node, "props"):
        return node.props.get(key, default)
    return getattr(node, key, default)


def has_prop(node, key):
    if (_backend == "ete4") and hasattr(node, "props"):
        return key in node.props
    return hasattr(node, key)


def del_prop(node, key):
    if (_backend == "ete4") and hasattr(node, "props"):
        if key in node.props:
            del node.props[key]
        return None
    if hasattr(node, key):
        delattr(node, key)
    return None


def link_to_alignment(tree, alignment, alg_format="fasta"):
    if _backend != "ete4":
        tree.link_to_alignment(alignment=alignment, alg_format=alg_format)
        return None
    if alg_format.lower() != "fasta":
        raise ValueError(f"Unsupported alignment format for ete4 compatibility: {alg_format}")
    seq_dict = _read_fasta(alignment)
    for leaf in iter_leaves(tree):
        seq = seq_dict.get(leaf.name)
        if seq is None:
            # Common fallback: FASTA headers may contain extra description fields.
            seq = seq_dict.get(leaf.name.split(" ", 1)[0], "")
        set_prop(leaf, "sequence", seq)
    return None


def write_tree(tree, format=1, outfile=None):
    if _backend == "ete4":
        text = tree.write(parser=format, format_root_node=True)
        if outfile is None:
            return text
        with open(outfile, "w") as handle:
            handle.write(text)
        return None
    return tree.write(format=format, outfile=outfile)


def get_treeview_module():
    required = ("TreeStyle", "NodeStyle", "TextFace", "add_face_to_node")
    if _backend == "ete4":
        try:
            from ete4 import treeview as tv  # type: ignore[import-not-found]
            if all(hasattr(tv, attr) for attr in required):
                return tv
        except Exception:  # pragma: no cover - optional feature
            tv = None
        # Fallback: use ete3 treeview APIs when ete4 treeview is unavailable.
        try:
            import ete3 as ete3_mod  # type: ignore[import-not-found]
            if all(hasattr(ete3_mod, attr) for attr in required):
                return ete3_mod
        except Exception:  # pragma: no cover - optional feature
            return None
        return None
    if all(hasattr(_ete_mod, attr) for attr in required):
        return _ete_mod
    return None
