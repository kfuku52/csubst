import gzip
import os
import shutil

from csubst import runtime


_FASTA_SUFFIXES = ('.fa', '.fasta', '.faa', '.fna')
_FASTA_GZ_SUFFIXES = tuple([suffix + '.gz' for suffix in _FASTA_SUFFIXES])
_IQTREE_INTERMEDIATE_SUFFIXES = ('.iqtree', '.log', '.rate', '.state', '.treefile')


def _is_plain_fasta_file_name(file_name):
    return str(file_name).lower().endswith(_FASTA_SUFFIXES)


def _is_gzipped_fasta_file_name(file_name):
    return str(file_name).lower().endswith(_FASTA_GZ_SUFFIXES)


def _is_iqtree_intermediate_file_name(file_name):
    lower_name = str(file_name).lower()
    return lower_name.startswith('alignment.fa') and lower_name.endswith(_IQTREE_INTERMEDIATE_SUFFIXES)


def _copy_file_as_gzip(path_from, path_to_gz):
    with open(path_from, mode='rb') as src, gzip.open(path_to_gz, mode='wb') as dst:
        shutil.copyfileobj(src, dst)


def _copy_dataset_files(name, dir_dataset, output_dir='.', iqtree_outdir=None, force=False):
    name = str(name)
    output_dir = os.path.abspath(str(output_dir))
    if iqtree_outdir is None:
        iqtree_outdir = os.path.join(output_dir, 'csubst_iqtree')
    layout = runtime.ensure_iqtree_layout({'iqtree_outdir': iqtree_outdir}, create_dir=True)
    iqtree_outdir = layout['iqtree_outdir']
    alignment_target = os.path.join(output_dir, 'alignment.fa.gz')
    iqtree_prefix = runtime.infer_iqtree_output_prefix(
        alignment_file=alignment_target,
        iqtree_outdir=iqtree_outdir,
    )
    files = sorted([f for f in os.listdir(dir_dataset) if f.startswith(name + '.')])
    copy_plan = []
    for file in files:
        new_file_name = file.replace(name + '.', '', 1)
        path_from = os.path.join(dir_dataset, file)
        if _is_plain_fasta_file_name(new_file_name):
            output_file_name = new_file_name + '.gz'
            path_to = os.path.join(output_dir, output_file_name)
            operation = 'gzip'
        elif _is_iqtree_intermediate_file_name(new_file_name):
            suffix = next(
                suffix
                for suffix in _IQTREE_INTERMEDIATE_SUFFIXES
                if new_file_name.lower().endswith(suffix)
            )
            path_to = iqtree_prefix + suffix
            operation = 'copy'
        else:
            output_file_name = new_file_name
            if _is_gzipped_fasta_file_name(new_file_name):
                output_file_name = new_file_name
            path_to = os.path.join(output_dir, output_file_name)
            operation = 'copy'
        copy_plan.append((path_from, path_to, operation))
    existing = [path_to for _, path_to, _ in copy_plan if os.path.lexists(path_to)]
    unsafe_existing = [
        path for path in existing if os.path.islink(path) or not os.path.isfile(path)
    ]
    if unsafe_existing:
        raise FileExistsError(
            'Dataset destination exists but is not a regular file: {}'.format(
                unsafe_existing[0]
            )
        )
    if existing and not bool(force):
        examples = ', '.join(os.path.relpath(path, start=output_dir) for path in existing[:10])
        raise FileExistsError(
            'Dataset output already exists ({}). Use --force yes to overwrite.'.format(examples)
        )
    for path_from, path_to, operation in copy_plan:
        relative = os.path.relpath(path_to, start=output_dir)
        print(f"Copying {name} file: {relative}")
        if operation == 'gzip':
            _copy_file_as_gzip(path_from=path_from, path_to_gz=path_to)
        else:
            shutil.copy(path_from, path_to)


def main_dataset(g):
    dir_csubst = os.path.dirname(os.path.abspath(__file__))
    dir_dataset = os.path.join(dir_csubst, 'dataset')
    _copy_dataset_files(
        name=g['name'],
        dir_dataset=dir_dataset,
        output_dir='.',
        iqtree_outdir=g.get('iqtree_outdir', None),
        force=bool(g.get('force', False)),
    )
