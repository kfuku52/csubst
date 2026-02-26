import gzip
import os
import shutil


_FASTA_SUFFIXES = ('.fa', '.fasta', '.faa', '.fna')
_FASTA_GZ_SUFFIXES = tuple([suffix + '.gz' for suffix in _FASTA_SUFFIXES])


def _is_plain_fasta_file_name(file_name):
    return str(file_name).lower().endswith(_FASTA_SUFFIXES)


def _is_gzipped_fasta_file_name(file_name):
    return str(file_name).lower().endswith(_FASTA_GZ_SUFFIXES)


def _copy_file_as_gzip(path_from, path_to_gz):
    with open(path_from, mode='rb') as src, gzip.open(path_to_gz, mode='wb') as dst:
        shutil.copyfileobj(src, dst)


def _copy_dataset_files(name, dir_dataset, output_dir='.'):
    name = str(name)
    output_dir = str(output_dir)
    files = sorted([f for f in os.listdir(dir_dataset) if f.startswith(name + '.')])
    for file in files:
        new_file_name = file.replace(name + '.', '', 1)
        path_from = os.path.join(dir_dataset, file)
        if _is_plain_fasta_file_name(new_file_name):
            output_file_name = new_file_name + '.gz'
            path_to = os.path.join(output_dir, output_file_name)
            print(f"Copying {name} file: {output_file_name}")
            _copy_file_as_gzip(path_from=path_from, path_to_gz=path_to)
            continue
        output_file_name = new_file_name
        if _is_gzipped_fasta_file_name(new_file_name):
            output_file_name = new_file_name
        path_to = os.path.join(output_dir, output_file_name)
        print(f"Copying {name} file: {output_file_name}")
        shutil.copy(path_from, path_to)


def main_dataset(g):
    dir_csubst = os.path.dirname(os.path.abspath(__file__))
    dir_dataset = os.path.join(dir_csubst, 'dataset')
    _copy_dataset_files(name=g['name'], dir_dataset=dir_dataset, output_dir='.')
