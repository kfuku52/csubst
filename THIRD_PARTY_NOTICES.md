# Third-Party Notices

## pyvolve (vendored under `csubst/_vendor/pyvolve`)

- Upstream project: https://github.com/sjspielman/pyvolve
- Vendored release: 1.1.0 (`6145db82049941d559c9d7ba9756525fa6419ea1`)
- Copyright: (c) 2015 Stephanie J. Spielman
- License: BSD-2-Clause-style terms (see `csubst/_vendor/pyvolve/LICENSE.txt`)

CSUBST vendors pyvolve source code for the simulation backend.
Local integration notes and the update procedure are documented in
`csubst/_vendor/pyvolve/README.csubst.md`.

## Biopython codon-table material

- Upstream project: https://github.com/biopython/biopython
- Copyright: (c) 1999-2024, The Biopython Contributors
- License: Biopython License Agreement or BSD-3-Clause, as applicable
- Full license text: `licenses/BIOPYTHON_LICENSE.rst`

`csubst/genetic_code.py` contains codon-table material copied and modified
from Biopython. The original attribution remains in that source file.

## Structural alphabet predictors

`csubst/structural_prediction.py` adapts the ProstT5 encoder-CNN inference
architecture and reconstructs ESM3Di checkpoints with Transformers and PEFT.
Model weights are downloaded separately and are not bundled with CSUBST.

- ProstT5: https://github.com/mheinzinger/ProstT5
  (`3f6c0666ac61d1025ce9473e34d3f67fc893a589`), Copyright (c) 2023 Michael Heinzinger.
- ESM3Di: https://github.com/DessimozLab/ESM3di
  (`2dabc9dabd1ffdb78e7c0aff9907116b51de0d4c`), Copyright (c) 2026 ESM3di Contributors.
- Both upstream implementations use the MIT license, reproduced in
  `licenses/STRUCTURAL_PREDICTORS_LICENSE.txt`.
