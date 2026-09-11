## End-to-end null evaluation for #46 (2026-09-10)

**The Q-based reporting rule gave 0/200 false-positive searches in every tested setting/tree regime (each pointwise 95% CI: 0–1.83%). The selected, unadjusted P-based rule reached 13.0% in the long-tree hypergeom/min_sub_pp=0.05 condition (26/200; 95% CI: 8.67–18.47%).**

This experiment held the production P/Q implementation unchanged and repeated the outer workflow: independent codon alignment generation → branch-length/model-parameter fitting → ASR/site-rate estimation → CSUBST branch-combination search and candidate selection.

### Design and reporting rules

- Tested source: [72c3c45](https://github.com/kfuku52/csubst/commit/72c3c4543e66e21ae48aaa7bec80b66e953d18da), CSUBST 1.15.2. The frozen snapshot of 101 tracked source files matched this commit.
- 200 independent alignments per tree regime, 400 alignments total; 8 tips, 400 codons. The long-tree regime scales every branch of the short tree by 3.
- Homogeneous biological null: stationary GY process, omega=0.2, kappa=2.5, F3X4 codon frequencies, four discrete-Gamma site-rate categories with shape=0.6; no branch/site-specific convergent process. Simulation used vendored Pyvolve with independent simulation/inference seeds.
- Each alignment was fitted afresh with [IQ-TREE 3.1.4](https://github.com/iqtree/iqtree3/releases/tag/v3.1.4), GY+F3X4+G4. The known rooted topology was supplied **without branch lengths**. Branch lengths, omega, kappa, frequencies, Gamma shape, ASR and site rates were re-estimated. True ancestral states/parameters were not supplied to inference. Fits were shared only between the four settings on the same alignment.
- Search: marginal endpoint posteriors; urn; ASRV=each; any2spe; exhaustive K=2; max_arity=3; exclude sister pairs; default expansion cutoff `OCNany2spe>=2` and `omegaCany2spe>=5`; max_combination=10000. Each test used 3,999 fixed null draws, stochastic rounding and 12-digit TSV precision. Independent long-tail maps used 1,000 additional draws.
- The primary reporting rule was: at least one row passing those effect-size cutoffs with production **Q<=0.05**. The secondary rule substituted unadjusted **P<=0.05**. These are explicit reporting rules applied to the search outputs. Under this complete null, their per-alignment discovery probability is both FWER and FDR for one search run.
- Denominators and exact two-sided 95% binomial intervals use independent alignments, **not correlated branch rows**. Intervals are pointwise, not simultaneous across settings. Production BH families remain per arity.

### Results

All eight setting/regime cells had **0/200 at Q<=0.05**, both before and after the effect-size cutoff. Thus the zero-Q result also holds without requiring a final candidate to pass the cutoff.

| Tree | Setting | Selected P<=0.05: false-positive searches (95% CI) | Any unadjusted P<=0.05 before cutoff | Prespecified a/e pair P<=0.05 | Alignments with cutoff-qualified pairs |
| --- | --- | --- | --- | --- | --- |
| short | `hypergeom_pp0` | 2/200 = 1.0% (0.12–3.57%) | 58/200 | 1/200 | 4/200 |
| short | `hypergeom_pp005` | 1/200 = 0.5% (0.01–2.75%) | 69/200 | 1/200 | 4/200 |
| short | `poisson_symmetric_pp005` | 0/200 = 0.0% (0.00–1.83%) | 14/200 | 0/200 | 0/200 |
| short | `poisson_independent_pp005` | 0/200 = 0.0% (0.00–1.83%) | 10/200 | 0/200 | 0/200 |
| long | `hypergeom_pp0` | 14/200 = 7.0% (3.88–11.47%) | 62/200 | 1/200 | 14/200 |
| long | `hypergeom_pp005` | 26/200 = 13.0% (8.67–18.47%) | 77/200 | 2/200 | 26/200 |
| long | `poisson_symmetric_pp005` | 0/200 = 0.0% (0.00–1.83%) | 18/200 | 0/200 | 0/200 |
| long | `poisson_independent_pp005` | 0/200 = 0.0% (0.00–1.83%) | 11/200 | 0/200 | 0/200 |

Setting names: `hypergeom_pp0/pp005` use no pseudocount or long-tail correction; `poisson_symmetric_pp005` uses symmetric alpha=1 on observed and expected counts without long-tail correction; `poisson_independent_pp005` adds independent-null long-tail maps. Both Poisson settings use min_sub_pp=0.05.

For the prespecified a/e pair, rejection was 0–2/200 (0–1%). Screening all 52 pairs and taking any unadjusted P<=0.05 yielded 5–38.5% of analyses. Those are different decision rules. In particular, min_sub_pp=0.05 did not turn the selected-P reporting rule into a 5% run-level test in the long-tree case. No significance claim is made for the paired difference between pp0 and pp005.

**Selection coverage limit:** every search ran the production K=3 candidate-selection step, but **no eligible triplet was generated**. The study therefore covers branch-pair candidate selection and search stopping, and supplies no empirical calibration evidence for post-selected K>=3 P/Q values. The Poisson/symmetric settings had no cutoff-qualified pairs; power under a convergence alternative was not tested.

### Audit and interpretation

- 400 fresh IQ-TREE fits, 1,600 searches, 1,600 output tables, **83,200 finite P/Q pairs**, zero failed or omitted replicates. Every search evaluated 52 branch pairs. The fixed a/e pair was finite in every replicate.
- Independent BH recalculation agreed with every output table; output-based cutoff counts agreed with the production `num_qualified_all` counts. All alignment, fitted-artifact and search-output hashes were verified.
- Total outer wall time: 30.08 min with four workers. Python 3.10.14 (x86_64), NumPy 1.26.4, SciPy 1.15.2, pandas 2.2.3; Apple M2 Max; native arm64 IQ-TREE. Runtime is descriptive on a shared host.
- Repository validation: **2,120 passed, 5 skipped**; Ruff, hygiene, documentation and configured mypy checks passed. Skips were optional PyTorch/gemmi requirements. One existing requests dependency warning remained.

The tested Q-based workflow was conservative in these matched-model eight-tip scenarios. This does not establish general calibration or useful power. Larger trees, model misspecification, topology/model-family selection, alignment error, and successful higher-arity selection remain untested. I would keep #46 open for those questions; an unadjusted selected P-value should not be interpreted as a 5% full-search error probability.

Compact artifacts, including every evaluated row and simulated alignment, are retained locally in `reports/issue46_pipeline_20260910/`. The complete runnable evaluator and per-replicate event lists are embedded below so this comment includes the method and rejection counts without depending on those local files.

### Reproduction

At the tested source revision, save the evaluator below as `.github/scripts/omega_pipeline_fpr.py` and run in a Python environment with the reported runtime dependencies and IQ-TREE 3.1.4:

```bash
python .github/scripts/omega_pipeline_fpr.py \
  --workdir /tmp/csubst_issue46_pipeline_full_20260910 \
  --iqtree /path/to/iqtree3 --replicates 200 --workers 4 --seed 4609201
```

Evaluator SHA256: `cf27e6b889271006966fa45faa6930fad3fccff78887307c9cf70f75e06cf4fd`. IQ-TREE executable SHA256: `c5bc5423664cde56253cf58c8c07dfce86fec8177daf55c50d799e275d0b7a9f`. The work directory must be new. The pilot used different seeds and was excluded. An older IQ-TREE 2.3.6 pilot retained supplied branch lengths; the 3.1.4 optimizer was verified before the confirmatory run.

<details>
<summary>Independent replicate IDs for each rejection/selection event</summary>

For each regime, the denominator is 200 and IDs run from 0 through 199. Lists contain the IDs with the named event; all others have that event=false. No replicate failed. This permits independent reconstruction of the reported counts and binomial intervals.

```json
{
  "replicates_per_regime": 200,
  "replicate_ids": "0..199; lists contain the IDs with the named event; all others have event=False",
  "events": {
    "short": {
      "hypergeom_pp0": {
        "any_p05": [
          4,
          10,
          13,
          24,
          28,
          31,
          36,
          43,
          45,
          46,
          47,
          51,
          53,
          59,
          60,
          64,
          69,
          71,
          73,
          76,
          78,
          81,
          84,
          85,
          89,
          92,
          96,
          107,
          108,
          112,
          118,
          122,
          124,
          128,
          136,
          138,
          142,
          143,
          144,
          147,
          149,
          150,
          151,
          154,
          157,
          160,
          161,
          163,
          172,
          174,
          180,
          181,
          182,
          183,
          188,
          193,
          194,
          199
        ],
        "any_q05": [],
        "selected_p05": [
          182,
          194
        ],
        "selected_q05": [],
        "selected": [
          145,
          182,
          190,
          194
        ],
        "fixed_pair_p05": [
          24
        ],
        "fixed_pair_unavailable": [],
        "generated_k3": []
      },
      "hypergeom_pp005": {
        "any_p05": [
          0,
          4,
          6,
          10,
          13,
          18,
          22,
          24,
          28,
          31,
          36,
          37,
          45,
          46,
          47,
          48,
          51,
          53,
          59,
          60,
          65,
          69,
          73,
          76,
          78,
          81,
          84,
          85,
          86,
          89,
          92,
          95,
          96,
          106,
          107,
          111,
          112,
          113,
          118,
          122,
          124,
          126,
          128,
          138,
          142,
          143,
          144,
          146,
          147,
          149,
          150,
          151,
          157,
          160,
          161,
          163,
          169,
          170,
          172,
          174,
          177,
          180,
          181,
          182,
          183,
          188,
          193,
          194,
          199
        ],
        "any_q05": [],
        "selected_p05": [
          194
        ],
        "selected_q05": [],
        "selected": [
          100,
          145,
          190,
          194
        ],
        "fixed_pair_p05": [
          24
        ],
        "fixed_pair_unavailable": [],
        "generated_k3": []
      },
      "poisson_symmetric_pp005": {
        "any_p05": [
          10,
          13,
          28,
          51,
          60,
          69,
          73,
          126,
          138,
          146,
          147,
          151,
          182,
          199
        ],
        "any_q05": [],
        "selected_p05": [],
        "selected_q05": [],
        "selected": [],
        "fixed_pair_p05": [],
        "fixed_pair_unavailable": [],
        "generated_k3": []
      },
      "poisson_independent_pp005": {
        "any_p05": [
          9,
          10,
          13,
          60,
          73,
          101,
          128,
          138,
          182,
          194
        ],
        "any_q05": [],
        "selected_p05": [],
        "selected_q05": [],
        "selected": [],
        "fixed_pair_p05": [],
        "fixed_pair_unavailable": [],
        "generated_k3": []
      }
    },
    "long": {
      "hypergeom_pp0": {
        "any_p05": [
          2,
          4,
          10,
          16,
          17,
          20,
          21,
          24,
          33,
          37,
          42,
          49,
          51,
          59,
          64,
          65,
          67,
          71,
          72,
          87,
          90,
          93,
          94,
          100,
          103,
          104,
          108,
          110,
          113,
          114,
          115,
          118,
          122,
          124,
          125,
          128,
          131,
          132,
          133,
          135,
          137,
          143,
          144,
          147,
          149,
          152,
          157,
          160,
          167,
          169,
          170,
          172,
          175,
          176,
          178,
          179,
          180,
          186,
          188,
          189,
          196,
          197
        ],
        "any_q05": [],
        "selected_p05": [
          42,
          49,
          64,
          72,
          87,
          104,
          108,
          110,
          113,
          152,
          160,
          178,
          179,
          196
        ],
        "selected_q05": [],
        "selected": [
          42,
          49,
          64,
          72,
          87,
          104,
          108,
          110,
          113,
          152,
          160,
          178,
          179,
          196
        ],
        "fixed_pair_p05": [
          94
        ],
        "fixed_pair_unavailable": [],
        "generated_k3": []
      },
      "hypergeom_pp005": {
        "any_p05": [
          2,
          4,
          10,
          16,
          17,
          20,
          21,
          24,
          33,
          37,
          41,
          42,
          44,
          48,
          49,
          51,
          59,
          63,
          64,
          65,
          67,
          71,
          72,
          76,
          77,
          87,
          90,
          93,
          97,
          100,
          103,
          104,
          106,
          108,
          110,
          113,
          114,
          115,
          118,
          121,
          122,
          124,
          125,
          128,
          131,
          132,
          133,
          135,
          137,
          138,
          143,
          144,
          146,
          147,
          149,
          152,
          153,
          157,
          158,
          160,
          163,
          164,
          167,
          169,
          170,
          172,
          175,
          176,
          178,
          179,
          180,
          181,
          186,
          188,
          189,
          196,
          197
        ],
        "any_q05": [],
        "selected_p05": [
          24,
          42,
          49,
          51,
          64,
          72,
          77,
          87,
          93,
          103,
          104,
          108,
          110,
          113,
          114,
          115,
          122,
          144,
          147,
          152,
          160,
          175,
          178,
          179,
          186,
          196
        ],
        "selected_q05": [],
        "selected": [
          24,
          42,
          49,
          51,
          64,
          72,
          77,
          87,
          93,
          103,
          104,
          108,
          110,
          113,
          114,
          115,
          122,
          144,
          147,
          152,
          160,
          175,
          178,
          179,
          186,
          196
        ],
        "fixed_pair_p05": [
          77,
          103
        ],
        "fixed_pair_unavailable": [],
        "generated_k3": []
      },
      "poisson_symmetric_pp005": {
        "any_p05": [
          2,
          51,
          72,
          103,
          104,
          108,
          110,
          113,
          118,
          122,
          124,
          135,
          147,
          149,
          167,
          178,
          186,
          196
        ],
        "any_q05": [],
        "selected_p05": [],
        "selected_q05": [],
        "selected": [],
        "fixed_pair_p05": [],
        "fixed_pair_unavailable": [],
        "generated_k3": []
      },
      "poisson_independent_pp005": {
        "any_p05": [
          21,
          51,
          103,
          104,
          108,
          110,
          113,
          118,
          178,
          186,
          196
        ],
        "any_q05": [],
        "selected_p05": [],
        "selected_q05": [],
        "selected": [],
        "fixed_pair_p05": [],
        "fixed_pair_unavailable": [],
        "generated_k3": []
      }
    }
  }
}
```

</details>

<details>
<summary>Complete runnable evaluator</summary>

```python
#!/usr/bin/env python3
"""Issue #46: independent codon simulation -> IQ-TREE fitting/ASR -> search.

No fitted-null parameters are taken from the generating model. A known rooted
topology is supplied without lengths. Replicate alignments, not branch rows,
are the independent experimental units. This measures calibration; it does
not recalibrate, select, or tune the production tests using the resulting FPR.
"""

import argparse
import concurrent.futures
import contextlib
import gzip
import hashlib
import io
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
TREE = '(((a:0.12,b:0.18):0.09,(c:0.14,d:0.20):0.11):0.10,((e:0.13,f:0.19):0.08,(g:0.16,h:0.22):0.12):0.10);'
TOPOLOGY = re.sub(r':[0-9.]+', '', TREE)
MODEL = 'GY+F3X4+G4'
CUTOFF = 'OCNany2spe,2.0|omegaCany2spe,5.0'
SETTINGS = {
    'hypergeom_pp0': dict(null='hypergeom', min_pp=0., alpha=0., calibrated=False),
    'hypergeom_pp005': dict(null='hypergeom', min_pp=.05, alpha=0., calibrated=False),
    'poisson_symmetric_pp005': dict(null='poisson', min_pp=.05, alpha=1., calibrated=False),
    'poisson_independent_pp005': dict(null='poisson', min_pp=.05, alpha=1., calibrated=True),
}
METRICS = ('any_p05', 'any_q05', 'selected_p05', 'selected_q05', 'top_selected_p05')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(seed, regime_id, replicate, channel):
    return int(np.random.SeedSequence([seed, regime_id, replicate, channel]).generate_state(1)[0] % 2147483646 + 1)


def simulate_alignment(directory, scale, sites, seed):
    from csubst._vendor import pyvolve

    codons = [''.join(c) for c in itertools.product('ACGT', repeat=3)
              if ''.join(c) not in ('TAA', 'TAG', 'TGA')]
    # Positional nucleotide frequencies make the generator a member of the
    # fitted F3X4 family without handing those frequencies to inference.
    base_pi = np.array([[.30, .20, .20, .30], [.20, .30, .30, .20], [.25, .15, .35, .25]])
    pi = np.array([np.prod([base_pi[i, 'ACGT'.index(b)] for i, b in enumerate(c)]) for c in codons])
    pi /= pi.sum()
    newick = re.sub(r':([0-9.]+)', lambda m: ':' + str(float(m.group(1)) * scale), TREE)
    model = pyvolve.Model('GY', {'omega': .2, 'kappa': 2.5, 'state_freqs': pi}, alpha=.6, num_categories=4)
    if list(model.code) != codons:
        raise AssertionError('Generator codon ordering differs from specified frequencies')
    if not np.allclose(pi @ model.matrix, 0., atol=1e-10):
        raise AssertionError('Generator is not stationary at specified codon frequencies')
    if not np.isclose(-np.dot(pi, np.diag(model.matrix)), 1.):
        raise AssertionError('Generator branch-length scaling is not one substitution/codon')
    if not np.isclose(np.dot(model.rate_factors, model.rate_probs), 1.):
        raise AssertionError('Site rates are not normalized')
    evolve = pyvolve.Evolver(tree=pyvolve.read_tree(tree=newick),
                            partitions=pyvolve.Partition(models=model, size=sites))
    evolve(seqfile=str(directory / 'alignment.fa'), seqfmt='fasta', write_anc=False,
           ratefile=False, infofile=False, seed=seed)
    (directory / 'topology.nwk').write_text(TOPOLOGY + '\n')
    return dict(tree=newick, omega=.2, kappa=2.5, gamma_shape=.6,
                frequencies=pi.tolist(), rates=model.rate_factors.tolist(),
                rate_probabilities=model.rate_probs.tolist())


def run_command(command, cwd, label, environment):
    wrapper = ['/usr/bin/time', '-l', '-p'] if sys.platform == 'darwin' else ['/usr/bin/time', '-v']
    started = time.perf_counter()
    with (cwd / (label + '.stdout.log')).open('w') as out, (cwd / (label + '.stderr.log')).open('w') as err:
        process = subprocess.run(wrapper + command, cwd=cwd, env=environment,
                                 stdout=out, stderr=err, timeout=900)
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError('{} failed with code {}; see {}'.format(label, process.returncode, cwd))
    stderr = (cwd / (label + '.stderr.log')).read_text()
    if sys.platform == 'darwin':
        matched = re.search(r'^\s*(\d+)\s+maximum resident set size$', stderr, re.M)
        rss = int(matched[1]) if matched else None
    else:
        matched = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', stderr)
        rss = int(matched[1]) * 1024 if matched else None
    if rss is None:
        raise RuntimeError('Missing peak RSS for ' + label)
    return dict(seconds=elapsed, peak_rss_bytes=rss)


def read_fit(directory):
    log = (directory / 'fit.log').read_text()
    fields = {
        'omega': r'Nonsynonymous/synonymous ratio \(omega\):\s*([0-9.eE+-]+)',
        'kappa': r'Transition/transversion ratio \(kappa\):\s*([0-9.eE+-]+)',
        'gamma_shape': r'Gamma shape alpha:\s*([0-9.eE+-]+)',
        'log_likelihood': r'Optimal log-likelihood:\s*([0-9.eE+-]+)',
    }
    result = {}
    for name, pattern in fields.items():
        matches = re.findall(pattern, log)
        if not matches:
            raise RuntimeError('Fitted ' + name + ' missing')
        result[name] = float(matches[-1])
    fitted_tree = (directory / 'fit.treefile').read_text()
    lengths = [float(x) for x in re.findall(r':([0-9.eE+-]+)', fitted_tree)]
    if len(lengths) != 13 or not all(np.isfinite(lengths)) or min(lengths) < 0:
        raise AssertionError('Invalid eight-tip fitted tree')
    result['tree_length'] = sum(lengths)
    result['fitted_tree'] = fitted_tree.strip()
    state = pd.read_csv(directory / 'fit.state', sep='\t', comment='#')
    probabilities = state.loc[:, state.columns.str.startswith('p_')].to_numpy(float)
    if probabilities.shape[1] != 61 or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-3):
        raise AssertionError('Invalid reconstructed codon posterior probabilities')
    result['mean_max_asr_posterior'] = float(probabilities.max(axis=1).mean())
    result['hashes'] = {ext: sha256(directory / ('fit.' + ext))
                        for ext in ('treefile', 'state', 'rate', 'iqtree')}
    return result


def search_command(directory, output, setting, draws, seed, iqtree):
    command = [sys.executable, '-m', 'csubst', 'search',
               '--alignment_file', str(directory / 'alignment.fa'),
               '--rooted_tree_file', str(directory / 'topology.nwk'),
               '--iqtree_exe', iqtree, '--iqtree_model', MODEL,
               '--outdir', str(output), '--output_prefix', 'csubst',
               '--max_arity', '3', '--exhaustive_until', '2', '--max_combination', '10000',
               '--exclude_sister_pair', 'yes', '--cutoff_stat', CUTOFF,
               '--output_stat', 'any2spe', '--expectation_method', 'urn',
               '--substitution_posterior', 'marginal', '--ml_anc', 'no',
               '--asrv', 'each', '--asrv_dirichlet_alpha', '0',
               '--min_sub_pp', str(setting['min_pp']),
               '--calc_omega_pvalue', 'yes', '--omega_pvalue_null_model', setting['null'],
               '--omega_pvalue_niter_schedule', str(draws), '--omega_pvalue_rounding', 'stochastic',
               '--pseudocount_mode', 'symmetric' if setting['alpha'] else 'none',
               '--pseudocount_alpha', str(setting['alpha']), '--pseudocount_target', 'both',
               '--calibrate_longtail', 'yes' if setting['calibrated'] else 'no',
               '--longtail_method', 'independent_null', '--longtail_null_niter', '1000',
               '--random_seed', str(seed), '--threads', '1', '--float_digit', '12',
               '--branch_dist', 'no', '--b', 'no', '--s', 'no', '--cs', 'no',
               '--bs', 'no', '--cbs', 'no', '--cb', 'yes']
    for ext in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        command.extend(['--iqtree_' + ext, str(directory / ('fit.' + ext))])
    return command


def summarize_frame(frame):
    p = frame['pomegaCany2spe'].to_numpy(float)
    q = frame['qomegaCany2spe'].to_numpy(float)
    omega = frame['omegaCany2spe'].to_numpy(float)
    selected = (frame['OCNany2spe'].to_numpy(float) >= 2.) & (omega >= 5.)
    if np.any(np.isfinite(p) & ((p < 0) | (p > 1))) or np.any(np.isfinite(q) & ((q < 0) | (q > 1))):
        raise AssertionError('P/Q values outside [0,1]')
    result = dict(rows=len(frame), finite_p=int(np.isfinite(p).sum()),
                  finite_q=int(np.isfinite(q).sum()), selected=int(selected.sum()),
                  selected_finite_p=int((selected & np.isfinite(p)).sum()),
                  selected_finite_q=int((selected & np.isfinite(q)).sum()),
                  any_p05=bool(np.any(p <= .05)), any_q05=bool(np.any(q <= .05)),
                  selected_p05=bool(np.any(selected & (p <= .05))),
                  selected_q05=bool(np.any(selected & (q <= .05))))
    indices = np.flatnonzero(selected)
    if len(indices):
        # Explicit secondary winner rule, with stable row-order tie breaking.
        top = indices[np.argmax(omega[indices])]
        result['top_selected_p05'] = bool(p[top] <= .05)
    else:
        result['top_selected_p05'] = False
    return result


def summarize_search(output, prespecified_pair):
    arities = {}
    pair_result = None
    for arity in (2, 3):
        path = output / ('csubst_cb_' + str(arity) + '.tsv')
        if not path.exists():
            if arity == 2:
                raise RuntimeError('Missing exhaustive branch-pair output')
            continue
        frame = pd.read_csv(path, sep='\t')
        arities[str(arity)] = summarize_frame(frame)
        arities[str(arity)]['sha256'] = sha256(path)
        if arity == 2:
            ids = frame[['branch_id_1', 'branch_id_2']].to_numpy(int)
            target = np.all(np.sort(ids, axis=1) == np.sort(prespecified_pair), axis=1)
            if target.sum() != 1:
                raise AssertionError('Prespecified a/e pair not found exactly once')
            p = float(frame.loc[target, 'pomegaCany2spe'].iloc[0])
            pair_result = dict(finite=bool(np.isfinite(p)), rejected=bool(p <= .05),
                               p=p if np.isfinite(p) else None)
    return dict(arities=arities, fixed_pair=pair_result,
                **{metric: any(row[metric] for row in arities.values()) for metric in METRICS},
                selected=any(row['selected'] > 0 for row in arities.values()))


def run_replicate(job):
    sys.path.insert(0, job['source'])
    from csubst import ete, tree

    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=False)
    seed = seed_for(job['seed'], job['regime_id'], job['replicate'], 0)
    with contextlib.redirect_stdout(io.StringIO()):
        generator = simulate_alignment(directory, job['scale'], job['sites'], seed)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=job['source'], OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    command = [job['iqtree'], '-s', str(directory / 'alignment.fa'), '-st', 'CODON',
               '-m', MODEL, '-te', str(directory / 'topology.nwk'), '-nt', '1',
               '-seed', str(seed_for(job['seed'], job['regime_id'], job['replicate'], 1)),
               '-asr', '-wsr', '-pre', str(directory / 'fit')]
    iqtree_runtime = run_command(command, directory, 'iqtree', environment)
    fit = read_fit(directory)
    parsed = tree.add_numerical_node_labels(ete.PhyloNode(TOPOLOGY, format=1))
    labels = {n.name: int(ete.get_prop(n, 'numerical_label')) for n in parsed.traverse()}
    result = dict(regime=job['regime'], replicate=job['replicate'], generation_seed=seed,
                  alignment_sha256=sha256(directory / 'alignment.fa'), fit=fit,
                  iqtree_runtime=iqtree_runtime, settings={})
    commands = {'iqtree': command}
    for name, setting in SETTINGS.items():
        output = directory / name
        command = search_command(directory, output, setting, job['draws'],
                                 seed_for(job['seed'], job['regime_id'], job['replicate'], 2), job['iqtree'])
        measured = run_command(command, directory, name, environment)
        result['settings'][name] = dict(runtime=measured,
                                       **summarize_search(output, [labels['a'], labels['e']]))
        commands[name] = command
    (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    (directory / 'generator.json').write_text(json.dumps(generator, indent=2) + '\n')
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def binomial_result(events):
    n, k = len(events), sum(events)
    interval = binomtest(k, n).proportion_ci()
    return dict(rejections=int(k), replicates=n, rate=k / n,
                interval_95=[interval.low, interval.high])


def aggregate(results):
    rows = []
    for regime in sorted({r['regime'] for r in results}):
        members = [r for r in results if r['regime'] == regime]
        for setting in SETTINGS:
            cases = [r['settings'][setting] for r in members]
            row = dict(regime=regime, setting=setting)
            for metric in METRICS:
                row[metric] = binomial_result([c[metric] for c in cases])
            row['datasets_with_candidates'] = sum(c['selected'] for c in cases)
            finite = [c['fixed_pair'] for c in cases if c['fixed_pair']['finite']]
            row['fixed_pair_finite'] = len(finite)
            row['fixed_pair_unconditional'] = binomial_result([c['fixed_pair']['rejected'] for c in cases])
            row['fixed_pair_conditional'] = binomial_result([c['rejected'] for c in finite]) if finite else None
            row['median_search_seconds'] = float(np.median([c['runtime']['seconds'] for c in cases]))
            row['max_search_rss_bytes'] = max(c['runtime']['peak_rss_bytes'] for c in cases)
            row['total_evaluated_rows'] = sum(a['rows'] for c in cases for a in c['arities'].values())
            row['total_finite_p_rows'] = sum(a['finite_p'] for c in cases for a in c['arities'].values())
            row['total_selected_rows'] = sum(a['selected'] for c in cases for a in c['arities'].values())
            row['datasets_with_k3'] = sum('3' in c['arities'] for c in cases)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True, help='New directory; never overwrites an existing run.')
    parser.add_argument('--iqtree', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=200, help='Independent alignments per tree regime.')
    parser.add_argument('--sites', type=int, default=400)
    parser.add_argument('--draws', type=int, default=3999)
    parser.add_argument('--seed', type=int, default=4609201)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if min(args.replicates, args.sites, args.workers) < 1 or args.draws < 999:
        parser.error('Require positive replicates/sites/workers and at least 999 test draws')
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    source = workdir / 'source'
    source.mkdir()
    shutil.copytree(ROOT / 'csubst', source / 'csubst', ignore=shutil.ignore_patterns('__pycache__', 'dataset'))
    shutil.copy2(__file__, source / 'omega_pipeline_fpr.py')
    original_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'csubst').rglob('*')
                       if p.is_file() and 'dataset' not in p.parts and '__pycache__' not in p.parts}
    for name, checksum in original_hashes.items():
        if sha256(source / name) != checksum:
            raise RuntimeError('Source changed while snapshotting ' + name)
    version = subprocess.check_output([str(args.iqtree.resolve()), '--version'], text=True).splitlines()[0]
    metadata = dict(replicates_per_regime=args.replicates, sites=args.sites,
                    test_draws=args.draws, seed=args.seed, workers=args.workers,
                    generating_tree=TREE, scale_by_regime={'short': 1., 'long': 3.},
                    inference_topology=TOPOLOGY, inference_model=MODEL,
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_hashes=original_hashes, script_sha256=sha256(__file__),
                    iqtree_version=version, iqtree_sha256=sha256(args.iqtree),
                    python=platform.python_version(), platform=platform.platform(), settings=SETTINGS,
                    cutoff=CUTOFF, max_arity=3, exhaustive_until=2,
                    primary_metric='Probability of any cutoff-selected row with production q<=0.05 across K=2 and selected K=3. Under the global null this is FWER and FDR for one search run.',
                    caveats='Known rooted topology and fixed model family; parameters/branch lengths/ASR are re-estimated. No topology/model-family selection, alignment error, recombination, or model misspecification. Q families remain the production per-arity families, not one global adjustment across arities. Missing P/Q count as no rejection in unconditional dataset metrics and are reported separately.')
    (workdir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    jobs = [dict(directory=str(workdir / regime / ('replicate_' + str(i).zfill(4))),
                 source=str(source), regime=regime, regime_id=j, replicate=i,
                 scale=scale, sites=args.sites, draws=args.draws, seed=args.seed,
                 iqtree=str(args.iqtree.resolve()))
            for i in range(args.replicates)
            for j, (regime, scale) in enumerate([('short', 1.), ('long', 3.)])]
    results, failures = [], []
    started = time.perf_counter()
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_replicate, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(regime=job['regime'], replicate=job['replicate'], error=str(error)))
            print('{}/{} finished; {} failed; {:.1f}s'.format(len(results) + len(failures), len(jobs), len(failures), time.perf_counter() - started), flush=True)
    results.sort(key=lambda x: (x['regime'], x['replicate']))
    with gzip.open(workdir / 'records.json.gz', 'wt') as handle:
        json.dump(results, handle)
    (workdir / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    if failures:
        raise RuntimeError('Failed replicates retained; do not silently omit them from FPR denominators')
    alignments = [r['alignment_sha256'] for r in results]
    if len(set(alignments)) != len(alignments):
        raise AssertionError('Duplicate simulated alignments')
    summary = dict(elapsed_seconds=time.perf_counter() - started, rows=aggregate(results))
    (workdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
```

</details>
