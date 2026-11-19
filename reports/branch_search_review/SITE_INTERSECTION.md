# Direct K=4 proposals from shared substitution sites

Two changes at different sites can identify a higher-order combination without
first scoring any of its branch pairs or triples. For each (site, derived amino
acid), record the branches with that change. Intersect the lists for two sites.
Any quartet from the intersection has two shared N changes under known states.
With binary events and unlimited enumeration this recovers every quartet with
OCNany2spe ≥ 2. With uncertain endpoints, sample a categorical outcome per
branch and site, propose from several draws, then evaluate proposed tuples on
the original probabilities. The stochastic and budgeted versions can miss hits.

## Small exhaustive comparison

The pilot used independent terminal branches, two possible N destinations per
site, one S destination, and known site-specific independent null probabilities.
It used their products for ECN and ECS. In each of 12 seeds the signal consisted
of four branches changing together at four sites. In the uncertain scenario the
signal was spread over 32 sites with N event probability 0.55 per branch
(32 × 0.55⁴ ≈ 2.93 expected shared changes). In the hotspot condition 2.5% of
sites had background N probability 0.53 versus 0.075 elsewhere. The null
control had no planted changes. No IQ-TREE fit was used.

Every quartet was scored in the reference. A hit required OCNany2spe ≥ 2 and
ωCany2spe ≥ 5. Existing searches used CSUBST's parent selector and union
generator with a cap of 32 parents per expansion. The site procedures could
evaluate at most as many distinct quartets as the corresponding existing search
generated in that seed. The budget counts full candidate evaluations at K=4.
Building indexes, checking site pairs and drawing tuples incur additional costs.

| Scenario | Exhaustive hits across 12 seeds | K=4 budget summed over seeds | `ocn_bound` hits | One-site hits | Two-site hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 branches, 512 sites | 15 | 91 | 12 | 15 | 15 |
| 40 branches, 2,048 sites | 420 | 357 | 12 | 304 | 327 |
| Same size, frequent N changes at hotspots | 87,854 | 756 | 293 | 198 | 299 |
| Same size, uncertain N changes | 51 | 93 | 16 | 44 | 48 |
| Same size, no planted signal | 382 | 364 | 9 | 298 | 312 |

`all_cutoffs` stopped before K=4 for these synthetic inputs. Forcing the
uncertain runs to use only certain events lost the planted group in all 12
seeds with the one-site method. Sampling recovered it in all 12 with either
site method. Under hotspots, the planted group passed the final cutoff in
five seeds. Parent expansion found it in zero, one-site proposals in five and
two-site proposals in three. Recovering more of the exhaustive table also
recovers chance hits, as shown by the no-signal control.

## Candidate generation budgets

One-site generation samples up to 24 quartets from each posting. It retains
quartets drawn from at least two distinct sites within a round, ranking by
repeated support across eight rounds. Two-site generation checks distinct pairs
of postings, ranks quartets by the number of sampled intersections supporting
them, and breaks ties with a seeded shuffle. Both methods cap tuple draws at
65,536 in total. Two-site generation also caps posting-pair checks at 4 million
and processes them in blocks of 4,096. For uncertain data it splits the budget
over eight independent outcome draws. These limits apply after reading the
branch-by-site data. Neither procedure fits substitution models.

The scale check injected the same four-site effect and used a fixed budget of
64 final quartet scores. The table reports recovery of the single injected
quartet, which met both cutoffs in each seed. This panel does not require an
exhaustive table of up to 174,792,640 quartets.

| Branches | Sites | One-site top 64 | Two-site top 64 | Injected quartet proposed by two-site search |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 2,048 | 12/12 | 12/12 | 12/12 |
| 128 | 2,048 | 5/12 | 12/12 | 12/12 |
| 256 | 2,048 | 0/12 | 12/12 | 12/12 |
| 256 | 8,192 | 0/12 | 0/12 | 5/12 |

At 8,192 sites the two-site procedure checked 4 million posting pairs per seed.
Seven planted quartets never entered the candidate list. The other five entered
but ranked below 64. As a separate diagnostic, we counted actual shared N
events across the whole alignment for each proposed quartet using bitsets,
then ranked by this count with the original votes as a tiebreaker. This
recovered four of the five, at a median cost of 48,449 additional quartet
OCN checks (48,448.5 before rounding). The final budget remained 64.

The follow-up measured the cost of indexing shared branch pairs and evaluated
candidate recovery after ancestral reconstruction on PGK codon simulations.
[Branch-pair index and site-PP results](BRANCH_PAIR_INDEX.md)

## Reproduction

The run used macOS arm64, Python 3.12.15, NumPy 2.5.5 and the CSUBST working
tree based on `1aabfbd`. The script writes per-seed counts to `small.tsv` and
`scale.tsv`. The independent null, endpoint uncertainty and absence of internal
branches limit extrapolation to real trees. The matched budgets do not establish
relative wall time or memory use in `csubst search`.

```bash
python reports/branch_search_review/site_intersection_pilot.py \
  --outdir reports/generated/site-intersection-pilot --seeds 12
```
