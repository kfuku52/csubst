"""Render the matched selected-family BH comparison from measured results."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / 'summary.json').read_text())
assert len(summary['conditions']) == 12
index = {(r['scenario'], r['filtered'], r['signal_sites']): r for r in summary['conditions']}
names = {'sparse': '疎な置換', 'unequal': '不均一な枝長', 'uncertain': '祖先不確実・速度混合'}
lines = [
    '# 選別後のtrait/match別BHに統一した比較', '',
    '両方式に同じ候補選別を適用し、選別後の同じ行集合をtrait × matchで分け、',
    '各群内の候補数を分母としてBH補正した。未定義Pは1として群の分母を共通に保った。',
    'これは性能比較の条件変更であり、CSUBST本体の新Pの補正仕様は変更していない。', '',
    '既知の4コドンCTMC、8 tips、12サイト、2 traits × 2 match classes。',
    '各条件1,000データセット、BH q≤0.05。前回と同じseed=51052026と入力生成条件を使用した。',
    '対立側は3/12サイトのforeground tips a/eをAACに置換した感度対照。', '',
    '**数値はすべて%、旧→新の順。**', '',
    '| 条件 | 不変サイト除去 | 帰無での誤検出率 | 検出感度 | 部分対立での経験的FDR |',
    '| --- | --- | ---: | ---: | ---: |',
]
ci_lines = ['| 条件 | 不変サイト除去 | 帰無誤検出：旧 | 帰無誤検出：新 | 感度：旧 | 感度：新 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
for scenario, name in names.items():
    for filtered in (False, True):
        null = index[scenario, filtered, 0]['metrics']
        signal = index[scenario, filtered, 3]['metrics']
        assert null['old_selected_false_any']['mean'] == null['diagnostic_bh_reject']['mean']
        assert signal['old_selected_true_any']['mean'] == signal['diagnostic_true_any']['mean']
        pairs = []
        for metrics, suffix, decimals in [(null, 'false_any', 1), (signal, 'true_any', 1), (signal, 'fdp', 2)]:
            pairs.append(' → '.join(f"{100 * metrics[method + '_selected_' + suffix]['mean']:.{decimals}f}" for method in ('old', 'new')))
        label = 'あり' if filtered else 'なし'
        lines.append(f"| {name} | {label} | " + ' | '.join(pairs) + ' |')
        cis = []
        for metrics, suffix in [(null, 'false_any'), (signal, 'true_any')]:
            for method in ('old', 'new'):
                lo, hi = metrics[method + '_selected_' + suffix]['ci95']
                cis.append(f'{100*lo:.2f}–{100*hi:.2f}')
        ci_lines.append(f'| {name} | {label} | ' + ' | '.join(cis) + ' |')
lines += ['',
    '- 帰無での誤検出率：データセット内で少なくとも1件の棄却があった割合。',
    '- 検出感度：注入サイトを少なくとも1つ検出した割合。',
    '- 経験的FDR：部分対立データでの `誤検出数 / max(1, 全検出数)` の1,000回平均。未注入サイトの棄却を誤検出とした。', '',
    'trait/match別BHは全体の5%制御を意味せず、選別自体の影響も補正しない。',
    'また、ここでの比較は既知モデル下での経験的性能であり、推定モデルの不確実性は含めない。', '',
    '## 比率の95%正確信頼区間（%）', '', *ci_lines, '',
    '同じseed・同じ候補に両方式を適用した対応比較。フィルター有無を独立試行としてプールしていない。', '',
    '## 再現', '', '```bash',
    'CSUBST_DISABLE_EXTENSIONS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
    '  python tools/benchmark_scan_analytic.py --outdir NEW_OUTPUT \\',
    '  --replicates 1000 --skip-runtime', '```', '',
    '[全測定値と実行環境](summary.json)。従来法の棄却率・感度が既存計算と一致することも照合した。',
    '実行時間の再比較は行っていない。新旧で補正群を同じにした統計的性能の比較である。', '',
]
(root / 'COMPARISON.md').write_text('\n'.join(lines))
repo = root.parents[2]
files = ['tools/validate_scan_calibration.py', 'tools/benchmark_scan_analytic.py', 'csubst/scan_analytic.py']
(root / 'source_hashes.json').write_text(json.dumps({p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in files}, indent=2) + '\n')
print('\n'.join(lines[13:22]))
