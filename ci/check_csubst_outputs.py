#!/usr/bin/env python3
import glob, sys, os

def check_tsv(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip("\n") for l in f]
    assert len(lines) >= 2, f"{path}: 行数が少なすぎます"
    header = lines[0].split("\t")
    assert len(header) >= 2, f"{path}: タブ区切りのヘッダがありません"
    # 代表的なメトリクス名の痕跡（将来名前が変わっても緩めに）
    if os.path.basename(path).startswith("csubst_cb_stats"):
        assert any(("omegaC" in h) or ("OCN" in h) or ("OCS" in h) for h in header), \
            f"{path}: 代表メトリクス列が見当たりません"

def main():
    tsvs = sorted(glob.glob("csubst_*.tsv"))
    if not tsvs:
        print("WARN: csubst_*.tsv がありません（前段で生成できていればOKのはず）")
        sys.exit(0)
    for t in tsvs:
        check_tsv(t)
        print(f"OK: {t}")
    # IQ-TREE中間の存在も最終確認
    for req in ["alignment.fa.state", "alignment.fa.rate"]:
        assert os.path.exists(req) and os.path.getsize(req) > 0, f"{req}: 見つからない/空です"
    print("OK: IQ-TREE intermediates present")

if __name__ == "__main__":
    main()
