"""ディレクトリ内の距離行列（CSV）を総当たりでMantel検定し、Holm補正して1つのCSVに出力する。

Pythonから:
    from mantel_dir import mantel_all_pairs
    res = mantel_all_pairs("path/to/dir")

コマンドラインから:
    python mantel_dir.py path/to/dir [-o 出力.csv]

入力CSVの形式:
    1行目と1列目がラベルの正方行列（対称な距離行列）。対角要素は使わない。
    全CSVに共通するラベルだけを使って検定する。
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def mantel_all_pairs(
    directory: str,
    output: str | None = "mantel_all_pairs.csv",
    *,
    pattern: str = "*.csv",
    label_func=None,
    method: str = "spearman",
    alternative: str = "greater",
    permutations: int = 9999,
    seed: int | None = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """directory 内の距離行列を総当たりでMantel検定し、Holm補正した結果を返す。

    Parameters
    ----------
    directory : 距離行列のCSVが入っているディレクトリ名
    output : 結果を書き出すCSVのパス。None なら書き出さない
    pattern : 読み込むファイル名のパターン
    label_func : ラベルを変換する関数（例: lambda s: s.split("-")[2]）。
        変換後に同じラベルが複数あれば、そのブロックの平均距離にまとめる
    method : "spearman" | "pearson"
    alternative : "greater"（正の相関）| "less" | "two-sided"
    permutations : 並べ替え回数
    seed : 乱数シード
    alpha : 有意水準（significant 列の判定に使う）

    Returns
    -------
    列 X, Y, n, r, p, p_holm, significant を持つ DataFrame（p の小さい順）
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method は 'spearman' か 'pearson'")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative は 'greater' / 'less' / 'two-sided'")
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"ディレクトリが見つかりません: {directory}")

    # 1. 読み込み
    mats = {}
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        if output is not None and os.path.abspath(path) == os.path.abspath(output):
            continue                               # 前回の出力ファイルは入力にしない
        name = os.path.splitext(os.path.basename(path))[0]
        mats[name] = _read_matrix(path, label_func)
    if len(mats) < 2:
        raise ValueError(f"距離行列が2つ以上必要です（{directory} に {len(mats)} 個）")

    # 2. 全行列に共通するラベルに絞り、上三角を標準化する（内積がそのまま相関になる）
    common = sorted(set.intersection(*(set(D.index) for D in mats.values())))
    n = len(common)
    if n < 3:
        raise ValueError(f"全行列に共通するラベルが3つ以上必要です（{n} 個）")
    iu = np.triu_indices(n, k=1)

    S = {}
    for name, D in mats.items():
        v = D.loc[common, common].to_numpy(dtype=float)[iu]
        if not np.isfinite(v).all():
            raise ValueError(f"{name}: 非対角要素に NaN または inf があります")
        if method == "spearman":
            v = rankdata(v)
        v = v - v.mean()
        norm = np.sqrt(v @ v)
        if norm == 0:
            raise ValueError(f"{name}: 非対角要素がすべて同じ値です")
        M = np.zeros((n, n))
        M[iu] = v / norm
        S[name] = M + M.T

    # 3. 総当たりでMantel検定
    rng = np.random.default_rng(seed)
    eps = 1e-12                                    # 浮動小数点誤差で同値を数え落とさないため
    rows = []
    for a, b in itertools.combinations(S, 2):
        y = S[b][iu]
        r = float(S[a][iu] @ y)
        null = np.empty(permutations)
        for k in range(permutations):
            p = rng.permutation(n)                 # ラベルを入れ替える（行と列を同時に）
            null[k] = S[a][np.ix_(p, p)][iu] @ y
        if alternative == "greater":
            count = np.sum(null >= r - eps)
        elif alternative == "less":
            count = np.sum(null <= r + eps)
        else:
            count = np.sum(np.abs(null) >= abs(r) - eps)
        rows.append({"X": a, "Y": b, "n": n, "r": r, "p": (count + 1) / (permutations + 1)})

    # 4. Holm補正（検定の総数 m に応じて）
    res = pd.DataFrame(rows).sort_values("p", kind="stable").reset_index(drop=True)
    m = len(res)
    res["p_holm"] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * res["p"].to_numpy()))
    res["significant"] = res["p_holm"] < alpha

    if output is not None:
        res.to_csv(output, index=False)
    return res


def _read_matrix(path: str, label_func=None) -> pd.DataFrame:
    """CSVを読み、ラベル付きの対称な距離行列として返す。"""
    D = pd.read_csv(path, index_col=0)
    D.index = D.index.map(str)
    D.columns = D.columns.map(str)
    if D.shape[0] != D.shape[1] or list(D.index) != list(D.columns):
        raise ValueError(f"{path}: 行と列のラベルが一致する正方行列ではありません")

    A = D.to_numpy(dtype=float)
    off = ~np.eye(len(A), dtype=bool)
    if not np.allclose(A[off], A.T[off], equal_nan=True):
        raise ValueError(f"{path}: 対称ではありません")

    if label_func is not None:
        labels = [str(label_func(s)) for s in D.index]
        D.index, D.columns = labels, labels
    if D.index.has_duplicates:
        # 同じラベルの行・列をまとめる（ブロックの平均距離）。対角は検定で使わない
        D = D.groupby(level=0).mean().T.groupby(level=0).mean()
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ディレクトリ内の距離行列を総当たりでMantel検定し、Holm補正する")
    ap.add_argument("directory", help="距離行列のCSVが入っているディレクトリ")
    ap.add_argument("-o", "--output", default="mantel_all_pairs.csv", help="出力CSV")
    ap.add_argument("--pattern", default="*.csv", help="読み込むファイル名のパターン")
    ap.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    ap.add_argument("--alternative", default="greater", choices=["greater", "less", "two-sided"])
    ap.add_argument("--permutations", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = mantel_all_pairs(
        args.directory, args.output, pattern=args.pattern, method=args.method,
        alternative=args.alternative, permutations=args.permutations, seed=args.seed,
    )
    print(f"検定: {len(res)}個 / 有意: {int(res['significant'].sum())}個 / ラベル数: {res['n'].iloc[0]} → {args.output}")
