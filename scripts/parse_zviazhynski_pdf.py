"""Parse Zviazhynski supplementary Table 1 from PDF into canonical CSV."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Table 1 from supplementary PDF (Zviazhynski et al., Data-Centric Engineering)
# Columns: mix_ref, cement, GGBS(slag), fine_agg, coarse_agg, water, w/b, ...
RAW_ROWS = """
1 328 0 853 930 254 0.78 0.48 5.44 50 2952.25 - - 2291.8 39.41 -
2 328 0 842 930 204 0.62 0.48 5.40 90 2346.63 - - 2304.3 43.25 -
3 416 0 1044 640 233 0.56 0.62 4.05 40 3375.949 - - 2299.3 51.43 -
4 416 0 1043 640 274 0.66 0.62 4.05 205 405.69 279.82 2.36 2303.2 45.33 -
5 416 0 1044 640 252 0.61 0.62 4.05 110 1443.111 832.19 4.95 2297.3 47.46 -
6 416 0 1044 640 263 0.63 0.62 4.05 190 599.699 410.98 2.18 2298.7 44.49 -
7 416 0 1044 640 271 0.65 0.62 4.04 225 283.549 249.95 1.36 2259.5 38.55 42.23
8 334 84 1044 640 271 0.65 0.62 4.03 230 262.717 202.15 2.86 2272.8 39.70 43.04
9 416 0 1044 640 256 0.62 0.62 4.04 240 127.604 117.65 2.28 2240.3 37.33 -
10 415 0 930 700 247 0.59 0.57 3.92 165 898.583 546.06 2.23 2290.7 44.13 -
11 416 0 929 700 215 0.52 0.57 3.92 100 1823.979 986.12 2.71 2307.2 51.37 -
12 391 0 883 883 196 0.50 0.50 4.51 80 2542.871 941.64 9.57 2309.3 48.23 50.40
13 382 0 860 860 229 0.60 0.50 4.50 205 594.706 372.11 4.57 2289.8 41.91 46.10
14 375 0 844 844 225 0.60 0.50 4.50 170 1083.779 572.18 4.75 2290.3 43.52 46.34
15 113 263 846 846 225 0.60 0.50 4.51 70 3240.215 - 12.84 2266.2 27.34 29.50
16 113 263 844 844 244 0.65 0.50 4.50 125 1388.167 655.28 6.2 2260.7 24.49 25.72
17 375 0 844 844 244 0.65 0.50 4.50 240 155.784 106.35 3.73 2301.7 36.79 41.95
18 375 0 846 846 233 0.62 0.50 4.51 230 314.166 175.52 6.67 2315.5 39.61 43.55
19 375 0 844 844 233 0.62 0.50 4.50 150 1221.598 625.21 6.3 2287.5 40.15 44.07
20 113 263 844 844 233 0.62 0.50 4.50 90 2067.95 897.46 8.69 2277.5 28.15 30.45
21 188 188 844 844 233 0.62 0.50 4.50 100 1916.251 841.67 6.29 2281.3 33.99 39.39
22 300 75 844 844 233 0.62 0.50 4.50 140 1303.011 609.07 7.64 2283.8 38.55 42.38
23 244 131 844 844 233 0.62 0.50 4.50 165 926.966 549.36 3.93 2291.8 36.97 41.47
24 349 0 873 873 228 0.65 0.50 5.00 175 1130.596 550.94 5.87 2293.8 37.07 40.37
25 130 130 907 907 236 0.91 0.50 7.00 190 647.902 286.2 5.78 2270.2 18.34 20.62
26 259 0 907 907 231 0.89 0.50 7.00 175 1101.065 354.15 9.89 2275.2 23.16 25.03
A 169 169 847 847 227 0.67 0.50 5.00 180 713.139 365.88 6.52 2270.0 28.04 32.79
B 256 256 768 768 249 0.49 0.50 3.00 150 1336.454 939.61 6.14 2275.3 39.32 43.69
27 113 263 846 846 233 0.62 0.50 4.51 130 1732.696 796.2 8.48 2265.5 24.56 28.53
28 375 0 846 846 233 0.62 0.50 4.51 150 1254.579 552.84 7.81 2298.8 42.35 48.03
29 188 188 846 846 233 0.62 0.50 4.51 65 3235.171 - - 2281.3 35.39 39.36
30 300 75 846 846 233 0.62 0.50 4.51 155 1322.475 607.43 6.38 2288.5 38.97 46.36
31 188 188 846 846 233 0.62 0.50 4.51 155 1451.268 574.35 8.18 2275.0 31.09 33.16
32 376 0 853 853 233 0.62 0.50 4.54 155 1292.491 646.77 5.91 2288.0 40.51 43.36
33 112 262 853 853 233 0.62 0.50 4.56 80 2912.065 994.67 9.32 2264.2 26.56 28.49
"""


def _parse_row(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^(\S+)\s+(.*)$", line)
    if not m:
        return None
    mix_ref, rest = m.group(1), m.group(2)
    tokens = [None if t == "-" else t for t in rest.split()]
    if len(tokens) < 15:
        return None

    def f(x):
        if x is None:
            return None
        try:
            return float(x)
        except ValueError:
            return None

    return {
        "mix_ref": mix_ref,
        "cement": f(tokens[0]),
        "slag": f(tokens[1]),
        "fine_agg": f(tokens[2]),
        "coarse_agg": f(tokens[3]),
        "water": f(tokens[4]),
        "w_b_ratio": f(tokens[5]),
        "sand_agg_ratio": f(tokens[6]),
        "agg_binder_ratio": f(tokens[7]),
        "slump_mm": f(tokens[8]),
        "static_yield_stress_pa": f(tokens[9]),
        "dynamic_yield_stress_pa": f(tokens[10]),
        "viscosity_pa_s": f(tokens[11]),
        "density_kg_m3": f(tokens[12]),
        "strength_mpa": f(tokens[13]),
        "strength_56d_mpa": f(tokens[14]),
    }


def build_dataframe() -> pd.DataFrame:
    rows = [_parse_row(line) for line in RAW_ROWS.strip().splitlines()]
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    df["fly_ash"] = 0.0
    df["superplasticizer"] = 0.0
    df["age"] = 28.0
    df["domain_id"] = "D3_zviazhynski"
    return df.dropna(subset=["strength_mpa", "cement", "water"])


def main():
    out = Path("data/raw/zviazhynski_2025/zviazhynski_supplementary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataframe()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} mixes -> {out}")
    print(df[["mix_ref", "cement", "slag", "water", "strength_mpa", "w_b_ratio"]].head(3).to_string())


if __name__ == "__main__":
    main()
