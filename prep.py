"""Tidy the CISCO-level TaRL pre/post files into long format.

Output: maths_cells.csv, reading_cells.csv with one row per
(region, cisco, grade, wave) and counts per exclusive level.
"""
import sys, pandas as pd, numpy as np

UP = sys.argv[1]
GR = ["T2", "T3", "T4", "T5"]


def block(d, start):
    """4 grade columns starting at `start`."""
    b = d.iloc[:, start:start + 4].astype(str).apply(
        lambda c: c.str.replace(r"[\s  ]", "", regex=True).replace({"nan": np.nan, "": np.nan, "-": np.nan}))
    return b.astype(float).to_numpy()


def cisco_rows(d):
    m = d[1].notna() & d[2].notna() & ~d[0].astype(str).str.startswith(("TOTAL", "N°", "RECAP"))
    return d[m]


def maths(path):
    out = []
    for wave, sh in [("pre", "PRETEST"), ("post", "POSTTEST")]:
        d = pd.read_excel(path, sheet_name=sh, header=None)
        r = cisco_rows(d)
        lv = {"D": 17, "C1": 22, "C2": 27, "C3": 32, "ADD": 47, "SUB": 52, "MUL": 57, "DIV": 62}
        tested = block(r, 12)
        arrs = {k: block(r, v) for k, v in lv.items()}
        for i in range(len(r)):
            for g in range(4):
                row = dict(region=str(r.iloc[i, 1]).strip(), cisco=str(r.iloc[i, 2]).strip(),
                           grade=GR[g], wave=wave, tested=tested[i, g])
                row.update({k: a[i, g] for k, a in arrs.items()})
                out.append(row)
    return pd.DataFrame(out)


def reading(path):
    out = []
    x = pd.read_excel(path, sheet_name=None, header=None)
    sheets = list(x)
    for wave, sh in [("pre", sheets[0]), ("post", sheets[1])]:
        d = x[sh]
        r = cisco_rows(d)
        lv = {"D": 13, "L": 19, "W": 25, "P": 31, "S": 37}
        tested = block(r, 8)
        arrs = {k: block(r, v) for k, v in lv.items()}
        for i in range(len(r)):
            for g in range(4):
                row = dict(region=str(r.iloc[i, 1]).strip(), cisco=str(r.iloc[i, 2]).strip(),
                           grade=GR[g], wave=wave, tested=tested[i, g])
                row.update({k: a[i, g] for k, a in arrs.items()})
                out.append(row)
    return pd.DataFrame(out)


m = maths(f"{UP}/3ec93a05-datasets2maths.xlsx")
r = reading(f"{UP}/db517c07-datasets2reading.xlsx")
m.to_csv("maths_cells.csv", index=False)
r.to_csv("reading_cells.csv", index=False)

for name, df, lv in [("MATHS", m, ["D", "C1", "C2", "C3"]), ("READING", r, ["D", "L", "W", "P", "S"])]:
    print(name, df.shape, "regions:", df.region.nunique(), "ciscos:", df.cisco.nunique())
    df["sumlv"] = df[lv].sum(1)
    bad = df[(df.sumlv - df.tested).abs() > 0.5]
    print("  rows where levels != tested:", len(bad))
    if len(bad):
        print(bad[["region", "cisco", "grade", "wave", "tested", "sumlv"]].head(15).to_string())
    print("  missing:", df[lv].isna().sum().sum())
    print(df.groupby(["region", "wave"]).tested.sum().unstack())
if True:
    nest = m[(m.ADD < m.SUB) | (m.SUB < m.MUL) | (m.MUL < m.DIV)]
    print("maths op-nesting violations:", len(nest), "of", len(m))
    print("ADD > tested:", (m.ADD > m.tested).sum())
