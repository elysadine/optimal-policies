"""Estimate regional absorbing transition matrices from aggregate (marginal)
pre/post level distributions, by constrained weighted least squares
(Lee, Judge & Zellner 1970; Kalbfleisch & Lawless 1984 macro-data setting).

Model, for each cell c = (CISCO, grade) in a region:
    q_c ≈ p_c P,   P row-stochastic, last state absorbing,
where p_c, q_c are pre and post proportions over the m ordered levels.
Weights: pre-test count n_c.  Uncertainty: cluster bootstrap over CISCOs.
"""
import json, numpy as np, pandas as pd
from itertools import permutations
from scipy.optimize import minimize

rng = np.random.default_rng(20261014)
B = 500

def chain_defs():
    m = pd.read_csv("maths_cells.csv")
    r = pd.read_csv("reading_cells.csv")
    # Operations ladder (cumulative Guttman scale): none < +add < +sub < +mul < all four
    for c in ["ADD", "SUB", "MUL", "DIV"]:
        m[c] = m[c].clip(upper=m.tested)
    sub = np.minimum(m.SUB, m.ADD); mul = np.minimum(m.MUL, sub); div = np.minimum(m.DIV, mul)
    m["O0"] = m.tested - m.ADD; m["O1"] = m.ADD - sub; m["O2"] = sub - mul
    m["O3"] = mul - div;        m["O4"] = div
    return {
        "reading":     (r, ["D", "L", "W", "P", "S"], ["Beginner", "Letter", "Word", "Paragraph"], "Story"),
        "maths_ops":   (m, ["O0", "O1", "O2", "O3", "O4"], ["None", "Add", "Add+Sub", "Add+Sub+Mul"], "All four"),
        "maths_num":   (m, ["D", "C1", "C2", "C3"], ["Beginner", "1-digit", "2-digit"], "3-digit"),
    }

def cells(df, lv):
    pre = df[df.wave == "pre"].set_index(["cisco", "grade"])[lv]
    post = df[df.wave == "post"].set_index(["cisco", "grade"])[lv]
    idx = pre.index.intersection(post.index)
    pre, post = pre.loc[idx], post.loc[idx]
    n = pre.sum(axis=1).to_numpy()
    ok = (n > 0) & (post.sum(axis=1).to_numpy() > 0)
    P = pre.to_numpy()[ok] / n[ok, None]
    Qp = post.to_numpy()[ok] / post.sum(axis=1).to_numpy()[ok, None]
    return P, Qp, n[ok], np.array([i[0] for i in idx])[ok]

def fit(p, q, w, m, lam=0.0, X0=None):
    k = m - 1                      # transient states
    W = w / w.sum()
    # residual r_c = q_c - p_c[:, :k] X - p_c[:, k] e_m ; X is k x m
    target = q.copy(); target[:, -1] -= p[:, -1]
    A = p[:, :k]
    def f(x):
        X = x.reshape(k, m); R = target - A @ X
        val = (W[:, None] * R ** 2).sum(); g = -2 * A.T @ (W[:, None] * R)
        if lam > 0:
            D = X - X0; val += lam * (D ** 2).sum(); g = g + 2 * lam * D
        return val, g.ravel()
    cons = [{"type": "eq", "fun": (lambda x, i=i: x.reshape(k, m)[i].sum() - 1),
             "jac": (lambda x, i=i: np.eye(k)[i].repeat(m))} for i in range(k)]
    best = None
    for s in range(3):
        x0 = rng.dirichlet(np.ones(m), size=k).ravel() if s else np.full(k * m, 1 / m)
        res = minimize(f, x0, jac=True, method="SLSQP", bounds=[(0, 1)] * (k * m),
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res
    X = np.clip(best.x.reshape(k, m), 0, None); X /= X.sum(1, keepdims=True)
    return X

def summarize(X):
    Q = X[:, :-1]
    rho = max(abs(np.linalg.eigvals(Q)))
    if rho >= 1 - 1e-9:
        return Q, rho, np.full_like(Q, np.inf), np.full(len(Q), np.inf)
    N = np.linalg.inv(np.eye(len(Q)) - Q)
    return Q, rho, N, N.sum(1)

LAMS = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 10.0]

def loco_cv(df, lv, m, Xnat):
    """Leave-one-CISCO-out prediction error within each region, per lambda."""
    err = {l: 0.0 for l in LAMS}
    for reg in df.region.unique():
        p, q, w, cis = cells(df[df.region == reg], lv)
        for c in np.unique(cis):
            tr, te = cis != c, cis == c
            if tr.sum() == 0:
                continue
            for l in LAMS:
                X = fit(p[tr], q[tr], w[tr], m, l, Xnat)
                pred = p[te, :-1] @ X; pred[:, -1] += p[te, -1]
                err[l] += (w[te] * ((q[te] - pred) ** 2).sum(1)).sum()
    tot = sum(w.sum() for w in [cells(df, lv)[2]])
    return {l: e / tot for l, e in err.items()}

out = {}
for name, (df, lv, names, absn) in chain_defs().items():
    m = len(lv)
    res = {}
    regions = sorted(df.region.unique())
    pN, qN, wN, cN = cells(df, lv)
    Xnat = fit(pN, qN, wN, m)
    cv = loco_cv(df, lv, m, Xnat)
    lam = min(cv, key=cv.get)
    print(name, "LOCO-CV:", {k: round(v, 6) for k, v in cv.items()}, "-> lambda =", lam)
    for reg in regions + ["NATIONAL"]:
        nat = reg == "NATIONAL"
        d = df if nat else df[df.region == reg]
        p, q, w, cis = cells(d, lv)
        L = 0.0 if nat else lam
        X = fit(p, q, w, m, L, Xnat)
        Q, rho, N, t = summarize(X)
        pred = p[:, :-1] @ X; pred[:, -1] += p[:, -1]
        rmse = np.sqrt(np.average(((q - pred) ** 2).sum(1), weights=w))
        uc = np.unique(cis); rhos, ts = [], []
        for b in range(B if not nat else 200):
            pick = rng.choice(uc, len(uc), replace=True)
            ix = np.concatenate([np.where(cis == c)[0] for c in pick])
            Xb = fit(p[ix], q[ix], w[ix], m, L, Xnat)
            _, rb, _, tb = summarize(Xb)
            rhos.append(rb); ts.append(tb[0])
        res[reg] = dict(X=X.round(4).tolist(), rho=rho, N=N.round(3).tolist(), t=t.round(3).tolist(),
                        rho_ci=np.percentile(rhos, [2.5, 97.5]).tolist(),
                        rho_boot_sd=float(np.std(rhos)), share_rho1=float(np.mean(np.array(rhos) >= 1 - 1e-9)),
                        t0_ci=np.percentile(ts, [2.5, 97.5]).tolist(),
                        n_cisco=len(uc), n_cells=len(w), n_pre=float(w.sum()), rmse=rmse,
                        pre_mix=np.average(p, axis=0, weights=w).round(3).tolist(),
                        post_mix=np.average(q, axis=0, weights=w).round(3).tolist())
        print(f"{name:10s} {reg:16s} rho={rho:.3f} CI=[{res[reg]['rho_ci'][0]:.3f},{res[reg]['rho_ci'][1]:.3f}] "
              f"E[T]0={t[0]:.2f} CI=[{res[reg]['t0_ci'][0]:.2f},{res[reg]['t0_ci'][1]:.2f}] rmse={rmse:.4f} ciscos={len(uc)}")
    tot = sat_N = sat_t = 0
    for a, b in permutations(regions, 2):
        if res[a]["rho"] <= res[b]["rho"]:
            tot += 1
            sat_N += np.all(np.array(res[a]["N"]) <= np.array(res[b]["N"]) + 1e-9)
            sat_t += np.all(np.array(res[a]["t"]) <= np.array(res[b]["t"]) + 1e-9)
    from scipy.stats import spearmanr
    rr = np.array([res[g]["rho"] for g in regions]); tt = np.array([res[g]["t"][0] for g in regions])
    sp = spearmanr(rr, tt)
    res["_mono"] = dict(pairs=tot, sat_N=int(sat_N), sat_t=int(sat_t),
                        spearman=float(sp.statistic), spearman_p=float(sp.pvalue))
    print(f"  monotonicity: N-order {sat_N}/{tot}, t-order {sat_t}/{tot}; Spearman(rho,t0)={sp.statistic:.2f} p={sp.pvalue:.3f}")
    out[name] = dict(levels=lv, names=names, absorbing=absn, lam=lam, cv=cv, res=res)

json.dump(out, open("results.json", "w"), indent=1, default=float)
