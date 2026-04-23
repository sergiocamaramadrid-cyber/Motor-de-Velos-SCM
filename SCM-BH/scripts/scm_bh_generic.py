import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm


class SCMGeneric:
    def __init__(self, name="SCM-BH"):
        self.name = name
        self.data = None
        self.results = {}

    def load_data(self, df):
        self.data = df.copy()

    def clean(self, cols):
        df = self.data[cols].replace([np.inf, -np.inf], np.nan).dropna()
        return df

    def spearman(self, x, y):
        df = self.clean([x, y])
        rho, p = spearmanr(df[x], df[y])
        return {"N": len(df), "rho": float(rho), "p": float(p)}

    def split_test(self, x, y, mass, thr):
        df = self.clean([x, y, mass])

        high = df[df[mass] >= thr]
        low = df[df[mass] < thr]

        def f(d):
            if len(d) < 4:
                return {"N": len(d), "rho": None, "p": None}
            rho, p = spearmanr(d[x], d[y])
            return {"N": len(d), "rho": float(rho), "p": float(p)}

        return {
            "threshold": thr,
            "global": f(df),
            "high": f(high),
            "low": f(low),
        }

    def ols(self, y, xs):
        df = self.clean([y] + xs)
        X = sm.add_constant(df[xs])
        model = sm.OLS(df[y], X).fit(cov_type="HC3")

        return {
            "r2": float(model.rsquared),
            "params": model.params.to_dict(),
            "p": model.pvalues.to_dict(),
        }

    def add_E(self):
        self.data["E_BH"] = self.data["logL_bol"] - 2 * self.data["logM_BH"]

    def permutation(self, x, y, n=3000):
        df = self.clean([x, y])
        obs, _ = spearmanr(df[x], df[y])

        arr = df[y].values.copy()
        count = 0

        for _ in range(n):
            np.random.shuffle(arr)
            rho, _ = spearmanr(df[x], arr)
            if abs(rho) >= abs(obs):
                count += 1

        return {"rho": float(obs), "p_perm": count / n}

    def bootstrap(self, x, y, n=1000):
        df = self.clean([x, y])
        vals = df[[x, y]].values
        res = []

        for _ in range(n):
            idx = np.random.randint(0, len(vals), len(vals))
            r, _ = spearmanr(vals[idx, 0], vals[idx, 1])
            res.append(r)

        return {
            "mean": float(np.mean(res)),
            "ci": [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))],
        }
