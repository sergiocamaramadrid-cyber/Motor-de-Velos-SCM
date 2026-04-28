class SCM_RAA_LITE:
    def __init__(self):
        self.history = []

    def analyze(self, r):
        delta_aic = r.get('delta_aic_vs_linear', 0)
        cv = r.get('cv_gain', 0)
        iqr = r.get('xcrit_iqr_frac', 1.0)
        n = r.get('n', 0)

        if delta_aic < 2:
            status, conf = "failure", "low"
        elif delta_aic < 6:
            status, conf = "weak", "moderate"
        else:
            status, conf = "strong", "high"

        if cv < -0.05:
            status, conf = "failure", "low"
        elif cv < 0 and status == "strong":
            status, conf = "weak", "moderate"

        if iqr > 0.25 and status == "strong":
            status, conf = "weak", "low"

        if n < 50 and status == "strong":
            conf = "moderate"

        out = {
            "status": status,
            "confidence": conf,
            "veredicto": f"{status.upper()} ({conf})",
            "delta_aic": delta_aic,
            "cv_gain": cv,
            "iqr_frac": iqr,
            "n": n
        }

        self.history.append(out)
        return out
