from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_delta_f3_vs_environment import check_columns, make_plot


def test_plot_generation(tmp_path: Path):
    rng = np.random.default_rng(12)
    x = rng.uniform(0.0, 1.0, 80)
    y = 0.2 + 0.7 * x + rng.normal(0.0, 0.03, 80)
    df = pd.DataFrame({"logSigmaHI_out": x, "delta_f3": y})
    check_columns(df)

    out = tmp_path / "delta_f3_vs_environment.pdf"
    make_plot(df, out, n_bootstrap=40, seed=3)
    assert out.exists()
