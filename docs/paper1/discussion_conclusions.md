# Discussion and Conclusions — SCM F3 paper (SPARC OOS)

> **Status:** submission-ready draft. Three referee-hardening patches applied (2026-03-29).
>
> Patch 1 — effect-size sentence added after Wilcoxon p-value.
> Patch 2 — "rules out any dependence" → "indicates that the result is not driven by".
> Patch 3 — phenomenological-framing sentence added before final paragraph of Conclusions.

---

## 5. Discussion

The out-of-sample validation presented in Section 4 demonstrates that the SCM F₃ term
provides a systematic and statistically robust improvement in predictive performance relative
to the baseline model.
The SCM model improves out-of-sample performance in all test cases considered (53/53),
indicating a systematic advantage over the baseline within the current validation setup.
The consistency of this result across five independent seeds indicates that the result is
not driven by a particular train-test partition,
and the one-sided Wilcoxon test yields p = 1.19 × 10⁻¹⁰ (Table 4).
The median improvement corresponds to
ΔRMSE_out = −7.27 (in β proxy units),
indicating a substantial effect size in addition to statistical significance.

A possible physical interpretation of this improvement is that the SCM friction term F₃
encodes a local dynamical memory of the gravitational environment not captured by the
baseline relation.
Galaxies with elevated F₃ values — indicative of steeper radial gradients in the
baryon-to-total acceleration ratio — may require a correction that the baseline MOND-like
relation does not provide.
This is broadly consistent with the recurrence analysis of Section 3, where the F₃ signal
was found to persist across radial windows in a subset of the SPARC sample.

We note two caveats that should be kept in mind when interpreting these results.
First, the prediction table used here is based on a proxy construction (β units derived
from the current SPARC photometry pipeline); the reported ΔRMSE_out values are therefore
expressed in proxy β units, and the four summary statistics in Table 4 will be updated once
the final rotational-velocity prediction CSV is available, using the identical validation
pipeline.
Second, the 70/30 galaxy-level split does not account for within-galaxy radial correlation;
a hierarchical cross-validation scheme would provide a stricter bound on generalisation
uncertainty, and is deferred to future work.

Importantly, the present analysis does not assume a specific microphysical origin for the
F₃ term, but instead establishes its empirical relevance as an additional observable that
improves predictive performance within the current SPARC validation framework.

---

## 6. Conclusions

We have presented an out-of-sample validation of the SCM F₃ friction term using the SPARC
rotation-curve sample. The main results are as follows.

1. **In-sample fit.** The SCM model with the F₃ correction provides a consistently
   improved description of the SPARC rotation-curve sample relative to the baseline model
   (Section 4).

2. **Out-of-sample generalisation.** Across repeated 70/30 train-test splits (seeds 42–46),
   the SCM model improves out-of-sample RMSE in all 53/53 test galaxies, with a median
   improvement of ΔRMSE_out = −7.27 and Wilcoxon p = 1.19 × 10⁻¹⁰ (Table 4). This
   demonstrates that the F₃ signal generalises beyond the training sample within the current
   validation setup.

3. **Empirical relevance.** The recurrence analysis (Section 3) confirms that F₃ is
   spatially coherent within individual galaxies and correlates with the local gravitational
   environment encoded in the baryon-deficit slope, supporting its use as a predictive
   observable independently of any assumed microphysical mechanism.

These results are therefore best interpreted as a phenomenological extension of the standard
framework, rather than a definitive statement on the underlying microphysics.

These results motivate a fuller observational programme in which the SCM F₃ term is tested
on the complete SPARC dataset with final photometric calibrations, and extended to
dwarf-irregular and low-surface-brightness galaxies where the friction term is expected to
be largest.
The validation pipeline presented here is fully reproducible and requires only the
substitution of the final prediction CSV to regenerate all reported statistics.

---

## LaTeX source

```latex
\section{Discussion}

The out-of-sample validation presented in Section~\ref{sec:results}
demonstrates that the SCM~$F_3$ term provides a systematic and statistically
robust improvement in predictive performance relative to the baseline model.
The SCM model improves out-of-sample performance in all test cases considered
($53/53$), indicating a systematic advantage over the baseline within the current
validation setup.
The consistency of this result across five independent seeds indicates that the
result is not driven by a particular train-test partition,
and the one-sided Wilcoxon test yields $p = 1.19\times10^{-10}$
(Table~\ref{tab:oos_validation}).
The median improvement corresponds to
$\Delta\mathrm{RMSE}_{\rm out} = -7.27$ (in $\beta$ proxy units),
indicating a substantial effect size in addition to statistical significance.

A possible physical interpretation of this improvement is that the SCM friction
term $F_3$ encodes a local dynamical memory of the gravitational environment not
captured by the baseline relation.
Galaxies with elevated $F_3$ values---indicative of steeper radial gradients in
the baryon-to-total acceleration ratio---may require a correction that the
baseline MOND-like relation does not provide.
This is broadly consistent with the recurrence analysis of
Section~\ref{sec:h3}, where the $F_3$ signal was found to persist across radial
windows in a subset of the SPARC sample.

We note two caveats that should be kept in mind when interpreting these results.
First, the prediction table used here is based on a proxy construction ($\beta$
units derived from the current SPARC photometry pipeline); the reported
$\Delta\mathrm{RMSE}_{\rm out}$ values are therefore expressed in proxy $\beta$
units, and the four summary statistics in Table~\ref{tab:oos_validation} will be
updated once the final rotational-velocity prediction CSV is available, using the
identical validation pipeline.
Second, the 70/30 galaxy-level split does not account for within-galaxy radial
correlation; a hierarchical cross-validation scheme would provide a stricter
bound on generalisation uncertainty, and is deferred to future work.

Importantly, the present analysis does not assume a specific microphysical origin
for the $F_3$ term, but instead establishes its empirical relevance as an
additional observable that improves predictive performance within the current
SPARC validation framework.

\section{Conclusions}

We have presented an out-of-sample validation of the SCM $F_3$ friction term
using the SPARC rotation-curve sample. The main results are as follows.

\begin{enumerate}

\item \textit{In-sample fit.}
The SCM model with the $F_3$ correction provides a consistently improved
description of the SPARC rotation-curve sample relative to the baseline model
(Section~\ref{sec:results}).

\item \textit{Out-of-sample generalisation.}
Across repeated 70/30 train-test splits (seeds 42--46), the SCM model improves
out-of-sample RMSE in all $53/53$ test galaxies, with a median improvement of
$\Delta\mathrm{RMSE}_{\rm out} = -7.27$ and Wilcoxon $p = 1.19\times10^{-10}$
(Table~\ref{tab:oos_validation}).
This demonstrates that the $F_3$ signal generalises beyond the training sample
within the current validation setup.

\item \textit{Empirical relevance.}
The recurrence analysis (Section~\ref{sec:h3}) confirms that $F_3$ is spatially
coherent within individual galaxies and correlates with the local gravitational
environment encoded in the baryon-deficit slope, supporting its use as a
predictive observable independently of any assumed microphysical mechanism.

\end{enumerate}

These results are therefore best interpreted as a phenomenological extension of
the standard framework, rather than a definitive statement on the underlying
microphysics.

These results motivate a fuller observational programme in which the SCM $F_3$
term is tested on the complete SPARC dataset with final photometric calibrations,
and extended to dwarf-irregular and low-surface-brightness galaxies where the
friction term is expected to be largest.
The validation pipeline presented here is fully reproducible and requires only
the substitution of the final prediction CSV to regenerate all reported
statistics.
```
