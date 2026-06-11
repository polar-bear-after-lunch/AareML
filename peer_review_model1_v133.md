# Peer Review — AareML v1.33 (Model 1: Scientific Validity & Claim Calibration)

**Reviewer focus:** Scientific validity, statistical correctness, comparison fairness, causal language, limitations honesty, numerical consistency, and novelty calibration.
**Document reviewed:** `report_text_v133.md` (AareML, CAS Advanced Machine Learning, University of Bern), report version 1.33, 11 Jun 2026.
**Context:** Evaluated against expectations for a CAS (Certificate of Advanced Studies) project report, not a top-tier journal submission.

---

## Strengths

1. **Generally well-calibrated headline claims.** The central single-site result is stated honestly: the tuned LSTM is reported as *comparable* to Ridge on RMSE (0.303 vs 0.303 mg/L) with *overlapping 95% bootstrap CIs* and an explicit statement of "no statistically significant difference in RMSE at this single gauge" (§5.2). This is a refreshing absence of the most common overclaiming failure mode, where authors declare a complex model superior on the basis of a third-decimal RMSE difference.

2. **Confidence intervals are reported throughout and the bootstrap is appropriate.** RMSE CIs use a temporal block bootstrap (block = 30 days, 500 replicates), which correctly preserves autocorrelation structure rather than naïvely resampling days (§4.4, Glossary). This is the right choice for serially correlated time series and is more sophisticated than typical CAS-level work.

3. **Causal language is carefully hedged in the right places.** The latitude–difficulty association is explicitly flagged as correlational and a possible proxy for elevation/area ("the causal mechanism is not established… latitude may serve as a proxy", §5.3). SHAP results are repeatedly qualified as "correlation, not causation" and as possibly reflecting autocorrelation rather than physical law (§5.4, §7). This is exactly the discipline a reviewer wants to see.

4. **Seed variance is reported.** The final model is a 3-seed ensemble with per-seed RMSE range (0.298–0.304, std ≈ 0.003) disclosed (Table 2/3 caption). Reporting run-to-run variance is good practice and shows the headline difference vs Ridge is within seed noise — consistent with the "comparable" framing.

5. **Leakage controls are described and tested.** Chronological splits, train-only normalisation, per-gauge scalers for multi-site, windows that never cross split boundaries, and dedicated unit tests (`TestTrainValTestSplit`) are documented (§3.5). The residual interpolation-before-split leakage risk is *self-disclosed* in §6.4 with a quantified bound (1.2% missingness at 2473) — commendable honesty.

6. **The cross-ecosystem narrative is scientifically sound and well-reasoned.** The river→lake zero-shot failure (NSE = −2.145) is correctly interpreted as active misfit ("worse than predicting the mean"), and the "transferable architecture, not transferable model" conclusion (§5.7, §6.3) is supported by the paired zero-shot-fail / retrain-succeed evidence. The mechanistic explanation (reaeration vs stratification, short vs long memory) is coherent and tied back to the SHAP memory finding.

7. **Limitations section is broad and honest.** Single focus gauge, untested tail-event performance, limited DO coverage (16/86 gauges), limited Optuna budget, small ensemble, and the LSTM extrapolation ceiling (with the Baste et al. 2025 analogue) are all stated plainly, and the cross-continental result is explicitly downgraded to "exploratory lower bound" (n=4).

---

## Issues

### High priority

**H1. Direct numerical contradiction in the headline Wilcoxon p-value.**
The same zero-shot-LSTM-vs-Ridge significance test is reported as **p = 0.037** in the Abstract (l.33), §5.3 (l.1247), and Limitations (l.2233), but as **p = 0.024** in the Conclusion (l.2332). These cannot both be correct for the same n=10 paired test. The version history (v1.31) records the move to p = 0.037, so the Conclusion's p = 0.024 is a stale value left over from an earlier run (v1.17/v1.18 used p = 0.024, n=11). For a report whose central multi-site claim *rests on this single test*, an internally inconsistent p-value is a credibility problem and must be reconciled.

**H2. The 0.451 mg/L zero-shot mean is described inconsistently as both an 11-gauge and a 12-gauge mean.**
- Abstract (l.31): "Zero-shot transfer to 12 Swiss gauges achieves mean DO RMSE = 0.451 mg/L."
- §5.3 (l.1213–1215): "mean DO RMSE of 0.451 mg/L across 11 gauges (excl. training gauge 2473)."
- §5.3 (l.1261): "The mean RMSE of 0.451 mg/L in Table 4 includes all 12 gauges."
- Conclusion (l.2324): "Zero-shot transfer to 12 Swiss gauges achieves a mean DO RMSE of 0.451 mg/L."
- Table 4 labels the 0.451 row "Mean (excl. 2473)".

These statements are mutually exclusive: a mean that *excludes* gauge 2473 (RMSE 0.303) cannot equal the mean that *includes* it. Including 2473 would pull the mean below 0.451; excluding it gives 0.451. The "includes all 12 gauges" sentence (l.1261) is therefore wrong and should be removed, and the Abstract/Conclusion should say "12 evaluation gauges (mean over the 11 non-training gauges = 0.451 mg/L)" to be exact. This same ambiguity also affects the per-gauge mean (0.390) and EA-LSTM mean (0.435), whose denominators (11 vs 12, and whether failed gauge 2018 is excluded) are not stated consistently.

**H3. Ablation-table baseline RMSE (0.290) is irreconcilable with the main-results RMSE (0.303) without an explicit caveat.**
Table 6 (§5.8) reports the LSTM baseline DO RMSE = 0.290 mg/L, but Table 3 reports the tuned LSTM at 0.303 mg/L and the default LSTM at 0.305 mg/L. The footnote concedes the ablations were run "at 30 epochs" and quotes a *different* set of numbers again (21d: 0.304 mg/L). So three different "baseline" RMSE values (0.290 / 0.304 / 0.303) appear for what a reader will assume is the same model. The ablation deltas (≤ 0.012 mg/L) are smaller than the seed std (0.003) only by a factor of ~4 and are mostly within plausible noise; presenting them in a table that doesn't match the headline model invites the reader to over-read tiny differences. Recommend: (i) state clearly that ablations use a reduced 30-epoch budget and are not directly comparable to Table 3, and (ii) add per-condition variance or mark deltas within seed noise as "n.s."

### Medium priority

**M1. Wilcoxon n=10 is under-powered and the "statistically significant" framing in the Abstract is stronger than the evidence warrants.**
With n=10 paired differences, Wilcoxon signed-rank has minimum attainable p ≈ 0.002 and very coarse p-value granularity; p = 0.037 sits just under the 0.05 threshold and would not survive any multiple-comparison correction across the several pairwise tests the report runs (zero-shot vs Ridge, per-gauge vs Ridge, latitude Spearman, etc.). The report *does* caveat n=10 and the gauge-exclusion rationale (§5.3, Glossary), which is good, but the Abstract's "suggests a statistically significant improvement" and the Conclusion's "confirms a statistically significant improvement" overstate a borderline, small-n, uncorrected result. Recommend reporting the effect size alongside p, softening "confirms" to "provides preliminary evidence", and noting the absence of multiple-comparison correction.

**M2. "Comparison fairness" of the LSTM-vs-Ridge significance test is ambiguous because two different Ridge variants exist.**
The report describes a *per-gauge* Ridge baseline (§4.2, fit on each gauge's train+val) and *also* a *zero-shot* Ridge transfer (mean RMSE 0.568, §5.3 l.1263). It is not explicitly stated which Ridge enters the n=10 Wilcoxon test. If the zero-shot LSTM (trained only on 2473) is compared against *per-gauge-trained* Ridge, the comparison is actually conservative (handicaps the LSTM) and the win is more impressive; if it is compared against *zero-shot* Ridge, that is the apples-to-apples comparison. Either is defensible, but the report must say which, because the interpretation of the significance claim depends entirely on it. As written, a reader cannot verify the comparison is fair.

**M3. Calling NeuralHydrology EA-LSTM an "independent benchmark that confirms the findings" overstates what it shows.**
The independent NeuralHydrology EA-LSTM achieves mean RMSE 0.512 mg/L — *worse* than AareML zero-shot (0.451) and per-gauge (0.390) (§5.3c, Abstract l.55). The report itself attributes the gap to "differences in training data, feature sets, and hyperparameter optimisation." A result that is materially worse does not "confirm the findings" in the strong sense the Abstract implies; it confirms only that an independent implementation reaches the *same qualitative regime* (sub-mg/L river DO RMSE). Recommend rewording to "an independent NeuralHydrology EA-LSTM reaches comparable order-of-magnitude performance (0.512 mg/L), supporting the feasibility of the approach" rather than "confirms the findings."

**M4. The VAR(7) baseline (0.299 mg/L) actually *beats* the LSTM (0.303 mg/L) at gauge 2473 — this undercuts the single-site value proposition and should be surfaced, not buried.**
A multivariate linear VAR(7) achieves lower single-site RMSE than the tuned LSTM (§5.3 l.1269: 0.299 vs 0.303; NSE 0.891, KGE 0.911). The report frames this as evidence that "multi-variable inputs account for the majority of the improvement," which is fair, but the implication — that no nonlinear model is needed for single-site point accuracy — is not stated as plainly in the Abstract/Conclusion as it should be. This is an underclaim-by-omission in the discussion of where the LSTM does and does not add value. The LSTM's genuine advantage (KGE/distributional fidelity and multi-site transfer) should be foregrounded precisely *because* it loses on single-site RMSE to two simpler baselines (Ridge and VAR).

**M5. Cross-dataset benchmark comparison (AareML lake-retrained 0.768 vs LakeBeD-US 1.40) is not like-for-like.**
The "1.82× better than the LakeBeD-US benchmark" claim (§5.7, Abstract, Conclusion) compares a model trained/tested on 21 *Swiss* lakes (Bärenbold et al. 2026) against a benchmark on 21 *US* lakes (McAfee et al. 2025) — different lakes, different DO regimes, different splits. The report does add a caution ("direct comparison should be interpreted cautiously… datasets differ in geographic scope"), which is good, but the headline multiplier ("1.82×") is repeated as if it were a controlled result. The same applies to the river "≈4.6× better than LakeBeD-US" framing. Recommend demoting the multipliers from headline status to "indicative, not a controlled benchmark," since there is no shared test set.

**M6. KGE "advantage" is asserted without an interval or significance test, unlike RMSE.**
The KGE 0.945 vs 0.908 gap is repeatedly promoted to "the stronger differentiator… indicating superior representation of DO dynamics" (Abstract, §5.2, §7). But KGE is reported as a bare point estimate with no bootstrap CI and no significance test, whereas RMSE gets both. Given that the whole methodological care of the paper rests on CIs for RMSE, elevating an *un-intervalled* KGE difference to the primary evidence of LSTM superiority is asymmetric and weakly supported. Recommend bootstrapping KGE on the same blocks and reporting its CI; if the KGE CIs overlap, soften "superior representation" to "higher KGE point estimate."

### Low priority

**L1. Dataset citation inconsistency in the Abstract.** The Abstract (l.21) attributes the training data to "CAMELS-CH-Chem, Höge et al. 2023", but every other instance (§2.3, §5.3, References) attributes CAMELS-CH-Chem to Nascimento et al. 2025; Höge et al. 2023 is the CAMELS-CH *base* hydrometeorological dataset (cited correctly in §5.3b for precipitation forcing). Fix the Abstract attribution to Nascimento et al. 2025.

**L2. Temperature NSE inconsistency.** Zero-shot temperature transfer NSE is reported as 0.727 in §5.3b (l.1303) but 0.730 in the Conclusion (l.2334); mean RMSE 2.59°C (§5.3b) vs 2.598°C (Abstract). Minor rounding/stale-value drift; reconcile.

**L3. Cascaded-model reference RMSE drift.** The Abstract/§6.3 compare cascaded setups against the "direct LSTM (0.303 mg/L)", but the version history (v1.31) notes the nb18 standard-LSTM reference was "corrected to 0.296 mg/L." The qualitative conclusion (cascade adds complexity without benefit) is unaffected, but the reference number should be made consistent with whatever the final run reports.

**L4. SAITS imputation is described as part of the mirrored LakeBeD-US pipeline (§2.2) but is explicitly *not* used by AareML** (linear-interp + train-mean fill, §3.5/§4.1, Glossary). This is correctly disclosed in the Glossary, but a reader of §2.2/§4.1 could infer SAITS was used. One clarifying sentence in §4.1 ("unlike LakeBeD-US, we do not use SAITS") would remove the ambiguity.

**L5. "Effective memory of 3–4 days" is inferred from 1-day-ahead SHAP only.** The SHAP attribution (§5.4) is computed for the 1-day-ahead forecast; the claim that the model "does not exploit the full input window" is then generalised to the 14-day task. The ablation (6-day lookback within 0.004 mg/L of 21-day) supports this, so the conclusion is probably right, but the generalisation from a horizon-1 attribution to the whole forecast should be stated as such rather than as established.

**L6. Spearman ρ = 0.64, p = 0.027 (n≈11) for the latitude–RMSE correlation** is itself a small-n, uncorrected test reported as the "strongest predictor." The correlational hedging is good (L of §5.3), but the same multiple-comparison caveat as M1 applies and could be noted.

---

## Suggested fixes (priority-ordered)

1. **Reconcile the Wilcoxon p-value to a single value (p = 0.037, n = 10) everywhere**, including the Conclusion (l.2332). [H1]
2. **Fix the 0.451 mg/L denominator description**: state it is the mean over the 11 non-training gauges, delete the contradictory "includes all 12 gauges" sentence (l.1261), and make the per-gauge (0.390) and EA-LSTM (0.435) denominators explicit (note gauge 2018 per-gauge failure). [H2]
3. **Add an explicit caveat to Table 6** that ablations use a 30-epoch budget and are not directly comparable to Table 3's fully-trained 0.303; flag deltas within seed noise. [H3]
4. **Soften the significance language** ("confirms" → "provides preliminary evidence"), report an effect size, and note no multiple-comparison correction was applied across the several pairwise/correlation tests. [M1, L6]
5. **State explicitly which Ridge variant (zero-shot or per-gauge) enters the n=10 Wilcoxon test** and confirm both arms use the same per-gauge test windows. [M2]
6. **Reword the NeuralHydrology claim** from "confirms the findings" to "reaches comparable performance, supporting feasibility." [M3]
7. **Foreground in the Abstract/Conclusion that the LSTM does not beat Ridge or VAR(7) on single-site RMSE**, and locate its genuine advantage in transfer + KGE. [M4]
8. **Demote the "1.82×" and "≈4.6×" cross-dataset multipliers** to indicative (no shared test set). [M5]
9. **Bootstrap KGE CIs** and soften "superior representation of DO dynamics" if intervals overlap. [M6]
10. **Fix the Abstract dataset citation (Nascimento et al. 2025, not Höge et al. 2023)** and reconcile minor stale numbers (temperature NSE 0.727/0.730; cascade reference 0.296/0.303). [L1–L3]
11. **Add one sentence in §4.1 clarifying SAITS is not used**, and frame the "3–4 day effective memory" as a horizon-1 SHAP inference corroborated by the lookback ablation. [L4, L5]

---

## Novelty-claim calibration (specific assessment)

The three novelty claims — "first LSTM river→lake transfer study for DO," "first ML study on CAMELS-CH-Chem," and "among the first EA-LSTM applications to river water quality" — are each hedged with "to our knowledge" / "as of June 2026" / "among the first" (§1, §2.1). This is appropriate hedging for a CAS report and is the correct register. No change needed beyond ensuring the hedges are retained in any future trimming of the Abstract.

---

## Overall score

**7 / 10** for a CAS project.

**Rationale:** This is a strong, methodologically careful CAS report whose *qualitative* scientific reasoning (ecosystem boundaries, transfer limits, correlation-not-causation discipline, honest limitations) is well above the level expected at this tier, and whose headline single-site claim is appropriately deflated to "comparable." It earns a clearly above-average score for the right instincts on CIs, leakage testing, and seed variance.

It is held back from an 8–9 by **claim-calibration and consistency defects that touch the central results**: a contradictory headline p-value (H1), a self-contradictory definition of the flagship 0.451 mg/L number (H2), ablation numbers that don't reconcile with the main table (H3), and a pattern of leaning on un-intervalled or under-powered statistics (KGE point estimate, n=10 Wilcoxon, cross-dataset multipliers) to support superiority narratives that the more rigorous RMSE-with-CI analysis does not support. None of these are fatal — they are fixable in a revision pass — but because several land on the numbers a reader would quote, they materially affect trust. Resolving H1–H3 and softening M1/M3/M6 would move this to an 8.5.
