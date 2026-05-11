# AareML Report — Fact-Check Report

**Prepared:** 2025  
**Scope:** Verification of specific claims from the AareML research report across four categories: first-of-its-kind claims, citation accuracy, fish/ecological facts, and technical claims.

---

## 1. "First-of-Its-Kind" Claims

---

### CLAIM 1a
> "To our knowledge, no published machine learning paper has yet applied predictive modelling to CAMELS-CH-Chem"

**VERDICT: LIKELY CORRECT (as of mid-2025)**

**EVIDENCE:**  
Extensive search found no published ML paper applying predictive modelling to CAMELS-CH-Chem. The dataset itself was only recently published — the Nascimento et al. 2025 data descriptor appeared on EarthArXiv in April 2025 (https://eartharxiv.org/repository/view/9046/) and was formally published in *Scientific Data* in 2025 (DOI: 10.1038/s41597-025-05625-1). The Zenodo data release (version 1.0) was posted July 2025 (https://zenodo.org/records/16158375). Given the recency of the dataset, it is plausible that no ML prediction paper had yet appeared. The EGU 2025 abstract introducing the dataset (https://meetingorganizer.copernicus.org/EGU25/EGU25-1208.html) does not cite any downstream ML modelling work. No counterexample was found.

**SUGGESTED FIX:** None required, but consider adding a date qualifier: "...as of [submission date]..." to hedge against publication lag.

---

### CLAIM 1b
> "AareML establishes the first machine learning benchmark on the CAMELS-CH-Chem Swiss river chemistry dataset"

**VERDICT: LIKELY CORRECT**

**EVIDENCE:**  
Same reasoning as Claim 1a. No prior ML benchmark on CAMELS-CH-Chem was found. The Eawag project page (https://www.eawag.ch/en/department/siam/projects/camels-switzerland/) notes that CAMELS-CH-Chem "facilitates advances particularly in the field of hydrological modelling" but cites no such work. No ML benchmarking study was identified in any search.

**SUGGESTED FIX:** None required. Qualifier for submission date advisable.

---

### CLAIM 1c
> "making this a genuine first-of-its-kind cross-ecosystem transfer study" (river-to-lake LSTM transfer)

**VERDICT: FLAG — partially unverified, one related precedent exists**

**EVIDENCE:**  
A directly relevant precedent was found: Myers et al. (2025), "Leveraging Transfer Learning to Predict Floodplain Dissolved Oxygen" (https://repository.library.noaa.gov/view/noaa/72760/noaa_72760_DS1.pdf), applies domain-adaptation LSTM transfer learning from rivers to floodplains for DO prediction. The LSTM was pre-trained on 480 USGS river gauges then fine-tuned to 7 Vermont floodplain sites. While floodplain ≠ lake (floodplains are hydrologically connected to and more similar to rivers than standing lakes are), this study weakens a broad "first-of-its-kind cross-ecosystem" claim. No study found performing river-to-standing-lake LSTM transfer for DO specifically.

**SUGGESTED FIX:** Narrow the claim: "To our knowledge, the first LSTM transfer study from rivers to standing lakes for dissolved oxygen prediction" to avoid conflation with floodplain or estuarine transfer work.

---

## 2. Citation Accuracy

---

### CLAIM 2a — Kratzert et al. (2018, 2019)
> "single LSTM trained across 531 US basins outperforms calibrated process-based models on 43% of catchments"

**VERDICT: FLAG — the 531-basin figure is correct but the "43%" figure is not supported by the cited papers**

**EVIDENCE:**  
- **Kratzert et al. (2018)** (https://hess.copernicus.org/articles/22/6005/2018/) used **241 catchments**, not 531. It compared LSTM against SAC-SMA and showed competitive performance. No "43%" figure appears.
- **Kratzert et al. (2019) "Benchmarking"** (https://hess.copernicus.org/preprints/hess-2019-368/hess-2019-368.pdf) used **531 basins** and showed the EA-LSTM "not only significantly outperforms hydrologically models that were calibrated regionally but also achieves better performance than hydrological models that were calibrated for each basin individually." The paper states VIC scored higher than EA-LSTM ensemble in only **2 out of 447 basins (0.4%)** and mHM scored higher in only **16 basins (3.58%)**. This means the LSTM outperforms on **~96–99.6% of catchments**, not 43%.
- **Kratzert et al. (2019) "Ungauged Basins"** (https://hess.copernicus.org/articles/23/5089/2019/) compared ungauged LSTM (trained on 531 basins, tested in leave-out basins) to per-basin calibrated SAC-SMA and showed higher median NSE (0.69 vs 0.64). No 43% figure.

The "43%" figure cannot be sourced to either Kratzert 2018 or 2019. It may be a misattribution or confusion with a different paper or metric.

**SUGGESTED FIX:** Remove the "43%" figure or identify its true source. The correct characterisation of Kratzert 2019 (Benchmarking) is that a single EA-LSTM trained on 531 basins outperforms both regionally- and individually-calibrated process-based models (VIC, mHM, SAC-SMA, FUSE, HBV) across the vast majority of the 447 evaluated catchments.

---

### CLAIM 2b — Zhi et al. (2021)
> "LSTMs predict river DO substantially better than process-based models at weekly timescales"

**VERDICT: FLAG — partially inaccurate characterisation; no process-based model comparison is made in this paper**

**EVIDENCE:**  
The Zhi et al. (2021) paper (https://pubs.acs.org/doi/abs/10.1021/acs.est.0c06783) does not compare LSTM to any process-based model and does not specifically discuss "weekly timescales" in its abstract or key findings. Its core contribution is demonstrating that an LSTM trained on hydrometeorology can predict DO in **236 US watersheds** from the CAMELS-chem dataset, including in "chemically ungauged basins." The paper reports that the model achieved satisfactory performance (NSE ≥ 0.4) for 74% of the core evaluation group of 84 sites, with mean RMSE of 1.2 mg/L. The paper notes it misses DO peaks and troughs when in-stream biogeochemical processes are important. No explicit comparison to a process-based model is made, and no weekly-timescale analysis is highlighted in the paper's abstract or GitHub summary (https://github.com/WeiZhiWater/From-Hydrometeorology-to-River-Water-Quality-Can-a-Deep-Learning-Model-Predict-Dissolved-Oxygen).

**SUGGESTED FIX:** Revise to: "Zhi et al. (2021) demonstrated that an LSTM trained on daily hydrometeorology data can predict river DO across 236 US watersheds, including in chemically ungauged basins, achieving satisfactory performance (NSE ≥ 0.4) at 74% of evaluated sites." The "substantially better than process-based models at weekly timescales" claim should be removed or re-sourced to a paper that actually makes that comparison.

---

### CLAIM 2c — McAfee et al. (2025)
> "LakeBeD-US, 21 US lakes, DO RMSE 1.40 mg/L"

**VERDICT: CONFIRMED**

**EVIDENCE:**  
The LakeBeD-US paper (https://essd.copernicus.org/articles/17/3141/2025/) confirms:
- **21 lakes** in the United States ✓
- **DO RMSE: 1.40 ± 0.09 mg/L** on the testing split (Table 12 of the paper) ✓
- The paper is by **McAfee et al.** (Bennett J. McAfee et al.) ✓

**SUGGESTED FIX:** None required.

---

### CLAIM 2d (Technical sub-claim) — LakeBeD-US Optuna trials
> "report says 50, but we use 75" — check whether McAfee used 50 or 75

**VERDICT: CONFIRMED — McAfee used 50 Optuna trials**

**EVIDENCE:**  
The LakeBeD-US paper states explicitly: "Model architecture and learning hyperparameters were optimally chosen using the 'tree-structured Parzen estimator' algorithm in the Optuna library by minimizing the validation cost over **50 trials**" (https://essd.copernicus.org/articles/17/3141/2025/essd-17-3141-2025.pdf). The report's claim that McAfee used 50 is **correct**. If AareML uses 75 trials, this is an enhancement over the baseline.

**SUGGESTED FIX:** None for citation accuracy. If comparing Optuna configurations, it may be worth noting "we used 75 Optuna trials compared to 50 in McAfee et al. (2025)" to make the methodological difference explicit.

---

### CLAIM 2e — Nascimento et al. (2025)
> "CAMELS-CH-Chem dataset, 86 Swiss gauges, 1981–2020, 115 catchment attributes"

**VERDICT: PARTIALLY CONFIRMED — significant discrepancy in gauge count and attribute count**

**EVIDENCE:**  
From the published paper (https://www.research-collection.ethz.ch/server/api/core/bitstreams/b7131bd4-5232-4790-a836-cca37b67ccb9/content) and Zenodo release (https://zenodo.org/records/16158375):
- **Total catchments: 115** (not 86) ✓ — this is the overall dataset size
- **Gauges with high-frequency (DO) data: 86** ✓ — the paper states "In total, 86 locations have high-frequency measurement data available"
- **Time period: 1981–2020** ✓ confirmed
- **Water quality parameters: up to 40** — the paper does not state "115 catchment attributes"; it says "up to 40 water quality parameters" for 115 catchments. The number 115 refers to the number of catchments, not the number of attributes.

The claim "86 Swiss gauges, 115 catchment attributes" conflates two different numbers: 86 = gauges with high-frequency DO data; 115 = total catchments in the dataset. The paper does not state "115 catchment attributes."

**SUGGESTED FIX:** Revise to: "Nascimento et al. (2025): CAMELS-CH-Chem, 115 Swiss catchments (86 with high-frequency DO data), 1981–2020, up to 40 water quality parameters." If "115 catchment attributes" was intended to refer to static catchment attributes from the parent CAMELS-CH dataset, that should be clarified with a citation to Höge et al. (2023).

---

### CLAIM 2f — Bärenbold et al. (2026)
> "Swiss lake monitoring dataset, 21 lakes"

**VERDICT: CONFIRMED**

**EVIDENCE:**  
Found as a preprint posted March 2026: Bärenbold et al. (2026), "Long-term temperature, oxygen and water clarity trends in Swiss lakes," *Earth System Science Data Discussions* (https://essd.copernicus.org/preprints/essd-2026-142/). The paper by Fabian Bärenbold, Camilla Capelli, et al. at Eawag presents a harmonised dataset of temperature, electrical conductivity, dissolved oxygen, and Secchi depth for **21 large Swiss lakes and lake basins** (from the beginning of consistent records to end of 2023). First author is Fabian Bärenbold (fabian.baerenbold@eawag.ch). The date 2026 is consistent with the preprint submission (discussion started March 2026).

**SUGGESTED FIX:** Note this is a preprint (under review in ESSD as of early 2026). If the paper is not yet formally published, cite as "Bärenbold et al. (2026, preprint)" with the ESSD Discussions DOI.

---

## 3. Fish / Ecological Facts

---

### CLAIM 3a
> "Swiss trout populations have declined 60% since the 1990s"

**VERDICT: LARGELY CORRECT but the timeframe should be "since the early 1980s"**

**EVIDENCE:**  
An ACS Environmental Science & Technology paper (Burkhardt-Holm et al., 2005, "Where Have All the Fish Gone?", https://pubs.acs.org/doi/pdf/10.1021/es053375z) states: "In Switzerland, the reported trout catch in streams and rivers has plummeted by **60% since the early 1980s**." Additional corroborating sources:
- A Frontiers in Veterinary Science paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC6714597/) states: "Beginning in the 1980s, the catch of brown trout in Switzerland experienced a massive decline of up to 50%."
- A second study (core.ac.uk/download/pdf/33052688.pdf) reports catch of brown trout declined by "approximately 40–50% over the last 10–20 years" (from ~2007), and specifically notes the Emme River declined ~60%.
- The Fischnetz final report (fischereiberatung.ch) documents declines exceeding 30% in major Swiss rivers.

The 60% figure is well-supported, but the decline started in the **early 1980s**, not the 1990s. The 1990s saw continued decline but the trend originates ~15 years earlier.

**SUGGESTED FIX:** Change "since the 1990s" to "since the early 1980s." Exact wording: "Swiss trout catches have declined approximately 60% since the early 1980s (Burkhardt-Holm et al., 2005)."

---

### CLAIM 3b
> "Grayling newly elevated to Endangered on Switzerland's Red List in 2022"

**VERDICT: FLAG — elevated, but to "stark gefährdet" (Strongly Endangered / EN), not simply "Endangered"**

**EVIDENCE:**  
The 2022 Swiss Red List for fish (BAFU/info fauna, https://www.bafu.admin.ch/rotelisten) was confirmed via the Kanton Luzern fisheries page (https://lawa.lu.ch/fischerei/artenfoerderung/Aesche): "Auf der vom Bundesamt für Umwelt BAFU im Jahr 2022 veröffentlichten Roten Liste der gefährdeten Arten der Schweiz ist die Fischart neu als **stark gefährdet** eingestuft." A news article from Phys.org (https://phys.org/news/2023-02-swiss-native-fish.html) confirms that the European grayling was among the nine native species that had their threat category raised in the 2022 Red List update. The Eawag press release states grayling endangerment "varies from 'vulnerable' and 'strongly endangered' to 'threatened with extinction' depending on the populations." The IUCN equivalent of *stark gefährdet* is EN (Endangered), so the use of "Endangered" is not wrong per se, but an older source (petri-heil.ch) categorises the grayling as *stark gefährdet* (EN) as of 2018 already, suggesting the 2022 update may have elevated it to a higher category (CR) for some populations, or confirmed EN species-wide.

**SUGGESTED FIX:** Confirm the exact category in the official 2022 Red List PDF (infofauna.ch/sites/default/files/files/publications/rote_liste_der_gefaehrdetenartenderschweizfischeundrundmaeuler.pdf). The claim as written is directionally correct. If the 2022 list elevated grayling from VU to EN (*stark gefährdet*), write "elevated from Vulnerable to Endangered (stark gefährdet)." If it was already EN and the 2022 update merely confirmed this, revise accordingly.

---

### CLAIM 3c
> "Barbel eggs toxic, documented since Roman antiquity"

**VERDICT: FLAG — documented since ~200 years ago (1843), not Roman antiquity**

**EVIDENCE:**  
The medical literature review on barbel cholera (Comelli et al., 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC6502096/) states the toxic effects of barbel eggs "have been described up to two centuries ago," citing the earliest reference as: *Anonymous. Cholera from eating Barbel Roe. Lancet. 1843;39:919.* No reference to Roman antiquity is made. The paper also cites a Provençal proverb ("Never eat eggs of fish whose name begins with b...") as a cultural marker, but this is not attributable to Roman times. No classical Roman source (e.g., Pliny, Galen, Columella) was found documenting barbel egg toxicity specifically.

**SUGGESTED FIX:** Change to "documented since at least the 19th century (Lancet, 1843)" or more cautiously "a long-documented phenomenon in European freshwater fish culture." The claim of Roman antiquity is unsupported and should be removed.

---

### CLAIM 3d
> "Pike can survive DO as low as 0.3 mg/L"

**VERDICT: CONFIRMED**

**EVIDENCE:**  
DFO Canada biological synopsis for Northern Pike (https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/337844.pdf) states: "The species is remarkably tolerant to variations in oxygen level, surviving concentrations as low as **0.3 mg/l** in shallow lakes, and having been caught alive at oxygen concentrations as low as 0.04 mg/l." The CABI Compendium entry for *Esox lucius* (https://www.cabidigitallibrary.org/doi/full/10.1079/cabicompendium.83118) also lists 0.3 mg/L as the harmful minimum DO level, citing Casselman (1978).

**SUGGESTED FIX:** None required.

---

### CLAIM 3e
> "European Eel migrates 6,000 km to Sargasso Sea"

**VERDICT: LARGELY CORRECT — within accepted range but 5,000–6,000 km is the consensus; 6,000 km is the upper end**

**EVIDENCE:**  
Scientific literature gives a range:
- Journal of Experimental Biology (2005) (https://journals.biologists.com/jeb/article/208/7/1329/16002/): "long-distance migration (5000–6000 km)" and "European eel *A. anguilla*, 5500 km"
- Biology Letters / PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC6501356/): "approximately 5000–7000 km away from their juvenile and adult habitats"
- Science Advances (2016) (https://www.science.org/doi/10.1126/sciadv.1501694): "must migrate a distance of between 5000 and 10,000 km (depending on their departure locations)"
- LifeWatch Belgium (https://lifewatch.be/news/press-release-critically-endangered-eels-arrive-our-flemish-rivers-after-6000-kilometer): "migrate for 6000 km towards the Sargasso Sea" (popular source)
- Discover Wildlife (https://www.discoverwildlife.com/animal-facts/fish/eel-migration): "5,000 km outward journey"

The 6,000 km figure appears in popular and some scientific sources as a round number; the scientific consensus is approximately 5,000–6,000 km depending on the eel's origin. For eels departing from Central Europe (Switzerland), the actual distance to the Sargasso Sea is approximately 6,000–7,000 km.

**SUGGESTED FIX:** "approximately 5,000–7,000 km" or "approximately 6,000 km" is acceptable. The claim is broadly correct; no change strictly required, but acknowledging the range would be more precise.

---

## 4. Technical Claims

---

### CLAIM 4a — BAFU/OFEV spending
> "Switzerland spends ~CHF 1.2B/year on water quality monitoring"

**VERDICT: FLAG — the CHF 1.2B figure likely misattributes the broader FOEN budget or confuses it with wastewater infrastructure costs**

**EVIDENCE:**  
The FOEN "brief" page (https://www.bafu.admin.ch/en/the-foen-in-brief) states the FOEN "administers a budget of approximately **CHF 1.3 billion**" — and ~81% of that is subsidies/incentive levy redistribution (CO2, VOC), not water monitoring. The actual operational/personnel budget is ~19% of CHF 1.3B ≈ CHF 250M for all environmental activities. A 2018 Swiss statistics release (https://snbchf.com/2018/04/statistics-switzerland-chf-4-protection-2016/) reports Switzerland spent CHF 11.4B on all environmental protection in 2016, with CHF 2.8B on wastewater and CHF 1.2B on "soil and water conservation." The VSA (Swiss wastewater association) report (https://vsa.ch/wp-content/uploads/2020/07/Branchenbericht_VSA_EN_LR.pdf) references CHF 1.2B in investments for wastewater infrastructure upgrades (micropollutant treatment), not water quality monitoring per se. No source was found supporting "CHF 1.2B/year on water quality monitoring" specifically.

**SUGGESTED FIX:** The "~CHF 1.2B/year on water quality monitoring" claim appears to conflate the total FOEN budget, wastewater infrastructure investment costs, or broader environmental protection expenditure. Recommend replacing with a specific, sourced figure. The FOEN total budget is CHF ~1.3B but covers all environmental domains. The CHF 1.2B figure for "soil and water conservation" from 2016 FSO data is the closest match but covers far more than monitoring. Unless a primary source can be found, this claim should be softened to "Switzerland invests substantially in water quality monitoring through FOEN's NAWA programme" without a specific CHF figure, or it should be properly sourced.

---

### CLAIM 4b — Internal data claim
> "Only 16 of 86 gauges have ≥10% DO data"

**VERDICT: N/A (internal claim from the data itself)**

**EVIDENCE:**  
This is a descriptive statistic derived from the CAMELS-CH-Chem dataset analysis by the AareML authors. It cannot be independently verified without access to the dataset and their analysis code. The existence of 86 high-frequency gauges in CAMELS-CH-Chem is confirmed (see Claim 2e). The sparsity of DO data is consistent with the Nascimento et al. (2025) paper's description of gaps in high-frequency measurements.

**SUGGESTED FIX:** Mark as internal/data-derived claim. Consider adding a sentence in the paper describing the method used to compute data availability (e.g., fraction of non-missing timestamps in the training period).

---

## Summary Table

| # | Claim | Verdict | Key Issue |
|---|-------|---------|-----------|
| 1a | No prior ML on CAMELS-CH-Chem | LIKELY CORRECT | Dataset very recently released |
| 1b | First ML benchmark on CAMELS-CH-Chem | LIKELY CORRECT | No counterexample found |
| 1c | First cross-ecosystem river-to-lake transfer | FLAG | River-to-floodplain LSTM transfer (Myers et al. 2025) is a related precedent |
| 2a | Kratzert 2019: 531 basins, 43% outperformance | FLAG | "43%" not found in either paper; Kratzert 2019 Benchmarking shows ~96-99.6% outperformance |
| 2b | Zhi 2021: LSTM better than process-based at weekly timescales | FLAG | Zhi 2021 makes no comparison to process-based models and does not discuss weekly timescales |
| 2c | McAfee 2025: 21 lakes, DO RMSE 1.40 mg/L | CONFIRMED | Exactly matches Table 12 of LakeBeD-US paper |
| 2d | McAfee used 50 Optuna trials | CONFIRMED | Paper explicitly states 50 trials |
| 2e | Nascimento 2025: 86 gauges, 1981-2020, 115 attributes | PARTIALLY CONFIRMED | 115 = number of catchments (not attributes); 86 = high-frequency gauges; no "115 attributes" |
| 2f | Bärenbold 2026: Swiss lake dataset, 21 lakes | CONFIRMED | Preprint posted March 2026; 21 Swiss lakes confirmed |
| 3a | Trout down 60% since 1990s | FLAG | Correct percentage, wrong decade; decline started early 1980s |
| 3b | Grayling elevated to Endangered 2022 | LARGELY CORRECT | Elevated in 2022 Red List; category is EN (*stark gefährdet*) |
| 3c | Barbel eggs toxic since Roman antiquity | FLAG | Earliest documented reference is 1843 (Lancet), not Roman antiquity |
| 3d | Pike survives DO as low as 0.3 mg/L | CONFIRMED | Confirmed by DFO Canada / CABI Compendium |
| 3e | European Eel migrates 6,000 km to Sargasso | LARGELY CORRECT | Scientific range is ~5,000–7,000 km; 6,000 km is within the accepted range |
| 4a | Switzerland spends ~CHF 1.2B/year on water monitoring | FLAG | Figure misattributed; FOEN total budget is CHF ~1.3B for all environment; no source for monitoring-specific CHF 1.2B |
| 4b | 16 of 86 gauges have ≥10% DO data | N/A | Internal data statistic; cannot be independently verified |

---

## Key Sources

- Nascimento et al. (2025) CAMELS-CH-Chem: https://doi.org/10.1038/s41597-025-05625-1
- Kratzert et al. (2019) Benchmarking: https://hess.copernicus.org/preprints/hess-2019-368/hess-2019-368.pdf
- Kratzert et al. (2019) Ungauged Basins: https://hess.copernicus.org/articles/23/5089/2019/
- Kratzert et al. (2018): https://hess.copernicus.org/articles/22/6005/2018/
- Zhi et al. (2021): https://pubs.acs.org/doi/abs/10.1021/acs.est.0c06783
- McAfee et al. (2025) LakeBeD-US: https://essd.copernicus.org/articles/17/3141/2025/
- Bärenbold et al. (2026) preprint: https://essd.copernicus.org/preprints/essd-2026-142/
- Myers et al. (2025) river-to-floodplain transfer: https://repository.library.noaa.gov/view/noaa/72760/noaa_72760_DS1.pdf
- Burkhardt-Holm et al. (2005) Swiss trout decline: https://pubs.acs.org/doi/pdf/10.1021/es053375z
- DFO Northern Pike synopsis: https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/337844.pdf
- Comelli et al. (2019) Barbel cholera: https://pmc.ncbi.nlm.nih.gov/articles/PMC6502096/
- FOEN brief: https://www.bafu.admin.ch/en/the-foen-in-brief
- Swiss Red List 2022 (fish): https://www.bafu.admin.ch/rotelisten
- Kanton Luzern grayling: https://lawa.lu.ch/fischerei/artenfoerderung/Aesche
