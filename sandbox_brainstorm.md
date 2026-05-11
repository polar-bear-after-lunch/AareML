# AareML × Canton Zurich Innovation Sandbox — Brainstorm

**Application draft:** [AareML_InnovationSandbox_Application.docx](AareML_InnovationSandbox_Application.docx)

**Prepared:** May 2026  
**Project:** AareML — ML-based dissolved oxygen and water temperature prediction for Swiss rivers  
**Programme:** Innovation Sandbox for Artificial Intelligence, Canton of Zurich (Phase III, open until 24 May 2026 → September 2026 implementation start)  
**Sources:** [zh.ch Innovation Sandbox](https://www.zh.ch/en/wirtschaft-arbeit/wirtschaftsstandort/innovation-sandbox.html) · [About the Sandbox](https://www.zh.ch/en/wirtschaft-arbeit/wirtschaftsstandort/innovation-sandbox/ueber-die-innovation-sandbox-fuer-ki.html) · [Greater Zurich Area explainer](https://www.greaterzuricharea.com/en/news/zurich-ai-innovation-sandbox) · [AareML GitHub](https://github.com/polar-bear-after-lunch/AareML)

---

## 1. Programme Summary: What Is the Innovation Sandbox?

The [Canton Zurich Innovation Sandbox for AI](https://www.zh.ch/en/wirtschaft-arbeit/wirtschaftsstandort/innovation-sandbox.html) is a structured test environment run by the Division of Business and Economic Development, supported by ETH AI Center, UZH ITSL, swissICT, and the Zurich Metropolitan Area. It is **not** a traditional regulatory sandbox (no temporary law derogations), but rather a collaborative implementation framework that addresses two perennial barriers to AI deployment in Switzerland:

1. **Regulatory opacity** — unclear legal frameworks for AI data use, liability, and governance
2. **Data access bottlenecks** — structured data from public agencies is unavailable or fragmented

**What participants receive:**
- Regulatory consulting: legal questions clarified within the project context (GDPR, cantonal data law, sector-specific regulation)
- Structured access to public data sources from cantonal offices
- A public sector implementation partner (a Zurich authority or institution tests/deploys the solution)
- Knowledge transfer network (ETH, UZH, ZHAW, Lucerne UAS)
- No financial compensation flows, but the sandbox covers costs for third-party services (computing, data processing, hardware)
- Published final report with results and best practices

**What participants must provide:**
- Own AI expertise and human resources (no outsourcing of solution development)
- Swiss presence (headquarters or subsidiary)
- Willingness to publish findings (code/IP excluded)
- Maximum one project submission per organisation

**Selection criteria (published):** maturity of the AI project, regulatory learning potential, societal added value, level of innovation, technical feasibility, transferability to other AI projects.

**Timelines:**
| Phase | Activity | Date |
|---|---|---|
| Phase III application window | Open | Until 24 May 2026 |
| Steering Committee selection | Final evaluation | End of June 2026 |
| Implementation start | Real-world testing begins | September 2026 |
| Results publication | Rolling, public | Ongoing |

**Track record:** Phase I (2022–2024) selected 5 of 21 proposals. Phase II (2024–2026) selected 5 of 24 submissions. **Selection rate ~20–21%.** The programme won the Digital Economy Award 2025 (swissICT) and the Location Promotion Award 2023.

---

## 2. Which Sandbox Category Does AareML Fit?

The sandbox does not officially define formal categories, but operates across three impact levels. AareML primarily fits **two**:

### 2a. Data Access Sandbox (primary fit)
AareML's bottleneck is not the ML model — it already achieves 0.300 mg/L RMSE at 14-day horizons — but **real-time operational data access**. Currently:
- CAMELS-CH-Chem (the training data) covers only **1981–2020** at hourly/daily resolution from FOEN gauges
- BAFU's NAWA FRACHT programme publishes **only the last 7 days of raw data online**, with calibrated data available only on request (hydrologie@bafu.admin.ch)
- Cantonal DO gauge data (Canton Zurich monitors 150 sites on watercourses via [AWEL](https://www.zh.ch/de/umwelt-tiere/wasser-gewaesser/gewaesserschutz/gewaesserqualitaet.html)) is not systematically integrated with federal data
- The Bärenbold et al. (2026) [Swiss lakes dataset](https://opendata.eawag.ch/dataset/long-term-temperature-oxygen-and-water-clarity-trends-in-swiss-lakes) explicitly notes "data access and use [is] difficult… FOEN consolidates data but the process is not always done on a regular basis and the data are not directly accessible to the public"

The sandbox could facilitate **structured, legally clarified access to AWEL's 150 cantonal monitoring sites**, which would dramatically increase the number of gauges available for AareML's multi-site transfer model — currently limited to the 86 CAMELS-CH-Chem gauges.

### 2b. Pilot Deployment Sandbox (secondary fit)
AareML has a working model but no operational deployment. The sandbox would pair it with a **public sector deployment partner** — plausibly AWEL or the cantonal fisheries office (Fischereiinspektorat) — for a real-world early warning use case. This maps exactly to how the bridge monitoring project (Phase II) worked: sensor infrastructure + AI model + cantonal infrastructure authority.

### 2c. Regulatory Clarity (tertiary)
Regulatory questions relevant to AareML include:
- Legal basis for a public cantonal authority (AWEL) ingesting a third-party ML prediction and acting on it (e.g., issuing a fish stress advisory)
- Data sovereignty: CAMELS-CH-Chem is sourced partly from BAFU and partly from cantons — who "owns" derivative model predictions?
- Liability if an AareML prediction triggers a public alert that turns out to be incorrect

---

## 3. Five Concrete Ways AareML Fits the Programme

### 3.1 Real-Time Fish Stress Early Warning System
**The problem:** DO drops below ~4 mg/L cause fish stress; below ~2–3 mg/L, mass mortality begins within hours. Canton Zurich's AWEL monitors 150 watercourse sites, but monitoring is periodic and reactive. Heat waves (2018, 2022, 2023) caused documented fish kills in Swiss rivers.

**The AareML fit:** AareML already includes `notebooks/09_canton_zurich_analysis.ipynb` — a Canton Zurich DO analysis with a national canton ranking, threshold analysis, seasonal patterns, and a DO stress index heat map (`figures/09_zh_river_heat_map.png`). The sandbox would operationalise this into a **14-day look-ahead DO alert system** deployed with AWEL and the cantonal Fischereiinspektorat, giving fisheries officers advance notice of stress events.

**Sandbox value-add:** Legal framework for acting on ML-derived predictions; structured access to AWEL's real-time gauge data; deployment on a cantonal IT infrastructure.

### 3.2 AWEL Gauge Data Integration for Multi-Site Transfer
**The problem:** CAMELS-CH-Chem covers 86 FOEN-monitored gauges. Canton Zurich's AWEL monitors an additional ~150 watercourse sites, but these are not in CAMELS-CH-Chem (the [original CAMELS-CH note](https://essd.copernicus.org/articles/15/5755/2023/) explicitly states "it does not include any data from the Swiss Cantons"). Many AWEL gauges also measure DO and temperature.

**The AareML fit:** AareML's zero-shot transfer already works across 12 Swiss gauges (0.464 mg/L mean RMSE). Incorporating even 20–30 AWEL gauges into the training set would materially improve the multi-site model. The sandbox's core mechanism — "structured access to relevant data sources" — maps precisely to this.

**Sandbox value-add:** Negotiate a data-sharing agreement between AWEL's Environmental Laboratory, BAFU, and the AareML project team; clarify data licensing for ML training; establish a pilot data pipeline.

### 3.3 Climate Adaptation Intelligence for Cantonal Water Management
**The problem:** [BAFU reports](https://www.bafu.admin.ch/en/lake-water-quality) that >60% of Swiss lakes fail the 4 mg/L DO minimum, and "in some lakes… the amount of oxygen-depleted deep water is continuously increasing" (specifically Lake Zurich). Climate change is predicted to worsen thermal stratification and deoxygenation.

**The AareML fit:** AareML's 14-day forecast horizon + SHAP attribution (rediscovering Henry's Law from data; temperature as dominant driver) makes it directly applicable to climate stress scenario planning. The model could produce seasonal DO outlooks that inform cantonal water management planning cycles — exactly the "evidence-based lake management and decision-making" identified by Bärenbold et al. (2026).

**Sandbox value-add:** Connect AareML to BAFU's seasonal hydrological outlooks; integrate with the cantonal Gewässerschutzgesetzgebung planning cycle; publish canton-level risk maps.

### 3.4 Regulatory Framework for AI Predictions in Fisheries Management
**The problem:** There is no established legal framework in Switzerland for a public authority to issue a binding fisheries advisory (e.g., temporary fishing ban, aeration order) based on an ML model output. This is a classic "trust-critical area" that the sandbox prioritises.

**The AareML fit:** The sandbox's UZH ITSL regulatory team would work with AWEL and the cantonal Fischereiinspektorat to define: (a) what evidentiary standard an ML prediction must meet to trigger a management action, (b) model documentation requirements (the LSTM is already documented with 88 tests and SHAP attributions), and (c) liability allocation. The resulting framework would be a transferable template for other ML-in-environment-management applications across Switzerland — high policy-level impact.

**Comparable precedent:** The AI in Medical Documentation project (Phase II) addressed the analogous question for healthcare: when can a professional rely on AI output, and what governance is required?

### 3.5 Zero-Shot Transfer Validation Across All 9 Canton Zurich DO Gauges
**The problem:** AareML notebook 09 includes a cantonal DO ranking for 9 Canton Zurich gauges from CAMELS-CH-Chem. But the AareML team notes that cantonal (non-FOEN) gauges are not in the dataset, limiting coverage.

**The AareML fit:** A focused sandbox project could run AareML zero-shot transfer against all available Canton Zurich AWEL gauges to produce a validated, deployable accuracy map. This is a modest, well-scoped project perfectly matching the sandbox's "implementation in practice" ethos — a quantified deliverable (e.g., "AareML predicts DO within X mg/L at Y% of Canton Zurich gauges") that AWEL can act on.

**Sandbox value-add:** Access to AWEL gauge data; cantonal IT partner for deployment; co-authored public report on ML reliability for cantonal water monitoring.

---

## 4. Value Proposition for Canton Zurich

### 4.1 Water Quality — Specific Canton Zurich Context
- AWEL monitors 11 lakes, 150 watercourse sites, and ~100 groundwater sources periodically ([Kanton Zürich Gewässerqualität](https://www.zh.ch/de/umwelt-tiere/wasser-gewaesser/gewaesserschutz/gewaesserqualitaet.html))
- The Zürich Gewässerbericht 2022 documents persistent water quality challenges despite improvements: micropollutants from agriculture and wastewater, structural habitat degradation, and fish population declines
- Lake Zurich specifically shows increasing DO depletion in deep water ([BAFU lake water quality](https://www.bafu.admin.ch/en/lake-water-quality))
- DO prediction is directly actionable by AWEL, whose mandate includes early detection and remediation initiation

### 4.2 Fish Monitoring and Fisheries Authority
- Eawag's Fish Ecology & Evolution department ([climate change and fish movement project](https://www.eawag.ch/en/department/fishec/projects/fish-movement-patterns)) is building acoustic receiver networks in the Rhine-Aare river network — directly overlapping with AareML's geographic focus
- The cantonal Fischereiinspektorat (fisheries inspectorate) under the Baudirektion is a natural deployment partner for a DO early warning tool; it issues fishing restrictions and responds to fish kills
- AareML's SHAP analysis confirms temperature is the dominant DO driver, directly linking to climate-driven thermal stress research at Eawag

### 4.3 BAFU Integration Pathway
- AareML is already built on BAFU-sourced data (CAMELS-CH-Chem uses FOEN gauge records for 1981–2020)
- BAFU is a core data provider and potential validation partner; a sandbox project creates a formal mechanism to negotiate real-time or near-real-time data feeds beyond the current 7-day raw data window
- BAFU's NAWA and NAWA FRACHT programmes monitor substance loads at selected Swiss watercourses — DO is one measured variable — but calibrated data is only "available on request"

### 4.4 Strategic AI-Hub Positioning
- Canton Zurich explicitly wants to "position and establish [itself] as an AI location internationally" ([Abraxas Magazin on the sandbox](https://magazin.abraxas.ch/inspiration/ki-sandbox-zuerich))
- An environmental AI project in water quality has high public legitimacy and broad transferability to other cantons — fitting the sandbox's emphasis on outputs that "serve as input for shaping future legal frameworks"
- AareML's cross-continental US transfer result is a strong "Switzerland AI exports" story

---

## 5. Potential Partners

| Partner | Role | Why Relevant |
|---|---|---|
| **Eawag** (Swiss Federal Institute of Aquatic Science and Technology) | Scientific anchor + data broker | Lead institution for CAMELS-CH-Chem; positive contact already established. Runs fish movement and lake monitoring projects. Trusted by BAFU and cantons. |
| **BAFU / FOEN** (Federal Office for the Environment) | Data provider + regulatory authority | Owns the NAWA/NAWA FRACHT monitoring network; issues water quality policy. Key to unlocking calibrated real-time DO data. |
| **AWEL** (Amt für Abfall, Wasser, Energie und Luft, Canton Zurich) | Public sector deployment partner | Monitors 150+ watercourse sites; mandated for early detection and remediation. Would be the "public partner" role that every sandbox project requires. Has published the Gewässerbericht 2022 and 2018 showing unresolved DO challenges. |
| **Fischereiinspektorat, Canton Zurich** (under Baudirektion) | End-user / operational deployment | Issues fishing restrictions; responds to fish kills. Natural end-user of a 14-day DO early warning. Provides ground-truth on fish stress events. |
| **ETH AI Center** | Technical advisory + validation | Already a core sandbox partner. Can validate model architecture and provide ML governance guidance. |
| **UZH ITSL** | Regulatory expert | Already embedded in sandbox team. Would draft the legal framework for AI-based advisories in fisheries management. |
| **Statistical Office, Canton Zurich** | Data infrastructure | Already a sandbox partner; could help integrate AWEL gauge data into a standardised pipeline. |
| **ZHAW digital / Lucerne UAS** | Technical partner | Both are sandbox ecosystem partners; ZHAW has environmental engineering and water-related research groups that could provide domain validation. |

---

## 6. Regulatory Barriers the Sandbox Could Remove

### 6.1 Real-Time BAFU/AWEL Data Access
**Barrier:** BAFU NAWA FRACHT publishes only the last 7 days of unvalidated raw data publicly. Calibrated, validated DO time series are "available on request." Canton gauge data (AWEL) is not in CAMELS-CH-Chem at all, even though many AWEL gauges measure DO and temperature.

**Sandbox mechanism:** The programme explicitly facilitates "structured access to relevant data sources" from public offices. A data-sharing MOU between AareML/Eawag and AWEL, negotiated under sandbox auspices and reviewed by UZH ITSL for data protection compliance, would unlock these sources.

**Downstream benefit:** Establishes a replicable template for cantonal environmental data access by AI researchers — a high-leverage policy output.

### 6.2 Legal Basis for AI-Informed Environmental Advisories
**Barrier:** Swiss water protection law (Gewässerschutzgesetz) and cantonal fisheries regulations require specific evidentiary bases for management actions. An ML-predicted DO threshold exceedance is not currently an accepted evidentiary basis for a fisheries advisory or aeration order.

**Sandbox mechanism:** UZH ITSL regulatory analysis would establish whether and how an ML prediction can supplement (not replace) human expert judgment in triggering regulatory actions, and what documentation/audit trail requirements apply (analogous to how the medical documentation project clarified AI's role in clinical documentation under Berufsgeheimnis law).

### 6.3 Data Sovereignty and Model IP
**Barrier:** AareML is trained on CAMELS-CH-Chem, which aggregates data from BAFU, cantonal offices, and Eawag. If AWEL contributes its gauge data to retrain the model, who owns the resulting model weights? Can AareML be deployed commercially, or only as a public-good tool?

**Sandbox mechanism:** The sandbox explicitly does not require IP sharing (only results and best practices). A licensing framework — likely MIT-compatible open model with a public-good deployment clause — could be defined under sandbox governance, providing legal clarity for future commercial or cross-cantonal deployments.

### 6.4 Cross-Cantonal Data Sharing
**Barrier:** Water quality data collection is decentralised across 25 cantons. AareML's zero-shot transfer already demonstrates generalisation across 12 gauges in different cantons, but no formal mechanism exists for cross-cantonal data pooling for ML training.

**Sandbox mechanism:** The sandbox operates under the Metropolitanraum Zürich umbrella (which spans 8 cantons including Schwyz), creating a natural governance structure for a pilot cross-cantonal data-sharing protocol — expandable to all of CAMELS-CH-Chem's 86 gauges in subsequent phases.

---

## 7. Comparison to Successful Sandbox Applications

### Most Analogous: Sensor-Based Bridge Monitoring (Phase II)
**Parallels with AareML:**
- Both involve **continuous sensor data + AI** to predict infrastructure/environmental stress before failure
- Both have a **public safety rationale** (bridge load capacity vs fish mortality / drinking water quality)
- Both require **structured access to sensor data** from a cantonal infrastructure authority
- Both produce an **early warning output** that informs a human decision-maker

**Key structural lesson:** The bridge project succeeded by scoping tightly — one technology (vibration sensors), one infrastructure type (bridges), one cantonal partner. AareML should similarly scope to **one river system in Canton Zurich** (e.g., Limmat or Glatt) with one deployment partner (AWEL) rather than pitching pan-Swiss scale.

### Analogous in Regulatory Terms: AI in Medical Documentation (Phase II)
**Parallels with AareML:**
- Both require a regulatory framework for **AI output informing a professional decision** in a trust-critical domain
- Both have questions around **liability**, **professional responsibility**, and **documentation requirements**
- The medical documentation project's output (legal framework + best practices) is the direct template for the "legal basis for AI-informed environmental advisory" barrier above

**Key lesson:** The sandbox team explicitly values regulatory output that is **transferable** — a legal framework for AI in fisheries management would be reusable by other Swiss cantons and environment agencies.

### Analogous in Data Access Terms: Machine Translation for Public Administration (Phase I)
**Parallels with AareML:**
- Both involve integrating **external AI tools** with **sensitive data held by cantonal offices** (Commercial Register and Integration Office data vs AWEL gauge data)
- Both required **data protection analysis** before deployment
- The project produced "recommendations for use" that were transferable to other departments

**Key lesson:** The sandbox team is comfortable with data access brokering — they have done it before. AareML's data access request (AWEL gauge time series) is not unusual for the programme.

### Precedent for Environmental AI: Autonomous Tractors / Smart Farming (Phase I)
**Parallels with AareML:**
- Both involve AI in **environmental/agricultural** domains with **regulatory uncertainty** (road use / data protection vs DO threshold triggers)
- Both required guidelines for manufacturers/operators on compliance

**Key lesson:** Environmental and agricultural AI projects have succeeded in Phase I. Water quality as a domain is directly continuous with agriculture (nutrient runoff, DO impacts) and has similarly complex regulatory layering. The sandbox team is not restricted to tech/administrative projects.

---

## 8. Application Framing Recommendations

### Core message
> "AareML is a production-ready deep learning system that predicts dissolved oxygen 14 days ahead in Swiss rivers with statistically significant accuracy (RMSE 0.300 mg/L, p=0.024 vs. baseline). We are applying to the sandbox to (a) obtain structured access to AWEL's real-time cantonal gauge data for operational deployment and (b) establish a legal framework for AI-based DO advisories in fisheries management — creating a replicable template for AI in Swiss environmental monitoring."

### Framing checklist (against selection criteria)
| Criterion | AareML Status |
|---|---|
| Maturity | High — v1.17, 88 tests, UBELIX-run, full reports, presentations |
| Regulatory learning potential | High — novel legal question (ML in environmental advisories) |
| Societal added value | High — fish mortality prevention, drinking water protection, climate adaptation |
| Level of innovation | High — first multi-site DO prediction at 14-day horizon for Switzerland |
| Technical feasibility | High — already demonstrated on CAMELS-CH-Chem; zero-shot transfer proven |
| Transferability | High — framework applies to all 25 Swiss cantons; Eawag can disseminate |
| Swiss presence | Yes — Eawag (Dübendorf, ZH), University of Bern |
| Own AI expertise | Yes — full LSTM pipeline, Optuna tuning, SHAP, 88 tests |
| Public partner identified | Proposed: AWEL + Fischereiinspektorat, Canton Zurich |
| Canton Zurich-specific analysis | Yes — dedicated notebook 09 + AareML-canton-zurich.pdf |

### Scope recommendation
Narrow the Phase III application to a focused, achievable scope:
1. **Data access goal:** AWEL shares DO time series for 10–20 Canton Zurich gauges → AareML retrained and validated
2. **Deployment goal:** Real-time 14-day DO alert dashboard for AWEL / Fischereiinspektorat, piloted on 2–3 rivers (e.g., Limmat, Glatt, Thur)
3. **Regulatory goal:** UZH ITSL co-authors a 1-page legal framework note for AI-informed water quality advisories

The national and Swiss lakes extensions (notebooks 10, cross-cantonal data pooling) can be framed as **Phase IV ambition** — demonstrating transferability without over-scoping Phase III.

---

## 9. Key Contacts

| Name | Role | Contact |
|---|---|---|
| Programme Lead | Innovation Sandbox, Canton Zurich | [zh.ch Innovation Sandbox](https://www.zh.ch/en/wirtschaft-arbeit/wirtschaftsstandort/innovation-sandbox.html) |
| Regulatory Expert | UZH ITSL | via sandbox team |
| Scientific Partner | Eawag, CAMELS-CH-Chem lead | (positive contact established) |
| AWEL | Cantonal water authority | via zh.ch/awel |

**Application submission:** Online form via zh.ch Innovation Sandbox page (Phase III closes 24 May 2026)

---

## Summary Matrix

| Task | Finding |
|---|---|
| Sandbox categories for AareML | Primary: Data Access. Secondary: Pilot Deployment. Tertiary: Regulatory Clarity |
| Best fit among 5 concrete applications | (1) Fish stress early warning; (2) AWEL gauge data integration; (3) Climate adaptation intelligence; (4) Legal framework for AI advisories; (5) Validated DO mapping for Canton Zurich |
| Canton Zurich value proposition | AWEL monitors 150+ sites with known DO challenges; Lake Zurich DO deteriorating; Fischereiinspektorat needs early warning tools; canton already has AareML-specific DO analysis (nb09) |
| Key partners | Eawag (scientific anchor), BAFU (data/regulation), AWEL (cantonal deployer), Fischereiinspektorat (end-user), UZH ITSL (legal), ETH AI Center (technical) |
| Regulatory barriers | Real-time BAFU/AWEL data access; legal basis for AI-informed environmental advisories; data sovereignty for ML training; cross-cantonal data sharing |
| Most analogous successful project | Bridge monitoring (Phase II) — sensor data + AI + cantonal infrastructure partner + early warning output |
| Selection probability | Estimated ~20% base rate (5/24 in Phase II); AareML scores high on maturity, societal value, and regulatory novelty — competitive |
