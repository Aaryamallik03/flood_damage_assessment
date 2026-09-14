# Case Study: August 26, 2026 Bhotekoshi/Trishuli Flash Flood

## Event Summary
- **Date/Time:** August 26, 2026, ~08:40 AM — an ice-rock avalanche in the upper Bhote Koshi
  watershed near the Nepal-China border (near the Lhende River, ~20km upstream of
  Rasuwagadhi) triggered a debris flow that transformed into a major flood.
- **Propagation:** Reached Galchhi (10:28), Malekhu (11:50), Mugling (13:00), Devghat (15:20).
  Peak discharge at Devghat: ~5,850 m³/s; total flood volume ~57.4 million m³.
- **River system:** Bhote Koshi → Trishuli → Narayani corridor
- **Districts affected:** 5 total — Rasuwa and Nuwakot (severely, ~1.6 million people
  impacted), Dhading, Gorkha, and Chitwan (moderately)
- **Source:** NDRRMA Situation Report #01, issued September 1, 2026, 09:00 AM
  (ndrrma.gov.np/mediafiles/rasuwa/Rasuwa_Flood_SitRep_Temp_ENG_01_01092026.pdf)

## Official NDRRMA Figures (as of Sept 1, 2026 Situation Report #01)

| Metric | Value |
|---|---|
| Deaths (incl. recovered remains) | 987 |
| Missing (across 62 districts) | ~3,916 |
| Injured (in treatment) | 279 |
| Total rescued | 11,814 (incl. 253 foreign nationals) |
| Security personnel deployed | 21,011 |
| Bridges washed away / damaged | 41 / 4 |
| Households isolated (Rasuwa) | 3,702 (population 14,461) |
| Holding centres / people sheltered | 30 centres, 3,930 people (Nuwakot: 19 centres/2,653 people; Rasuwa: 11 centres/1,277 people) |

### Deaths by district (as of Sept 1 report)
Chitwan 321, Nawalparasi East 216, Nawalparasi West 170, Nuwakot 95, Gorkha 65,
Dhading 55, Tanahun 38, Rasuwa 27 (total 987 - note deaths were concentrated
downstream as bodies were carried by the river, not necessarily at point of impact).

### Housing & Infrastructure (Nuwakot, from same report)
- 1,267 houses fully damaged, 1,253 partially damaged
- "Satellite mapping identified 4,689 buildings within the flood-affected zone along the
  Bhote Koshi-Trishuli corridor. This figure represents estimated exposure, not verified
  damage." (quoted directly from the NDRRMA report - see note below)
- 19 schools reported across Rasuwa/Nuwakot/Dhading: 8 fully damaged, 8 partially damaged,
  3 physically safe
- Hydropower: 12 named facilities damaged, combined ~783 MW capacity affected

### Death toll and missing-persons trajectory (verified against every report we could obtain)
Figures below are transcribed directly from each report's own headline tickers — see
`reports/ndrrma_headline_timeseries_manual.json` for the full sourced dataset (including
security personnel deployed and total rescued per date), and `src/data/ndrrma_sitrep.py`
for the fetcher/parser used to obtain report #01's full field set programmatically.

| Date | Deaths | Missing (approx.) | Injured | Total Rescued |
|---|---|---|---|---|
| Sep 1 (SitRep #01) | 987 | 3,916 | 279 | 11,814 |
| Sep 2 (SitRep #02) | 1,204 | 4,216 | 301 | 11,993 |
| Sep 3 (SitRep #03) | 1,259 | 5,053 | 5,384 | 12,038 |
| Sep 4 (SitRep #04) | 1,294 | 5,053 | 5,663 | 13,095 |
| Sep 5 (SitRep #05) | 1,331 | 4,896 | 5,929 | 13,101 |
| Sep 7 (SitRep #07) | 1,356 | 4,894 | 6,676 | 13,396 |
| Sep 8 (infographic update, 1PM) | 1,357 | 5,326 | 6,827 | 13,583 |
| Sep 9 (SitRep #09) | 1,369 | 5,132 | n/a* | 13,646 |

\*Sept 9's report only exposed a same-day "Treated: 333" figure at health facilities, not a
running injured total, so it's left blank rather than guessed.

Note the large jump in "Injured" between Sep 2 (301) and Sep 3 (5,384) — this isn't a sudden
mass-casualty event; it reflects NDRRMA switching from counting only hospital-admitted cases
to a broader field-treatment figure (including army/APF medical teams) starting with report
#03. Always check what a figure is actually counting before treating a jump as signal.

We were unable to obtain SitRep #06 (there's a gap between #05 and #07 in the set of reports
we could locate), and an earlier, separate Nepali-language series ("स्थिति प्रतिवेदन #01"-"#08")
covers Aug 26-29 — the first ~4 days of the event, before this daily English series began on
Sept 1. That earlier window wasn't transcribed into this table: extracting reliable figures
from Devanagari-script OCR text carries a real misreading risk, and getting this event's
opening death toll wrong would be worse than leaving the gap explicit. If you want that
earlier window added, it's worth doing with a native Nepali reader rather than a quick
transcription.

## Why the NDRRMA Situation Report Is a Better Source Than the BIPAD Portal UI

Browsing BIPAD Portal's Incident module directly (bipadportal.gov.np/incidents/) with
district-level filters for Rasuwa, Nuwakot, and Dhading showed 0-1 deaths recorded -
dramatically lower than the true toll. This isn't a data quality failure of the government's
disaster system overall; it reflects that this large-scale, multi-district disaster was
tracked through centralized Situation Reports (published as PDFs, aggregating input from
District Administration Offices and Emergency Operation Centers) rather than being fully
reflected in BIPAD's per-district Incident module at the time of checking. This is itself
worth a sentence in your report: even within one government disaster-management ecosystem,
different data products can be updated on very different timelines, which is exactly the
kind of latency/fragmentation problem an automated satellite-based assessment layer could help
address by providing an independent, faster estimate that doesn't depend on any single
reporting pipeline catching up.

## Key Finding for Project Motivation: NDRRMA Already Did Rough Satellite Exposure Mapping

The Sept 1 situation report explicitly states satellite mapping identified 4,689 buildings
within the flood-affected zone, but is careful to caveat this as "estimated exposure, not
verified damage." This is a very strong, citable justification for your project:

> "NDRRMA's own official situation report (Sept 1, 2026) used satellite mapping to estimate
> building exposure in the flood zone, but explicitly noted this reflects exposure, not
> verified damage - precisely the gap this project's building-level damage classification
> model (no damage / minor / major / destroyed) is designed to help close, moving from a
> simple exposure count to an automated per-building damage severity estimate."

Use this as a direct quote with citation in your report's introduction/motivation section -
it shows the government itself recognizes this exact capability gap, which is a much stronger
argument than a general "disaster response could benefit from AI" framing.

## Model Outputs (fill in after running the pipeline)

| Metric | Value |
|---|---|
| Estimated flooded area (km²) | |
| Buildings assessed | |
| No damage | |
| Minor damage | |
| Major damage | |
| Destroyed | |
| Model building count vs. NDRRMA's 4,689 satellite-exposure estimate | |
| Model output vs. NDRRMA official figures | |

## Limitations Observed
- Sentinel-1 revisit time meant the earliest available post-flood scene was [X] days after
  the event - note this against how fast an operational system would actually need to be
  (NDRRMA's own satellite exposure mapping was already available by Sept 1, ~6 days after
  the event - that's a benchmark to compare your pipeline's turnaround against).
- xBD training data does not include this specific event or region, so this is a test of
  cross-region generalization, not in-domain performance - be upfront about this in your report.
- Official figures themselves were still being revised over the following week (987 -> 1,204
  -> 1,369 deaths across SitReps #01, #02, #09 respectively — see the verified table above)
  - cross-reference the specific report/date you cite rather than treating any single
  number as final.
- BIPAD Portal's public Incident module did not reflect this event's true scale at the
  district level when checked - the NDRRMA Situation Report PDFs were the authoritative
  source; document which source you used for which figure.
