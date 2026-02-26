# Data Processing Pipeline (4 Steps)

This repository contains a **4-step data processing pipeline** for nanopore / ionic current signal analysis.

Workflow:

**Step 1 (Event Extraction) → Step 2 (KDE Feature Extraction) → Step 3 (Normalization) → Step 4 (Plotting & Clustering)**

Each step produces outputs that are used as inputs for the next step.

---

## 0. Requirements

### 0.1 Python version
Recommended: Python 3.9+ (tested on Windows)

### 0.2 Dependencies

Install required packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn openpyxl pyabf
```
## 1. Step 1 — Event Extraction (refine event windows)

Script: BAs_extract_events.py

### 1.1 Purpose

- Read ABF current signal

- Read an event-time Excel (start/end times)

- Convert event times to ABF indices

- Filter invalid events

- Refine event boundaries

- Export refined event time Excel

- Export event preview images (limited count)

### 1.2 Key parameters (edit inside script)

- EVENT_START_COL / EVENT_END_COL: column names in the input Excel for event start/end times (usually in ms)

- INPUT_ROOT / OUTPUT_ROOT

- MAX_IMG_COUNT: maximum number of preview images per file

- BATCH_MODE: batch mode vs single file mode

### 1.3 Inputs

- ABF: *.abf

- Event time Excel: *.xlsx containing EVENT_START_COL and EVENT_END_COL

- Batch mode rule: ABF and Excel must match by key.

### 1.4 Outputs

- Refined event time Excel:

  - Refined event time_{key}.xlsx

- Event preview images:

  - O- UTPUT_ROOT/{key}/raw event_*.png

### 1.5 Output Excel columns

The refined event time Excel typically contains:

- Refined event number

- Original interval index

- Refinement start time (s)

- Final refinement completion time (s)

- Refine event duration (s)

## 2. Step 2 — KDE + Feature Extraction (event feature table)

Script: BAs_KDE.py

### 2.1 Purpose

For each refined event window:

- Extract event signal from ABF

- Perform cleaning (<= -10 pA)

- KDE estimation + peak detection

- Support single peak / double peak

- 3σ cleaning visualization

- Export KDE diagnostic plots (per-event)

- Export event feature Excel (for the next normalization step)

### 2.2 Inputs

- ABF: *.abf

- Refined event time Excel from Step 1:

  - Refined event time_{key}.xlsx

### 2.3 Outputs

Under output folder <output_root>/<key>/:

- KDE diagnostic images:

  - event_{i}_Kernel density analysis graph图.png

- Event feature Excel:

  - {key}_Event characteristic parameter statistics.xlsx

### 2.4 Output Excel columns (important)

The event feature Excel contains (typical structure):

- Refined event number

- Original interval index

- Number of peaks

- Duration of the event

- mean1

- SD1

- mean2

- SD2

- IsReal (True if peak count != 0)

Meaning:

- If Number of peaks == 2:

  - mean1/SD1 = high level

  - mean2/SD2 = low level

- If Number of peaks == 1:

  - mean1/SD1 and mean2/SD2 are both filled with the single level stats

## 3. Step 3 — Normalization (compute DT/M/S features)

Script: FN_归一化.py

### 3.1 Purpose

Transform Step 2 output into normalized ML-ready features:

- Compute DT = log10(duration)

- Compute M1, S1, M2, S2 using the real CD current from a parameter table

### 3.2 Inputs

- Event feature Excels from Step 2:

  - {key}_Event characteristic parameter statistics.xlsx

- Parameter table Excel (you create manually), e.g. params.xlsx

Recommended parameter table columns:

- key

- open_pore_current (optional)

- cd_current (required)

### 3.3 Key matching rule

Key definition: filename prefix up to the first underscore.

Example:

- file: CDCA_15_filtered_Event characteristic parameter statistics.xlsx

- key used for lookup: CDCA

So your parameter table should contain CDCA (not CDCA_15_filtered) if you want key to be CDCA.

### 3.4 Normalization formulas (FINAL)

For each event:

- DT = log10(|duration|)

- M1 = (cd_current - mean1) / cd_current

- S1 = SD1

- M2 = (cd_current - mean2) / cd_current

- S2 = SD2

Important: This final normalization does not use open pore current.

### 3.5 Outputs

Processed Excel files:

- processed_{original_filename}.xlsx

Typically includes:

- ID columns (e.g., Refined event number,Original interval index , Number of peaks, IsReal)

- New features: DT, M1, S1, M2, S2
## 4. Step 4 — Plotting + DBSCAN clustering (noise removal + dataset export)

Script: BAs_cluster.py

### 4.1 Purpose

Using normalized features:

- Filter only IsReal == True

- Build scatter points:

  - SD plot uses (M, S)

  - DT plot uses (M, DT)

- For double-peak events: add 2 points per event

- DBSCAN clustering on SD and DT separately

- Mark noise points / abnormal events

- Export final dataset

- Export SD/DT scatter plots (pre/post denoise)

- Export combined scatter plots for selected classes

### 4.2 Inputs

- Output from Step 3:

  - processed_*.xlsx

### 4.3 Outputs

- Per-file cleaned dataset:

  - final_{input_file}.xlsx

  - Adds:

    - IsNoise

    - type (class extracted from filename)

- Scatter plots:

  - Total plots:

    - Selected category SD total scatter plot.png

    - Selected category DT total scatter plot.png

### 4.4 Filename → type extraction

type is extracted from filename like:

- processed_CDCA_...xlsx → type = CDCA
