# Data Access Instructions

No EEG data is included in this repository. Each dataset requires a separate
registration / data use agreement. This file describes where to obtain each
dataset and the expected on-disk layout.

---

## 1. I-CARE (Primary evaluation dataset)

**Citation:** Kjaergaard et al., PhysioNet 2023  
**Access:** PhysioNet Credentialed Health Data Licence (free registration required)  
**URL:** https://physionet.org/content/i-care/

### Steps
1. Create a PhysioNet account at https://physionet.org/register/
2. Complete the required CITI training and submit your credential application
3. Once approved, download via:
   ```bash
   wget -r -N -c -np --user <your-physionet-username> \
       https://physionet.org/files/i-care/2.0/training/
   ```
4. Place downloaded files at:
   ```
   data/icare/training/<patient_id>/
   ```

### Expected structure
```
data/icare/training/
├── ICARE_0001/
│   ├── ICARE_0001_001.edf
│   └── ICARE_0001_001.hea
├── ICARE_0002/
│   └── ...
└── ...
```

**Note:** The experiment uses only patents with CPC 1–4 and at least one valid
EEG file. CPC-5 patients (n = 353) and patients without usable EEG (n = 199)
are excluded automatically by `src/stage3_doc/dataset_icare.py`.

---

## 2. DEAP (E2 Lorentzian pre-training)

**Citation:** Koelstra et al., IEEE T-AFFC 2012  
**Access:** Free, requires registration at Keele University  
**URL:** https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

### Steps
1. Register and request access at the URL above
2. Download the MATLAB-format preprocessed data (`data_preprocessed_matlab.zip`)
3. Unzip and place at:
   ```
   data/DEAP/DEAP/data_preprocessed_matlab/
   ├── s01.mat
   ├── s02.mat
   └── ... (s01.mat – s32.mat, 32 subjects)
   ```

---

## 3. DREAMER (E2 Lorentzian pre-training)

**Citation:** Katsigiannis & Ramzan, IEEE J-BHI 2018  
**Access:** Contact the authors; available on IEEE DataPort  
**URL:** https://ieee-dataport.org/open-access/dreamer

### Steps
1. Download `DREAMER.mat` from IEEE DataPort (free IEEE account required)
2. Place at:
   ```
   data/DREAMER/DREAMER/DREAMER.mat
   ```

---

## 4. TUH EEG (E1 FractalSSL pre-training)

**Citation:** Obeid & Picone, Frontiers in Neuroscience 2016  
**Access:** Free, requires NEDC account registration  
**URL:** https://isip.piconepress.com/projects/tuh_eeg/

### Steps
1. Register at https://www.isip.piconepress.com/projects/tuh_eeg/
2. Download TUH EEG v2.0.1 (EDF files)
3. Place at:
   ```
   data/tuh_eeg/
   └── v2.0.1/edf/
       ├── 000/
       ├── 001/
       └── ...
   ```

---

## Updating paths

Once data is downloaded, edit `src/config.py` — the four path variables at the top
of Section 2 (DATASET PATHS):

```python
ICARE_DIR    = Path("data/icare/training")
DEAP_DIR     = Path("data/DEAP/DEAP/data_preprocessed_matlab")
DREAMER_MAT  = Path("data/DREAMER/DREAMER/DREAMER.mat")
TUH_DIR      = Path("data/tuh_eeg")
```

All other scripts import from `src/config.py`; no further changes are needed.
