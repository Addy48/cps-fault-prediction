# Data

## Raw data (not stored here)

NASA C-MAPSS turbofan engine degradation simulation, from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The notebooks read two raw files: `train_FD001.txt` (notebook 01) and `train_FD003.txt` (notebook 10). Upload them to a Kaggle dataset, or point `BASE` at a local folder.

## Intermediate datasets (in `intermediate/`)

| File | Made by | Contents |
|---|---|---|
| `v2_phase1_clean_dataset.csv` | notebook 01 | FD001 with named columns, 14 informative sensors, `max_cycle` per engine |
| `v2_phase2_labeled_dataset.csv` | notebook 02 | the same plus `label_w10`, `label_w20`, `label_w30` |
| `v2_phase3_feature_dataset.csv` | notebook 03 | rolling 5-cycle mean, standard deviation and slope per sensor |

Notebooks 02 to 10 read these from Kaggle dataset folders (`v2-phase1-clean`, `v2-phase2-labeled`, `v2-phase3-features`, ...). Locally, point `BASE` at `data/intermediate` and adjust the sub-folder names.

The engine split, node layout, dropped sensors and window sizes are in [`../config/config.json`](../config/config.json).
