# Data Access Notes

This repository does not redistribute the underlying competition data or the external NIFC download used in the supplementary analyses.

## WiDS Datathon Files

Required for `code/run_pipeline.py`:

- `train.csv`
- `test.csv`

Useful but not strictly required for the current scripts:

- `metaData.csv`

The manuscript describes a training set of 221 fire-zone pairs and an unlabeled test set of 95 observations from the WiDS Datathon 2026 wildfire task.

## Supplementary External File

Required for `code/supplement_analysis.py`:

- NIFC WFIGS perimeter CSV

The script header expects a file such as:

- `WFIGS_Interagency_Perimeters_-848118526729381764.csv`

## How To Obtain The Data

### WiDS data

1. Register through the WiDS Datathon platform.
2. Download the wildfire task files.
3. Store them in a local data directory of your choice.

### NIFC WFIGS data

1. Visit the NIFC open-data portal.
2. Download the WFIGS Interagency Fire Perimeters CSV snapshot used for the supplementary context analysis.
3. Pass the local file path explicitly when running `code/supplement_analysis.py`.

## Path Handling

Recommended usage is to pass data locations explicitly:

```bash
python code/run_pipeline.py --data-dir /path/to/wids_data --output-dir /path/to/output
python code/supplement_analysis.py --train /path/to/train.csv --nifc /path/to/WFIGS.csv --output /path/to/output
```

No raw data files are tracked in this repository.

