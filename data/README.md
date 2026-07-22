# Local data directory

This directory intentionally contains documentation only. Obtain each dataset from the source listed in [`../DATA_AVAILABILITY.md`](../DATA_AVAILABILITY.md), create the analysis-ready form under its applicable terms, and store it here using the expected filename.

Do not commit raw data, analysis-ready participant-level data, or temporary padded CSV files. The `.gitignore` rules enforce this boundary.

For each locally prepared file, add one row to `MANIFEST.tsv` with the source version, retrieval date, byte size, and SHA-256 checksum. The manifest identifies an input without redistributing it.

Example checksum command:

```bash
shasum -a 256 data/framingham.csv
```
