# Task Templates

## Data Collection
- Verify source ownership and license
- Log dataset name/version/date and path
- Download via authenticated client; checksum verify

## Data Processing
- Validate schema; fail on missing critical fields
- No imputation with fabricated values without user approval
- Persist lineage: input files → transforms → outputs

## Modeling
- Use real datasets only; document train/val/test split logic
- Log params/metrics; save models with versioned artifact paths

## Dashboard/Server
- Serve only real outputs; disable demo toggles
- Expose health/info endpoints showing data version/timestamp
