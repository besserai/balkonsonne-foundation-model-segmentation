# SAM3 Sky Mask Prediction

This folder keeps model weights out of git and downloads them on setup.

## What this does

- creates or reuses a local virtual environment in .venv
- installs Python dependencies from requirements.txt
- syncs the SAM3 bucket to sam3-bucket via hf sync

## One-time setup

```bash
./setup.sh
```

Optional (skip model download):

```bash
./setup.sh --skip-sync
```

## Run prediction

```bash
.venv/bin/python predict_sky_masks.py \
  --model-dir sam3-bucket \
  --test-images-dir test_images \
  --output-dir output
```

## Run one pipeline simulation

This runs exactly one inference on:

- `test_images/Alfred/Alfred-Kaestner-Sonne/testPanorama.png`

and writes:

- `output/Alfred/Alfred-Kaestner-Sonne/testPanorama_mask.png`

```bash
.venv/bin/python predict_single_pipeline_run.py
```

## Output

Masks are written as testPanorama_mask.png under mirrored folders in output.

## Notes

- The model bucket is large and should not be committed.
- You can override the bucket URI:

```bash
SAM3_BUCKET_URI=hf://buckets/moritz-envite-1/sam3-bucket ./setup.sh
```

## Green Metrics measurement setup

A ready-to-run Green Metrics Tool setup is available in `green-metrics-measurement`.

It uses a dedicated Docker volume mounted on `sam3-bucket`, so model artifacts are downloaded on the measurement machine and are not taken from local checked-in files.
