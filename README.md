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

## Output

Masks are written as testPanorama_mask.png under mirrored folders in output.

## Notes

- The model bucket is large and should not be committed.
- You can override the bucket URI:

```bash
SAM3_BUCKET_URI=hf://buckets/moritz-envite-1/sam3-bucket ./setup.sh
```
