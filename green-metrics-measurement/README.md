# Green Metrics setup for `predict_only_without_app`

This folder contains a Green Metrics Tool measurement setup for the SAM3 sky-mask inference pipeline.

The scenario uses `predict_single_pipeline_run.py`, which predicts only:

- `test_images/Alfred/Alfred-Kaestner-Sonne/testPanorama.png`

## Why this setup

- Uses a dedicated Docker volume for `sam3-bucket`.
- The volume is mounted over the project path `sam3-bucket`, so any checked-in/local model files are ignored.
- `setup.sh` runs inside the measurement container and downloads the SAM3 artifacts on the measurement machine via `hf sync`.

## Files

- `usage_scenario.yml`: Green Metrics scenario flow.
- `docker-compose.yml`: service and volume mapping for measurement runs.
- `Dockerfile`: minimal runtime image with Python and Hugging Face CLI.

## Notes

- GPU access is enabled in `usage_scenario.yml` with `--gpus=all` (same style as the Ollama example).
- If your measurement machine has no NVIDIA GPU, remove `--gpus=all` from `usage_scenario.yml`.
- The named Docker volumes (`sam3-model-cache`, `sam3-venv-cache`) persist between runs on the measurement machine.
- To force a cold model download benchmark, remove the `sam3-model-cache` volume before running.
