#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def find_panorama_images(test_images_dir: Path) -> list[Path]:
    return sorted(test_images_dir.rglob("testPanorama.png"))


def masks_to_binary(mask_result: dict, image_size: tuple[int, int]) -> np.ndarray:
    masks = mask_result.get("masks", None)
    if masks is None:
        return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    if hasattr(masks, "cpu"):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.size == 0:
        return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    sky_mask = np.any(masks_np.astype(bool), axis=0)
    return (sky_mask.astype(np.uint8) * 255)


def predict_sky_mask(
    image_path: Path,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    prompt: str,
    score_threshold: float,
    mask_threshold: float,
    max_side: int,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    work_image = image

    longest_side = max(original_size)
    if longest_side > max_side:
        scale = max_side / float(longest_side)
        resized = (
            max(1, int(round(original_size[0] * scale))),
            max(1, int(round(original_size[1] * scale))),
        )
        work_image = image.resize(resized, Image.Resampling.BILINEAR)

    inputs = processor(images=work_image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    post = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs["original_sizes"].tolist(),
    )
    result = post[0] if isinstance(post, list) else post
    mask = masks_to_binary(result, work_image.size)
    if work_image.size != original_size:
        mask_img = Image.fromarray(mask, mode="L")
        mask = np.asarray(mask_img.resize(original_size, Image.Resampling.NEAREST), dtype=np.uint8)
    return mask


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sky/no-sky masks with SAM3.")
    parser.add_argument("--model-dir", type=Path, default=Path("sam3-bucket"))
    parser.add_argument("--test-images-dir", type=Path, default=Path("test_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--prompt", type=str, default="sky")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-side", type=int, default=768)
    args = parser.parse_args()

    panoramas = find_panorama_images(args.test_images_dir)
    assert panoramas, "No testPanorama.png files found"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(str(args.model_dir)).to(device)
    processor = Sam3Processor.from_pretrained(str(args.model_dir))

    for image_path in panoramas:
        relative = image_path.relative_to(args.test_images_dir)
        out_name = relative.with_name("testPanorama_mask.png")
        out_path = args.output_dir / out_name

        mask = predict_sky_mask(
            image_path=image_path,
            model=model,
            processor=processor,
            device=device,
            prompt=args.prompt,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            max_side=args.max_side,
        )
        save_mask(mask, out_path)
        print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
