#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from transformers import Sam3Model, Sam3Processor

from predict_sky_masks import predict_sky_mask, save_mask


def build_parser(root_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a sky mask for one fixed panorama image to simulate one pipeline run."
    )
    parser.add_argument("--model-dir", type=Path, default=root_dir / "sam3-bucket")
    parser.add_argument(
        "--input-image",
        type=Path,
        default=root_dir / "test_images/Alfred/Alfred-Kaestner-Sonne/testPanorama.png",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=root_dir / "output/Alfred/Alfred-Kaestner-Sonne/testPanorama_mask.png",
    )
    parser.add_argument("--prompt", type=str, default="sky")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-side", type=int, default=768)
    return parser


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    args = build_parser(root_dir).parse_args()

    if not args.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(str(args.model_dir)).to(device)
    processor = Sam3Processor.from_pretrained(str(args.model_dir))

    mask = predict_sky_mask(
        image_path=args.input_image,
        model=model,
        processor=processor,
        device=device,
        prompt=args.prompt,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        max_side=args.max_side,
    )
    save_mask(mask, args.output_image)
    print(f"saved {args.output_image}", flush=True)


if __name__ == "__main__":
    main()
