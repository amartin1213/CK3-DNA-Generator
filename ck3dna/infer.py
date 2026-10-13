"""ck3dna.infer

First-class inference entry point -- this did **not** exist upstream. Given
a single portrait photo and a trained checkpoint, produces a CK3-style DNA
text file ready to paste into a save or mod.

Usage::

    python -m ck3dna.infer \\
        --image path/to/portrait.jpg \\
        --checkpoint model.ckpt \\
        --output result.txt \\
        --predictor shape_predictor_68_face_landmarks.dat

Pipeline: align the face with dlib -> run it through the trained model ->
convert the predicted tensor back into a DNA text dump with
`ck3dna.dna_parser.tensor_ToDna`.
"""
from __future__ import annotations

import argparse

import torch
from torchvision import transforms

from .alignment import align_face
from .dna_parser import tensor_ToDna
from .training import Net

# Matches the transform used during preprocessing in `ck3dna.preprocess_tensors`.
_TRANSFORM = transforms.Compose([transforms.ToTensor()])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to a portrait photo")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .ckpt file")
    parser.add_argument("--output", required=True, help="Path to write the predicted DNA .txt file")
    parser.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib's 68-landmark shape predictor (see README.md for how to obtain it)",
    )
    parser.add_argument(
        "--output_size", type=int, default=256, help="Alignment crop size in pixels; must match training"
    )
    parser.add_argument("--device", default="cpu", help="torch device to run inference on, e.g. cpu or cuda")
    return parser


def predict_dna(
    image_path,
    checkpoint_path,
    predictor_path: str,
    output_size: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Align `image_path`, run it through the model at `checkpoint_path`, and
    return the raw predicted DNA tensor (see `ck3dna.dna_parser` for its layout).
    """
    aligned = align_face(image_path, predictor_path, output_size=output_size)
    if aligned is None:
        raise RuntimeError(f"No face detected in {image_path}")

    # align_face returns a BGR numpy array (OpenCV convention); flip channels
    # to RGB to match the transform used during training/preprocessing.
    rgb = aligned[:, :, ::-1].copy()
    image_tensor = _TRANSFORM(rgb).unsqueeze(0)

    net = Net.load_from_checkpoint(checkpoint_path, map_location=device)
    net.to(device)
    net.eval()
    with torch.no_grad():
        prediction = net(image_tensor.to(device))
    return prediction.squeeze(0).cpu()


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    prediction = predict_dna(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        predictor_path=args.predictor,
        output_size=args.output_size,
        device=args.device,
    )
    tensor_ToDna(prediction, args.output)
    print(f"Wrote predicted DNA to {args.output}")


if __name__ == "__main__":
    main()
