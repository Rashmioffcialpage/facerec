"""Compare two images: are they the same person? Mirrors OpenFace's demos/compare.py."""

import argparse

from facelib import DEFAULT_DISTANCE_THRESHOLD, distance, embed_largest_face


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image1")
    parser.add_argument("image2")
    parser.add_argument("--threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD)
    args = parser.parse_args()

    emb1 = embed_largest_face(args.image1)
    emb2 = embed_largest_face(args.image2)

    if emb1 is None:
        raise SystemExit(f"no face found in {args.image1}")
    if emb2 is None:
        raise SystemExit(f"no face found in {args.image2}")

    d = distance(emb1, emb2)
    same = d < args.threshold
    print(f"distance: {d:.4f}  (threshold: {args.threshold})")
    print("same person" if same else "different people")


if __name__ == "__main__":
    main()
