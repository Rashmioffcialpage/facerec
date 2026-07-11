"""
Batch-represent: generate embeddings for a batch of images, organized as
one subfolder per identity. Mirrors OpenFace's batch-represent tool, which
also took a directory of per-person subfolders and wrote out embeddings +
labels as CSVs.

    training-images/
        person_a/*.jpg
        person_b/*.jpg
"""

import argparse
import csv

from facelib import load_identity_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir")
    parser.add_argument("--out-prefix", default="reps")
    args = parser.parse_args()

    embeddings, labels = load_identity_folder(args.images_dir)
    print(f"embedded {len(labels)} faces across {len(set(labels))} identities")

    reps_path = f"{args.out_prefix}.csv"
    labels_path = f"{args.out_prefix}_labels.csv"

    with open(reps_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in embeddings:
            writer.writerow(row.tolist())

    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([label])

    print(f"wrote {reps_path} ({embeddings.shape if len(embeddings) else '(empty)'})")
    print(f"wrote {labels_path}")


if __name__ == "__main__":
    main()
