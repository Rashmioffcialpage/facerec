"""
Train a classifier on a folder of labeled face images (one subfolder per
identity) and use it to identify people in new images. Mirrors OpenFace's
demos/classifier.py (train / infer subcommands).
"""

import argparse
import pickle

from sklearn.svm import SVC

from facelib import embed_largest_face, load_identity_folder


def train(images_dir, model_path):
    embeddings, labels = load_identity_folder(images_dir)
    if len(set(labels)) < 2:
        raise SystemExit("need at least 2 identities (subfolders) to train a classifier")

    clf = SVC(kernel="linear", probability=True)
    clf.fit(embeddings, labels)

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"trained on {len(labels)} faces across {len(set(labels))} identities -> {model_path}")


def infer(image_path, model_path):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    emb = embed_largest_face(image_path)
    if emb is None:
        raise SystemExit(f"no face found in {image_path}")

    pred = clf.predict([emb])[0]
    proba = clf.predict_proba([emb])[0]
    confidence = max(proba)
    print(f"{image_path}: predicted '{pred}' (confidence {confidence:.3f})")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("images_dir")
    p_train.add_argument("--model", default="classifier.pkl")

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("image")
    p_infer.add_argument("--model", default="classifier.pkl")

    args = parser.parse_args()
    if args.command == "train":
        train(args.images_dir, args.model)
    else:
        infer(args.image, args.model)


if __name__ == "__main__":
    main()
