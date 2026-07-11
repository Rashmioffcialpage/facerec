"""
Core face recognition primitives: detect faces in an image and compute a
128-dimensional embedding for each one, such that embeddings of the same
person's face are close together (Euclidean distance) and different
people's are far apart. This is the same idea as OpenFace's embedding
network (itself inspired by FaceNet's triplet-loss training), backed here
by dlib's pretrained ResNet face-recognition model via the `face_recognition`
package instead of OpenFace's Torch7/Lua nn4.small2 model.
"""

import os

import face_recognition
import numpy as np

DEFAULT_DISTANCE_THRESHOLD = 0.6  # dlib's own recommended same-person cutoff


def embed_image(path):
    """Returns (embeddings, face_locations) for every face found in an image."""
    image = face_recognition.load_image_file(path)
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    return encodings, locations


def embed_largest_face(path):
    """Returns the embedding of the largest detected face, or None if no face found."""
    encodings, locations = embed_image(path)
    if not encodings:
        return None
    areas = [(bottom - top) * (right - left) for (top, right, bottom, left) in locations]
    idx = int(np.argmax(areas))
    return encodings[idx]


def distance(emb_a, emb_b):
    return float(np.linalg.norm(emb_a - emb_b))


def is_same_person(emb_a, emb_b, threshold=DEFAULT_DISTANCE_THRESHOLD):
    return distance(emb_a, emb_b) < threshold


def load_identity_folder(root_dir):
    """
    root_dir/
        person_a/*.jpg
        person_b/*.jpg
    Returns (embeddings, labels), one row per detected face across all images.
    """
    embeddings, labels = [], []
    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_dir, fname)
            emb = embed_largest_face(path)
            if emb is not None:
                embeddings.append(emb)
                labels.append(person)
    return np.array(embeddings), labels
