"""
Real-time webcam classification: run a trained classifier.py model against
a live camera feed. Mirrors OpenFace's demos/classifier_webcam.py.

NOT TESTED in this build -- the machine this was built on has no camera
access. Verify it works on your own hardware before relying on it; see
DESIGN.md.
"""

import argparse
import pickle

import cv2
import face_recognition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="classifier.pkl")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="min confidence to show a label")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        clf = pickle.load(f)

    video = cv2.VideoCapture(args.camera_index)
    if not video.isOpened():
        raise SystemExit(f"could not open camera index {args.camera_index}")

    print("press 'q' to quit")
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)

            for (top, right, bottom, left), emb in zip(locations, encodings):
                proba = clf.predict_proba([emb])[0]
                idx = proba.argmax()
                label, confidence = clf.classes_[idx], proba[idx]
                text = f"{label} ({confidence:.2f})" if confidence >= args.threshold else "unknown"

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("facerec webcam demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
