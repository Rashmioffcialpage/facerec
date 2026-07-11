# facerec

Face detection, 128-dimensional face embeddings, face comparison, and
identity classification — a modernized -implementation of
openface(CMU's OpenFace),
rebuilt on `dlib`/`face_recognition` instead of OpenFace's Torch7/Lua
`nn4.small2` network. Same idea (a deep embedding network trained so same
-person faces land close together and different people land far apart),
different — and more maintainable in 2026 — implementation. See
[DESIGN.md](DESIGN.md) for why, and for a real classifier failure case
found and diagnosed while testing this.

## Setup

```bash
pip install -r requirements.txt
```

`dlib` and `face_recognition` don't depend on PyTorch or TensorFlow at
all, so installing this won't touch/downgrade either of those if you have
them installed for other projects (this was tested directly — see
DESIGN.md for a dependency conflict this avoided).

## What's here

- `facelib.py` — core primitives: `embed_image()` / `embed_largest_face()`
  (detect + 128-d embed), `distance()`, `is_same_person()`,
  `load_identity_folder()` (reads a `person_name/*.jpg` directory tree).
- `compare.py` — are these two photos the same person? Mirrors OpenFace's
  `demos/compare.py`.
- `represent.py` — batch-embed a directory of per-person photo folders to
  CSV. Mirrors OpenFace's `batch-represent` tool.
- `classifier.py train` / `classifier.py infer` — train an SVM on top of
  face embeddings to recognize specific people, then classify new photos.
  Mirrors OpenFace's `demos/classifier.py`.
- `webcam_demo.py` — real-time webcam classification using a trained
  model. Mirrors `demos/classifier_webcam.py`. **Not tested** — see
  [DESIGN.md](#tested-vs-not-tested).

## Verified: comparing real photos

Using the actual test images from the `face_recognition` library's own
test suite (official White House portrait photos of Obama and Biden):

```bash
$ python3 compare.py examples/obama.jpg examples/obama2.jpg
distance: 0.3457  (threshold: 0.6)
same person

$ python3 compare.py examples/obama.jpg examples/biden.jpg
distance: 0.8402  (threshold: 0.6)
different people
```

Both directions verified correctly: two different photos of the same
person land close together (0.35), two different people land far apart
(0.84), with a clean margin around the 0.6 threshold in both cases.

## Verified: training a classifier, and a real failure case

```bash
$ python3 represent.py training-images --out-prefix reps
embedded 3 faces across 2 identities

$ python3 classifier.py train training-images
trained on 3 faces across 2 identities -> classifier.pkl

$ python3 classifier.py infer examples/obama3.jpg
examples/obama3.jpg: predicted 'obama' (confidence 0.656)
```

Correctly identified a **held-out** third Obama photo the classifier had
never seen. But:

```bash
$ python3 classifier.py infer examples/biden.jpg
examples/biden.jpg: predicted 'obama' (confidence 0.719)
```

This misclassifies `biden.jpg` — which *was* in the training set. This is
a real result, not a code bug: `training-images/biden/` has only 1 photo
vs. `training-images/obama/`'s 2, and an SVM with `n_support_ = [1, 2]`
doesn't have enough points to fit a reliable boundary — even the raw
`decision_function` (not just the calibrated probability) favors 'obama'.
**Use at least a handful of photos per identity**, not 1, or this happens.
See [DESIGN.md](DESIGN.md#a-real-classifier-failure-case) for the full
diagnosis.

## Design

See [DESIGN.md](DESIGN.md) for what changed from OpenFace and why, what
was verified vs. what's untested, and the classifier failure case above
in full detail.
