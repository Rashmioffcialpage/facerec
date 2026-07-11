# Design Doc: facerec

## Goal

Reproduce OpenFace's actual capability set — face detection, a face
embedding such that same-person distance is small and different-person
distance is large, comparison, and identity classification — with a
dependency stack that installs cleanly today.

## Why not OpenFace's actual stack

OpenFace's embedding network (`nn4.small2`) is a Torch7/Lua model. Torch7
(the Lua one, not PyTorch) has been unmaintained for years; getting it
building on a current machine is a real undertaking before any face
recognition code even runs.

## Why dlib/face_recognition specifically, and a dependency conflict it avoided

The first instinct was `facenet-pytorch` (a maintained PyTorch FaceNet
implementation, architecturally closer to what OpenFace was itself
inspired by). Installing it **downgraded torch from 2.6.0 to 2.2.2 and
torchvision to 0.17.2** on this machine, which broke `torchvision::nms`
(a real, immediate breakage — `torchvision.models.detection` stopped
importing) and put `torchaudio`, `ctgan`, and `opacus` into a reported
version conflict. This is exactly the kind of side effect worth taking
seriously: a face-recognition demo isn't worth breaking other projects'
environments over.

Fix: uninstalled `facenet-pytorch`, restored `torch==2.6.0` /
`torchvision==0.21.0`, and switched to `dlib` + `face_recognition`
instead, which have **zero PyTorch/TensorFlow dependency at all** —
installing them is inherently safe against this class of conflict, which
was directly verified (torch/torchvision versions checked before and
after installing both packages, unchanged).

`dlib`'s bundled face recognition model is itself a ResNet trained with a
triplet-loss-style metric learning objective — the same *category* of
technique as OpenFace's own network, and reports higher accuracy on LFW
(99.38%) than OpenFace's nn4.small2 (~92.9%) in dlib's own published
benchmarks.

## Verification methodology

Real official White House portrait photos of Barack Obama and Joe Biden
(the standard test assets used throughout the `face_recognition` library's
own README and test suite) were used to verify:

1. **True positive**: two different photos of the same person
   (`obama.jpg` vs `obama2.jpg`) → distance 0.3457, correctly below the
   0.6 threshold.
2. **True negative**: two different people (`obama.jpg` vs `biden.jpg`) →
   distance 0.8402, correctly above threshold.
3. **Classifier generalization**: a third Obama photo (`obama3.jpg`),
   *never seen during training*, correctly classified as 'obama' (65.6%
   confidence).

## A real classifier failure case

Training `classifier.py` on `training-images/` (2 Obama photos, 1 Biden
photo) and then running inference on `biden.jpg` itself — a training
example — produced `predicted 'obama' (confidence 0.719)`. Wrong, and on
its own training data.

Diagnosis (see the exact commands/output in git history — this wasn't
guessed, it was measured):

```python
>>> clf.decision_function([biden_embedding])
[0.34753453]   # positive margin toward 'obama' -- not a probability-calibration artifact
>>> clf.n_support_
[1 2]          # only 1 support vector available for the 'biden' class
```

The raw SVM decision boundary itself — not just `predict_proba`'s Platt
scaling — favors 'obama'. With only one example for a class, `SVC` cannot
determine a boundary that reliably separates it from a 2-example class;
this is expected small-sample-size SVM behavior, not a bug in
`classifier.py` or `facelib.py`. **The practical fix is more training
photos per identity** (a handful minimum), not a code change — documented
here rather than silently working around it with a held-out validation
image that happened to look better, because the whole point of testing
this project for real was to find failures like this instead of hiding
them.

## Tested vs. not tested

**Tested, with real photos, on this machine:**
`facelib.py` (detection + embedding), `compare.py`, `represent.py`,
`classifier.py train` and `infer` (including the failure case above).

**Not tested:** `webcam_demo.py`. This machine has no camera. The code
follows the same pattern as the tested `classifier.py infer` path
(load embedding, `clf.predict_proba`), so it's likely correct, but
"likely correct" is not "verified" — try it on your own hardware before
trusting it, and expect to debug camera/OpenCV backend issues that are
inherently untestable here.

## What was cut

- **No face alignment step.** OpenFace explicitly aligns faces (rotates/
  crops to a canonical pose) via dlib landmarks before embedding.
  `face_recognition`'s embedding model handles unaligned crops internally
  (it's trained on non-aligned data), so a separate alignment step wasn't
  necessary here — verified indirectly by the true-positive/true-negative
  results above, which used un-aligned source photos throughout.
- **No triplet-loss training code.** Both OpenFace's and dlib's embedding
  networks are pretrained; neither this repo nor the original ships code
  to train a new embedding network from scratch, only to use one.
