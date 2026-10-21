# Face Mask Detection: Classical Computer Vision Pipeline

An end-to-end face mask detector built **without deep learning**. It finds a face with a
Haar Cascade, describes its texture with Local Binary Patterns (LBP), and classifies it
as *with mask* or *without mask* using an SVM.

This was an independent project for a Computer Vision course.

## Why not a CNN?

The brief was to use traditional image-processing methods, and to see how far they go.
The whole pipeline runs on a CPU and each face is reduced to just **26 numbers**, so it is
very light. (I have not tested it on edge hardware, so that part is untested.)

## Pipeline

1. **Load and split:** 7,553 images from a public Kaggle dataset, split 80/20 into
   6,042 training and 1,511 test images *before* any analysis, to avoid leakage.
2. **Preprocess:** grayscale, histogram equalisation, 5x5 Gaussian blur, resize to 100x100.
3. **Localise the face:** OpenCV's pre-trained `haarcascade_frontalface_default.xml`
   (`scaleFactor=1.1`, `minNeighbors=4`). If no face is found, the whole image is used.
4. **Extract features:** uniform LBP (radius 3, 24 sampling points), summarised as a
   26-bin histogram per face.
5. **Classify:** scikit-learn SVM with an RBF kernel (`C=1.0`, `random_state=42`).
6. **Visualise:** a prototype function draws a colour-coded box (green = mask,
   red = no mask) on any input image.

## Dataset

[Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) by
`omkargurav` on Kaggle: 3,828 *without mask* and 3,725 *with mask* images.

The images are **not included in this repository**. Download them from Kaggle and place
them so the notebook can find `./data/with_mask` and `./data/without_mask`. Please follow
the dataset's own terms on its Kaggle page.

## Results

Evaluated on 1,511 held-out test images.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Without mask | 0.74 | 0.82 | 0.78 | 766 |
| With mask | 0.79 | 0.71 | 0.75 | 745 |
| **Overall accuracy** | | | **0.77** | 1,511 |

Confusion matrix: 630 true negatives, 527 true positives, 136 false positives (unmasked
faces flagged as masked) and 218 false negatives (masked faces missed).

The model finds unmasked faces more reliably than masked ones. Skin has consistent
texture, while masks vary a lot in colour, shape and fabric, and some look like smooth
skin under LBP.

## What went wrong, and why

The first prototype reported "no face detected" for most masked images. The cause is
that Haar Cascades locate faces using the eyes, nose and mouth, and a mask hides the
nose and mouth. I added a fallback: if no face is found, a centred crop is used instead,
so every image still gets a prediction.

## Known limitations

- **The fallback was not measured.** The notebook does not record how often the Haar
  Cascade failed, so the 77% includes an unknown share of images classified from a
  fallback crop rather than a detected face.
- **Training and prototype fallbacks differ.** During training and evaluation the
  fallback is the whole image. In the final prototype function it is the centre 60% of
  the image. The prototype therefore does not process images exactly as the evaluated
  pipeline did.
- **LBP only sees micro-texture.** Beards, shadows and low contrast can look like fabric
  to the classifier, and some fabric masks look like skin.
- **One dataset, one split.** There is no cross-validation or external test set, so the
  accuracy may not carry over to real-world photos.

## Next steps

Replace the Haar Cascade and LBP + SVM stages with a small CNN, which learns shape and
context instead of local texture and should handle masked faces much better. The same
train/test split would allow a direct comparison.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook CVProject_FaceMaskDetection.ipynb
```

Run the notebook top to bottom after the dataset is in `./data`.

## Tech stack

`Python` · `OpenCV` · `scikit-image` · `scikit-learn` · `NumPy` · `Matplotlib`

## Credits

- Dataset: `omkargurav` on Kaggle (link above).
- Face detector: OpenCV's pre-trained Haar Cascade.

## License

Released under the MIT License. See `LICENSE`.
