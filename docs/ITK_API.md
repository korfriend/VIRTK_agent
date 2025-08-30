# ITK Python Quick Reference (80/20)

> Goal: Quick reference for common APIs: image I/O, basic metadata, filter chains, transforms. Prereq: `pip install itk`.

---

## 0) Basic import

```python
import itk
```

---

## 1) Read/Write images

```python
# Read
img = itk.imread("input.nii.gz", pixel_type=itk.F)   # itk.UC=uint8, itk.F=float32

# Write
itk.imwrite(img, "output.mha")
```

---

## 2) Metadata (size, spacing, origin, direction)

```python
sz = itk.size(img)         # (x,y,z)
sp = itk.spacing(img)      # voxel spacing
org = itk.origin(img)      # physical origin
dirM = itk.direction(img)  # 3x3 direction matrix
```

---

## 3) Types & casting

```python
# Explicit image type: (pixel, dimension)
Float3D = itk.Image[itk.F, 3]

# Casting
Caster = itk.CastImageFilter[type(img), Float3D].New(Input=img)
img_f = Caster.GetOutput()
```

---

## 4) Common filters

### Median Filter

```python
med = itk.MedianImageFilter[type(img)].New(Input=img, Radius=2)
out = med.GetOutput()
```

### Connected Threshold (Region Growing)

```python
seed = (50, 60, 30)
seg = itk.ConnectedThresholdImageFilter[
    type(img), itk.Image[itk.UC,3]
].New(Input=img, Lower=100, Upper=300, SeedList=[seed])
mask = seg.GetOutput()
```

### Resample (change resolution/spacing)

```python
res = itk.ResampleImageFilter[type(img), type(img)].New(
    Input=img, OutputSpacing=[1.0, 1.0, 1.0]
)
out_res = res.GetOutput()
```

### Smoothing (Curvature Anisotropic Diffusion)

```python
sm = itk.CurvatureAnisotropicDiffusionImageFilter[type(img), type(img)].New(
    Input=img, TimeStep=0.0625, ConductanceParameter=3.0, NumberOfIterations=5
)
out_sm = sm.GetOutput()
```

### Gradient Magnitude

```python
grad = itk.GradientMagnitudeImageFilter[type(img), type(img)].New(Input=img)
out_grad = grad.GetOutput()
```

---

## 5) Registration (Rigid, Simple)

```python
fixed = itk.imread("fixed.nii.gz", itk.F)
moving = itk.imread("moving.nii.gz", itk.F)

reg = itk.ImageRegistrationMethodv4[itk.Image[itk.F,3], itk.Image[itk.F,3]].New()
# Configure Optimizer, Metric, Transform (advanced)
```

*(Structure only; set parameters per use case)*

---

## 6) Run pipelines

```python
# Way 1: .Update() directly
med.Update()
out = med.GetOutput()

# Way 2: itk.pipeline() helper
out = itk.pipeline(med)
```

---

## 7) NumPy conversion

```python
import numpy as np

arr = itk.array_from_image(img)   # ITK → numpy (z,y,x)
img2 = itk.image_from_array(arr)  # numpy → ITK
```

---

## 8) Transforms (rotation/translation)

```python
tf = itk.Euler3DTransform[itk.D].New()
tf.SetRotation(0.0, 0.0, 0.5)      # radians
tf.SetTranslation((10.0, 0.0, 0.0))

resamp = itk.ResampleImageFilter[type(img), type(img)].New(
    Input=img, Transform=tf, Size=sz, OutputSpacing=sp, OutputOrigin=org
)
out_tf = resamp.GetOutput()
```

---

## 9) Segmentation example: Otsu Threshold

```python
otsu = itk.OtsuThresholdImageFilter[type(img), itk.Image[itk.UC,3]].New(Input=img)
mask = otsu.GetOutput()
```

---

## 10) Tips

* ITK is template-based; in Python, many filters are `Filter[InputType, OutputType]`.
* Type mismatch is common → use `CastImageFilter` as needed.
* You can always annotate `itk.Image[pixel, dimension]` types.
* Preserve spacing/origin/direction for medical images.

---

## 11) Minimal examples

### (A) Median filter

```python
import itk
img = itk.imread("ct.nii.gz", itk.F)
med = itk.MedianImageFilter[type(img)].New(Input=img, Radius=2)
out = itk.pipeline(med)
itk.imwrite(out, "ct_med.nii.gz")
```

### (B) Region Growing

```python
img = itk.imread("ct.nii.gz", itk.F)
seed = (60,70,40)
seg = itk.ConnectedThresholdImageFilter[type(img), itk.Image[itk.UC,3]].New(
    Input=img, Lower=100, Upper=300, SeedList=[seed]
)
itk.imwrite(seg.GetOutput(), "ct_seg.nii.gz")
```

### (C) NumPy roundtrip

```python
img = itk.imread("ct.nii.gz", itk.F)
arr = itk.array_from_image(img)
arr2 = (arr > 150).astype(np.uint8)
mask = itk.image_from_array(arr2)
mask.CopyInformation(img)   # keep spacing, origin
itk.imwrite(mask, "mask.nii.gz")
```

