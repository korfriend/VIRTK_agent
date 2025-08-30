# RTK (Radiotherapy ToolKit) — Python Quick Reference (80/20)

> Goal: Minimal steps to build CBCT projection geometry → load projections → FDK/Iterative reconstruction → save. Prereq:
>
> ```bash
> pip install itk itk-rtk
> ```
>
> After install, RTK classes are usually exposed under `import itk`. In some environments, a separate `import rtk` may exist.

---

## 0) Import & availability check

```python
import itk

has_geom = hasattr(itk, "ThreeDCircularProjectionGeometry")
has_fdk  = hasattr(itk, "FDKConeBeamReconstructionFilter")
print("RTK available:", has_geom and has_fdk)
```

---

## 1) Core concepts

- ThreeDCircularProjectionGeometry: C‑arm circular geometry (angles, SID/SDD, offsets)
- ProjectionsReader: load projection stacks (or use ImageSeriesReader pattern)
- FDKConeBeamReconstructionFilter: classic cone‑beam FDK reconstruction
- Iterative algorithms: OSEM/OS‑SART/TV (availability depends on build/wrapping)

---

## 2) Typical FDK pipeline (skeleton)

```python
import itk

# 1) Build projection geometry
geom = itk.ThreeDCircularProjectionGeometry.New()

# Example: set angles/distances (replace with real values)
# for i, ang_deg in enumerate(angles_deg):
#     geom.AddProjection(itk.D(ang_deg), itk.D(SID), itk.D(SDD))
#     geom.SetSourceOffsetsX(i, src_offset_x); geom.SetSourceOffsetsY(i, src_offset_y)
#     geom.SetProjectionOffsetsX(i, det_offset_x); geom.SetProjectionOffsetsY(i, det_offset_y)

# 2) Load projections (single stack or series)
projs = itk.imread("projections.mha", pixel_type=itk.F)  # (nProj, rows, cols) or (z,y,x)

# 3) FDK reconstruction
fdk = itk.FDKConeBeamReconstructionFilter.New()
fdk.SetInput(projs)
fdk.SetGeometry(geom)

# Optional: specify output volume metadata
# fdk.SetOutputOrigin([ox, oy, oz])
# fdk.SetOutputSpacing([sx, sy, sz])
# fdk.SetOutputSize([nx, ny, nz])

fdk.Update()
vol = fdk.GetOutput()

# 4) Save
itk.imwrite(vol, "recon_fdk.nii.gz")
```

---

## 3) Geometry quick patterns

```python
geom = itk.ThreeDCircularProjectionGeometry.New()

# Common parameters
SID = 1000.0  # source-to-isocenter distance [mm]
SDD = 1500.0  # source-to-detector distance [mm]

# Angle loop example
import math
n = 360
for i in range(n):
    ang_deg = i * (360.0 / n)
    geom.AddProjection(itk.D(ang_deg), itk.D(SID), itk.D(SDD))
    # Optional fine-tuning: detector/source offsets, pixel spacing, center of rotation shifts
    # geom.SetProjectionOffsetsX(i, 0.0); geom.SetProjectionOffsetsY(i, 0.0)
    # geom.SetSourceOffsetsX(i, 0.0);      geom.SetSourceOffsetsY(i, 0.0)
```

---

## 4) Readers

```python
# (A) Single file stack (MetaImage/NIfTI)
projs = itk.imread("projections.mha", itk.F)

# (B) Series folder (e.g., PNG/JPG/DICOM) -> stack
from glob import glob
files = sorted(glob("projs_dir/*.png"))  # or DICOM: *.dcm
projs = itk.imread(files, itk.F)  # uses ImageSeriesReader internally
```

> Tip: Ensure projections are in the correct order (angle sequence). For DICOM, check angle-related metadata and, if needed, implement explicit sorting.

---

## 5) Output volume region (important)

Correct FOV and scaling require consistent output Size/Spacing/Origin across FDK/Iterative.

```python
fdk.SetOutputSpacing([0.5, 0.5, 0.5])  # mm
fdk.SetOutputOrigin([-64.0, -64.0, -64.0])
fdk.SetOutputSize([256, 256, 256])
```

---

## 6) Iterative reconstruction (concept)

> Availability of classes varies by build/wrapping. Example pattern only.

```python
# Example: OS-SART + TV regularization (check availability)
# has_ossart = hasattr(itk, "OSSARTConeBeamReconstructionFilter")
# if has_ossart:
#     rec = itk.OSSARTConeBeamReconstructionFilter.New()
#     rec.SetInput(projs)
#     rec.SetGeometry(geom)
#     rec.SetNumberOfIterations(10)
#     rec.SetRelaxation(0.2)
#     rec.SetTVWeight(0.005)
#     # Same output volume metadata settings as needed
#     rec.Update()
#     vol = rec.GetOutput()
```

---

## 7) Essential pre/post (recommended)

```python
# (A) Flat-field/Dark-field correction when available
# projs_corr = flatdark_correction(projs, flat, dark)

# (B) Log transform (I0/I) -> linear attenuation
# projs_log = -itk.log(projs_corr + eps)

# (C) Light scatter correction / geometry tweaks (optional)
# projs_scatter = scatter_correction(projs_log, ...)

# (D) Clamp or filter final volume
# vol = itk.ClampImageFilter[type(vol), type(vol)].New(Input=vol, Bounds=[lo, hi]).GetOutput()
```

---

## 8) NumPy round-trip & quick view

```python
import numpy as np

arr = itk.array_from_image(vol)      # (z,y,x)
mid = arr.shape[0] // 2
mid_slice = arr[mid, :, :]

# Optional preview via matplotlib
# import matplotlib.pyplot as plt
# plt.imshow(mid_slice, cmap="gray"); plt.show()

# Back to ITK image
vol2 = itk.image_from_array(arr)
vol2.CopyInformation(vol)  # keep spacing/origin/direction
```

---

## 9) Troubleshooting checklist

- Missing classes? → `pip show itk-rtk` / check RTK Python wrapping build.
- White/noisy output? → mismatch in output FOV/spacing/origin/size vs geometry.
- Ghost/blur? → wrong angle order; SID/SDD; detector offset/spacing; rotation center; log transform issues.
- Performance/memory: adjust series loading, chunking, or sparsity.
- Environment: verify projection count/units; validate output region before computation.

---

## 10) Minimal examples

### (A) Minimal FDK (with simple geometry defaults)

```python
import itk

geom = itk.ThreeDCircularProjectionGeometry.New()
# for i, ang in enumerate(angles_deg):
#     geom.AddProjection(itk.D(ang), itk.D(SID), itk.D(SDD))

projs = itk.imread("projections.mha", itk.F)

fdk = itk.FDKConeBeamReconstructionFilter.New(Input=projs, Geometry=geom)
fdk.SetOutputSpacing([0.5,0.5,0.5])
fdk.SetOutputOrigin([-64,-64,-64])
fdk.SetOutputSize([256,256,256])
fdk.Update()
itk.imwrite(fdk.GetOutput(), "recon_fdk.nii.gz")
```

### (B) Load from series + log transform

```python
from glob import glob
import itk, numpy as np

files = sorted(glob("projs/*.png"))
projs = itk.imread(files, itk.F)

arr = itk.array_from_image(projs)
arr = -np.log(np.clip(arr, 1e-6, None))
projs_log = itk.image_from_array(arr); projs_log.CopyInformation(projs)

# then feed into FDK/Iterative
```

### (C) Save a mid slice from a reconstructed volume

```python
import itk, numpy as np

vol = itk.imread("recon_fdk.nii.gz", itk.F)
arr = itk.array_from_image(vol)
mid = arr.shape[0]//2
sl = arr[mid]
itk.imwrite(itk.image_from_array(sl.astype("float32")), "mid_slice.mha")
```

