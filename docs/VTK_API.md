# VTK Python Quick Reference (80/20)

> Goal: One place to recall mesh/image I/O, visualization pipelines, and common filters/transforms. Prereq: `pip install vtk`.

---

## 0) Basic snippet

```python
import vtk
```

---

## 1) Core data structures

- vtkPolyData: surfaces (points/lines/triangles)
- vtkImageData: regular grids (scalars/volumes)
- vtkUnstructuredGrid: unstructured cells (tetra, etc.)
- vtkTable: table (column-based)

---

## 2) Mesh I/O (STL/OBJ/PLY)

```python
# Read STL
r = vtk.vtkSTLReader(); r.SetFileName("mesh.stl"); r.Update()
poly = r.GetOutput()

# Write STL
w = vtk.vtkSTLWriter(); w.SetFileName("out.stl"); w.SetInputData(poly); w.Write()
```

---

## 3) Image/Volume I/O (DICOM/PNG/NIfTI)

```python
# DICOM (directory)
dr = vtk.vtkDICOMImageReader(); dr.SetDirectoryName("dicom_dir"); dr.Update()
vol = dr.GetOutput()

# PNG/JPG (2D)
pr = vtk.vtkPNGReader(); pr.SetFileName("img.png"); pr.Update()
img2d = pr.GetOutput()

# Write PNG (2D)
pw = vtk.vtkPNGWriter(); pw.SetFileName("out.png"); pw.SetInputData(img2d); pw.Write()
```

---

## 4) Visualization pipeline (PolyData)

```python
mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
actor = vtk.vtkActor(); actor.SetMapper(mapper)
ren = vtk.vtkRenderer(); ren.AddActor(actor)
win = vtk.vtkRenderWindow(); win.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)
win.Render(); iren.Start()
```

### Appearance

```python
actor.GetProperty().SetColor(1,0.8,0.2)
actor.GetProperty().SetLineWidth(2)
```

---

## 5) Common mesh filters

```python
# Recompute normals
norm = vtk.vtkPolyDataNormals(); norm.SetInputData(poly); norm.AutoOrientNormalsOn(); norm.Update()
poly_n = norm.GetOutput()

# Decimate
dec = vtk.vtkDecimatePro(); dec.SetInputData(poly); dec.SetTargetReduction(0.5); dec.Update()
poly_d = dec.GetOutput()

# Smooth
sm = vtk.vtkSmoothPolyDataFilter(); sm.SetInputData(poly); sm.SetNumberOfIterations(20); sm.Update()
poly_s = sm.GetOutput()
```

---

## 6) Marching Cubes → isosurface

```python
mc = vtk.vtkMarchingCubes(); mc.SetInputData(vol); mc.SetValue(0, 500); mc.Update()
iso = mc.GetOutput()
```

---

## 7) Volume rendering (direct)

```python
ctf = vtk.vtkColorTransferFunction();  # color TF
otf = vtk.vtkPiecewiseFunction();      # opacity TF

mapper = vtk.vtkGPUVolumeRayCastMapper(); mapper.SetInputData(vol)
prop = vtk.vtkVolumeProperty(); prop.SetColor(ctf); prop.SetScalarOpacity(otf)

vol_actor = vtk.vtkVolume(); vol_actor.SetMapper(mapper); vol_actor.SetProperty(prop)
ren = vtk.vtkRenderer(); ren.AddViewProp(vol_actor)
win = vtk.vtkRenderWindow(); win.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)
win.Render(); iren.Start()
```

---

## 8) Camera & light

```python
cam = ren.GetActiveCamera(); cam.SetPosition(0,0,100); cam.SetFocalPoint(0,0,0)
light = vtk.vtkLight(); light.SetLightTypeToSceneLight(); ren.AddLight(light)
```

---

## 9) Interaction/Picking

```python
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

picker = vtk.vtkPropPicker()
# Usage: picker.Pick(x, y, 0, ren); pos = picker.GetPickPosition()
```

---

## 10) Screenshots / Offscreen render

```python
# Screenshot
w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
pw = vtk.vtkPNGWriter(); pw.SetFileName("shot.png"); pw.SetInputConnection(w2i.GetOutputPort()); pw.Write()

# Offscreen
win.OffScreenRenderingOn(); win.Render()
```

---

## 11) NumPy interop (fast data ops)

```python
import numpy as np
from vtk.util import numpy_support as ns

arr = ns.vtk_to_numpy(vol.GetPointData().GetScalars())
arr_np = arr.copy()
arr_np = arr_np.reshape(vol.GetDimensions()[::-1])     # e.g., reshape to (z,y,x)

img2 = vtk.vtkImageData(); img2.DeepCopy(vol)          # copy metadata
img2.GetPointData().SetScalars(ns.numpy_to_vtk(arr_np.ravel(order="C")))
```

---

## 12) Point clouds & glyphs

```python
points = vtk.vtkPoints()
for x,y,z in [(0,0,0),(10,0,0),(0,10,0)]: points.InsertNextPoint(x,y,z)
poly = vtk.vtkPolyData(); poly.SetPoints(points)
glyph = vtk.vtkGlyph3D()
```

---

## 13) Pipeline tips

* Call `Update()` when outputs are needed.
* Use `SetInputData` for already-computed data, `SetInputConnection` for pipeline ports.
* Performance: prefer `vtkGPUVolumeRayCastMapper` for volume rendering; adjust scalar ranges and sampling.
* Coordinates/spacing: set `SetSpacing`, `SetOrigin`; watch DICOM direction matrices.

---

## 14) Mini troubleshooting

* No window: check backend/remote environment (use offscreen).
* No vertex colors for PLY/OBJ: ensure `vtkUnsignedCharArray` named "Colors" is present.
* Wrong normal orientation: `vtkPolyDataNormals().AutoOrientNormalsOn()`.

---

## 15) Minimal examples

### (A) Load & render STL

```python
import vtk
r = vtk.vtkSTLReader(); r.SetFileName("model.stl"); r.Update()
poly = r.GetOutput()
mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
actor = vtk.vtkActor(); actor.SetMapper(mapper)
ren = vtk.vtkRenderer(); ren.AddActor(actor)
win = vtk.vtkRenderWindow(); win.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)
win.Render(); iren.Start()
```

### (B) Volume render

```python
# see section 7
```

### (C) Marching Cubes → isosurface

```python
import vtk
mc = vtk.vtkMarchingCubes(); mc.SetInputData(vol); mc.SetValue(0, 500); mc.Update()
iso = mc.GetOutput()
```

---

## Appendix: common class names

* Rendering: `vtkRenderer`, `vtkRenderWindow`, `vtkRenderWindowInteractor`, `vtkActor`, `vtkPolyDataMapper`, `vtkVolume`, `vtkGPUVolumeRayCastMapper`, `vtkVolumeProperty`
* Filters: `vtkPolyDataNormals`, `vtkDecimatePro`, `vtkSmoothPolyDataFilter`, `vtkTransformPolyDataFilter`, `vtkClipPolyData`, `vtkConnectivityFilter`, `vtkMarchingCubes`
* Utils: `vtkWindowToImageFilter`, `vtkColorTransferFunction`, `vtkPiecewiseFunction`, `vtkInteractorStyleTrackballCamera`, `vtkPropPicker`

