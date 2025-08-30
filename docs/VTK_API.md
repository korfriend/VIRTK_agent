# VTK Python – Quick Reference (80/20)

> 목적: **메시/이미지 I/O, 시각화 파이프라인, 자주 쓰는 필터/트랜스폼**을 한 번에 참고
> 전제: `pip install vtk` (또는 환경에 맞는 설치)

## 0) 기본 스니펫

```python
import vtk

# Renderer/Window/Interactor
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow(); renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(renWin)
```

---

## 1) 데이터 구조 핵심

* **vtkPolyData**: 점/선/삼각형 등 표면 메시
* **vtkImageData**: 정규 격자(스칼라/볼륨)
* **vtkUnstructuredGrid**: 비정형 셀(테트라 등)
* **vtkTable**: 테이블(열 기반)

---

## 2) 메시 I/O (STL/OBJ/PLY 등)

```python
# 읽기
stl = vtk.vtkSTLReader(); stl.SetFileName("mesh.stl"); stl.Update()
poly = stl.GetOutput()  # vtkPolyData

obj = vtk.vtkOBJReader(); obj.SetFileName("mesh.obj"); obj.Update()
poly2 = obj.GetOutput()

# 쓰기
plyw = vtk.vtkPLYWriter(); plyw.SetFileName("out.ply"); plyw.SetInputData(poly); plyw.Write()
stlw = vtk.vtkSTLWriter(); stlw.SetFileName("out.stl"); stlw.SetInputData(poly2); stlw.Write()
```

---

## 3) 이미지/볼륨 I/O (DICOM/PNG/NIfTI 등)

```python
# DICOM 읽기 (디렉터리)
dcm = vtk.vtkDICOMImageReader()
dcm.SetDirectoryName("DICOM_DIR"); dcm.Update()
img = dcm.GetOutput()  # vtkImageData

# PNG/JPG 읽기
png = vtk.vtkPNGReader(); png.SetFileName("slice.png"); png.Update()
img2 = png.GetOutput()

# MetaImage(.mhd/.raw)
mha = vtk.vtkMetaImageReader(); mha.SetFileName("vol.mha"); mha.Update()
vol = mha.GetOutput()

# NIfTI
nii = vtk.vtkNIFTIImageReader(); nii.SetFileName("vol.nii.gz"); nii.Update()
vol2 = nii.GetOutput()

# PNG 쓰기 (2D)
pw = vtk.vtkPNGWriter(); pw.SetFileName("out.png"); pw.SetInputData(img2); pw.Write()

# MetaImage 쓰기 (3D)
mw = vtk.vtkMetaImageWriter(); mw.SetFileName("out.mha"); mw.SetInputData(vol); mw.Write()
```

---

## 4) 시각화 파이프라인 (PolyData)

```python
mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
actor = vtk.vtkActor(); actor.SetMapper(mapper)
ren.AddActor(actor); ren.ResetCamera()
renWin.Render(); iren.Start()
```

### 재질/색/선두께

```python
actor.GetProperty().SetColor(1, 0.8, 0.2)
actor.GetProperty().SetOpacity(0.9)
actor.GetProperty().SetLineWidth(2.0)
```

---

## 5) 자주 쓰는 메시 필터

```python
# 노멀 재계산
norm = vtk.vtkPolyDataNormals(); norm.SetInputData(poly)
norm.SetFeatureAngle(60); norm.ConsistencyOn(); norm.AutoOrientNormalsOn()
norm.Update(); poly_n = norm.GetOutput()

# 디케메이션(간략화)
dec = vtk.vtkDecimatePro(); dec.SetInputData(poly)
dec.SetTargetReduction(0.5)  # 0.5 → 50% 감소
dec.PreserveTopologyOn(); dec.Update()
poly_dec = dec.GetOutput()

# 스무딩
smo = vtk.vtkSmoothPolyDataFilter(); smo.SetInputData(poly)
smo.SetNumberOfIterations(30); smo.SetRelaxationFactor(0.1); smo.FeatureEdgeSmoothingOff()
smo.Update(); poly_smooth = smo.GetOutput()

# 트랜스폼(이동/회전/스케일)
tf = vtk.vtkTransform(); tf.Translate(10,0,0); tf.RotateY(30); tf.Scale(1.2,1,1)
tfF = vtk.vtkTransformPolyDataFilter(); tfF.SetTransform(tf); tfF.SetInputData(poly); tfF.Update()
poly_tf = tfF.GetOutput()

# 클리핑(평면)
plane = vtk.vtkPlane(); plane.SetOrigin(0,0,0); plane.SetNormal(1,0,0)
clip = vtk.vtkClipPolyData(); clip.SetInputData(poly); clip.SetClipFunction(plane); clip.Update()
poly_clip = clip.GetOutput()

# 분리(connected components)
cc = vtk.vtkConnectivityFilter(); cc.SetInputData(poly); cc.SetExtractionModeToAllRegions()
cc.ColorRegionsOn(); cc.Update(); poly_cc = cc.GetOutput()
```

---

## 6) 이미지/볼륨 처리 & 등가면

```python
# Marching Cubes: vtkImageData → 등가면(vtkPolyData)
mc = vtk.vtkMarchingCubes()
mc.SetInputData(vol)          # vtkImageData
mc.SetValue(0, 500)           # 등가면 값
mc.Update()
iso = mc.GetOutput()
```

---

## 7) 볼륨 렌더링(직접)

```python
volMapper = vtk.vtkGPUVolumeRayCastMapper()
volMapper.SetInputData(vol)

# 색/불투명도 전이함수
ctf = vtk.vtkColorTransferFunction(); ctf.AddRGBPoint(-1000, 0.0,0.0,0.0); ctf.AddRGBPoint(400,1.0,1.0,1.0)
otf = vtk.vtkPiecewiseFunction();     otf.AddPoint(-1000, 0.0);          otf.AddPoint(400, 0.8)

volProp = vtk.vtkVolumeProperty()
volProp.SetColor(ctf); volProp.SetScalarOpacity(otf)
volProp.SetInterpolationTypeToLinear(); volProp.ShadeOn()

volume = vtk.vtkVolume()
volume.SetMapper(volMapper); volume.SetProperty(volProp)
ren.AddVolume(volume); ren.ResetCamera()
renWin.Render(); iren.Start()
```

---

## 8) 카메라 & 라이트

```python
cam = ren.GetActiveCamera()
cam.SetPosition(0, -500, 200); cam.SetFocalPoint(0,0,0); cam.SetViewUp(0,0,1)
ren.ResetCameraClippingRange()

light = vtk.vtkLight(); light.SetPosition(200,200,300); light.SetFocalPoint(0,0,0)
ren.AddLight(light)
```

---

## 9) 인터랙션/피킹

```python
# 트랙볼 스타일
style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)

# 피킹(마우스 위치에서 3D 점/배우 선택)
picker = vtk.vtkPropPicker()
iren.SetPicker(picker)
# 사용 시: picker.Pick(x, y, 0, ren); pos = picker.GetPickPosition()
```

---

## 10) 스크린샷/오프스크린 렌더

```python
# 스크린샷
w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(renWin); w2i.Update()
pngw = vtk.vtkPNGWriter(); pngw.SetFileName("screenshot.png"); pngw.SetInputConnection(w2i.GetOutputPort()); pngw.Write()

# 오프스크린
renWin.OffScreenRenderingOn()
renWin.Render()
w2i.Update(); pngw.Write()
```

---

## 11) NumPy 연동 (→ 빠른 데이터 조작)

```python
import numpy as np
from vtk.util import numpy_support as nps

# vtkImageData → numpy
arr_vtk = vol.GetPointData().GetScalars()              # vtkDataArray
arr_np = nps.vtk_to_numpy(arr_vtk)                     # 1D
arr_np = arr_np.reshape(vol.GetDimensions()[::-1])     # (z,y,x) 등으로 reshape

# numpy → vtkImageData
arr2 = (arr_np * 0.5).astype(np.float32).ravel(order='C')
arr2_vtk = nps.numpy_to_vtk(arr2, deep=True, array_type=vtk.VTK_FLOAT)
img2 = vtk.vtkImageData(); img2.DeepCopy(vol)          # 메타 복사 후
img2.GetPointData().SetScalars(arr2_vtk)
```

---

## 12) 포인트 클라우드 & 글리프

```python
# XYZ numpy → vtkPolyData(Points + Vertices)
pts_np = ...  # (N,3)
pts = vtk.vtkPoints(); pts.SetData(nps.numpy_to_vtk(pts_np, deep=True))
verts = vtk.vtkCellArray()
for i in range(pts_np.shape[0]):
    verts.InsertNextCell(1); verts.InsertCellPoint(i)
cloud = vtk.vtkPolyData(); cloud.SetPoints(pts); cloud.SetVerts(verts)

# 구 글리프로 시각화
sphere = vtk.vtkSphereSource(); sphere.SetRadius(1.0); sphere.Update()
glyph = vtk.vtkGlyph3D()
glyph.SetSourceConnection(sphere.GetOutputPort())
glyph.SetInputData(cloud); glyph.ScalingOff(); glyph.Update()
mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(glyph.GetOutputPort())
actor = vtk.vtkActor(); actor.SetMapper(mapper); ren.AddActor(actor)
```

---

## 13) 파이프라인 팁

* **Update() 호출**: 필터 결과가 필요할 때 잊지 말 것.
* **SetInputData vs SetInputConnection**:

  * 이미 계산 완료된 객체 → `SetInputData`
  * 다른 필터의 출력 포트 → `SetInputConnection(other.GetOutputPort())`
* **성능**: 볼륨 렌더링은 `vtkGPUVolumeRayCastMapper` 권장, 스칼라 범위/샘플링 줄이며 최적화.
* **좌표계/스페이싱**: 의료 이미지 사용 시 `SetSpacing`, `SetOrigin`, DICOM 방향행렬 주의.

---

## 14) 미니 트러블슈팅

* 창이 안 뜰 때: 백엔드/리모트 환경 확인(오프스크린 사용).
* PLY/OBJ에 색이 안 보일 때: 컬러 어트리뷰트 존재 여부(`vtkUnsignedCharArray` + “Colors”) 확인.
* 노멀 방향 이상: `vtkPolyDataNormals().AutoOrientNormalsOn()` 사용.

---

## 15) 최소 예제 3종

### (A) STL 로드 & 렌더

```python
import vtk
ren=vtk.vtkRenderer(); rw=vtk.vtkRenderWindow(); rw.AddRenderer(ren)
iren=vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(rw)

r=vtk.vtkSTLReader(); r.SetFileName("mesh.stl"); r.Update()
m=vtk.vtkPolyDataMapper(); m.SetInputData(r.GetOutput())
a=vtk.vtkActor(); a.SetMapper(m); ren.AddActor(a)
ren.ResetCamera(); rw.Render(); iren.Start()
```

### (B) 볼륨 렌더

```python
import vtk
ren=vtk.vtkRenderer(); rw=vtk.vtkRenderWindow(); rw.AddRenderer(ren)
iren=vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(rw)

rd=vtk.vtkMetaImageReader(); rd.SetFileName("vol.mha"); rd.Update()
mp=vtk.vtkGPUVolumeRayCastMapper(); mp.SetInputData(rd.GetOutput())
ctf=vtk.vtkColorTransferFunction(); ctf.AddRGBPoint(-1000,0,0,0); ctf.AddRGBPoint(400,1,1,1)
otf=vtk.vtkPiecewiseFunction(); otf.AddPoint(-1000,0.0); otf.AddPoint(400,0.8)
vp=vtk.vtkVolumeProperty(); vp.SetColor(ctf); vp.SetScalarOpacity(otf); vp.ShadeOn()
v=vtk.vtkVolume(); v.SetMapper(mp); v.SetProperty(vp); ren.AddVolume(v)
ren.ResetCamera(); rw.Render(); iren.Start()
```

### (C) Marching Cubes → 등가면

```python
import vtk
rd=vtk.vtkNIFTIImageReader(); rd.SetFileName("vol.nii.gz"); rd.Update()
mc=vtk.vtkMarchingCubes(); mc.SetInputData(rd.GetOutput()); mc.SetValue(0, 300); mc.Update()
m=vtk.vtkPolyDataMapper(); m.SetInputData(mc.GetOutput())
a=vtk.vtkActor(); a.SetMapper(m)
ren=vtk.vtkRenderer(); ren.AddActor(a); ren.ResetCamera()
rw=vtk.vtkRenderWindow(); rw.AddRenderer(ren)
iren=vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(rw)
rw.Render(); iren.Start()
```

---

### 부록) 자주 까먹는 클래스 이름

* 렌더링: `vtkRenderer`, `vtkRenderWindow`, `vtkRenderWindowInteractor`, `vtkActor`, `vtkPolyDataMapper`, `vtkVolume`, `vtkGPUVolumeRayCastMapper`, `vtkVolumeProperty`
* I/O: `vtkSTLReader/Writer`, `vtkOBJReader`, `vtkPLYWriter`, `vtkPNGReader/Writer`, `vtkDICOMImageReader`, `vtkMetaImageReader/Writer`, `vtkNIFTIImageReader`
* 필터: `vtkPolyDataNormals`, `vtkDecimatePro`, `vtkSmoothPolyDataFilter`, `vtkTransformPolyDataFilter`, `vtkClipPolyData`, `vtkConnectivityFilter`, `vtkMarchingCubes`
* 유틸: `vtkWindowToImageFilter`, `vtkColorTransferFunction`, `vtkPiecewiseFunction`, `vtkInteractorStyleTrackballCamera`, `vtkPropPicker`
