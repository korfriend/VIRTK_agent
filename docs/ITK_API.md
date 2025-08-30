# ITK Python – Quick Reference (80/20)

> 목적: **이미지 I/O, 기본 메타데이터 접근, 필터 체인, 변환** 등 자주 쓰는 API를 빠르게 참조.
> 전제: `pip install itk`

---

## 0) 기본 import

```python
import itk
```

---

## 1) 이미지 읽기/쓰기

```python
# 읽기
img = itk.imread("input.nii.gz", pixel_type=itk.F)   # itk.UC=uint8, itk.F=float32 등

# 쓰기
itk.imwrite(img, "output.mha")
```

---

## 2) 메타데이터 (size, spacing, origin, direction)

```python
sz = itk.size(img)         # (x,y,z)
sp = itk.spacing(img)      # voxel spacing
org = itk.origin(img)      # physical origin
dirM = itk.direction(img)  # 3x3 방향행렬
```

---

## 3) 타입 & 캐스팅

```python
# 명시적 이미지 타입 정의: (pixel, dimension)
Float3D = itk.Image[itk.F, 3]

# 캐스팅
Caster = itk.CastImageFilter[type(img), Float3D].New(Input=img)
img_f = Caster.GetOutput()
```

---

## 4) 자주 쓰는 필터

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

### Resample (해상도/spacing 변경)

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

## 5) 레지스트레이션 (Rigid, Simple)

```python
fixed = itk.imread("fixed.nii.gz", itk.F)
moving = itk.imread("moving.nii.gz", itk.F)

reg = itk.ImageRegistrationMethodv4[itk.Image[itk.F,3], itk.Image[itk.F,3]].New()
# Optimizer, Metric, Transform 설정 필요 (고급)
```

*(※ 여기서는 구조만, 실제 파라미터는 상황별 설정)*

---

## 6) 파이프라인 실행

```python
# 방식 1: .Update() 직접 호출
med.Update()
out = med.GetOutput()

# 방식 2: itk.pipeline() 헬퍼
out = itk.pipeline(med)
```

---

## 7) NumPy 변환

```python
import itk
import numpy as np

arr = itk.array_from_image(img)   # ITK → numpy (z,y,x 순)
img2 = itk.image_from_array(arr)  # numpy → ITK
```

---

## 8) Transforms (회전/이동)

```python
tf = itk.Euler3DTransform[itk.D].New()
tf.SetRotation(0.0, 0.0, 0.5)      # radian
tf.SetTranslation((10.0, 0.0, 0.0))

resamp = itk.ResampleImageFilter[type(img), type(img)].New(
    Input=img, Transform=tf, Size=sz, OutputSpacing=sp, OutputOrigin=org
)
out_tf = resamp.GetOutput()
```

---

## 9) Segmentation 예시: Otsu Threshold

```python
otsu = itk.OtsuThresholdImageFilter[type(img), itk.Image[itk.UC,3]].New(Input=img)
mask = otsu.GetOutput()
```

---

## 10) 팁

* ITK는 **템플릿 기반**이라 Python에서도 `Filter[InputType, OutputType]` 형태로 지정.
* 타입 불일치 오류가 자주 발생 → 필요하면 `CastImageFilter`로 맞추기.
* `itk.Image[pixel, dimension]` 로 항상 픽셀 타입/차원 명시 가능.
* 의료 이미지 다룰 때는 **spacing/origin/direction** 보존 주의.

---

## 11) 최소 예제 3종

### (A) Median 필터

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
mask.CopyInformation(img)   # spacing, origin 유지
itk.imwrite(mask, "mask.nii.gz")
```