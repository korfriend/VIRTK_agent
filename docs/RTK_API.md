# RTK (Radiotherapy ToolKit) – Python Quick Reference (80/20)

> 목적: **CBCT 투영 기하 생성 → 투영 로드 → FDK/Iterative 재구성 → 저장**까지 바로 실행 가능하도록 최소 핵심만 정리
> 권장 설치:
>
> ```bash
> pip install itk itk-rtk
> ```
>
> 설치 후 RTK 클래스들은 보통 `import itk` 네임스페이스에 노출됩니다. (일부 환경에선 `import rtk` 별도 모듈도 가능)

---

## 0) 임포트 & 가용성 체크

```python
import itk

# RTK 심볼이 정상 래핑되었는지 간단 확인
has_geom = hasattr(itk, "ThreeDCircularProjectionGeometry")
has_fdk  = hasattr(itk, "FDKConeBeamReconstructionFilter")
print("RTK available:", has_geom and has_fdk)
```

---

## 1) 핵심 개념

* **ThreeDCircularProjectionGeometry**: 소스–디텍터 원궤도(C-arm) 투영 기하(각도, SID/SDD, 오프셋 등)
* **ProjectionsReader**: 투영 이미지(시리즈) 로딩 유틸(환경에 따라 ImageSeriesReader 활용)
* **FDKConeBeamReconstructionFilter**: 고전적 FDK(cone-beam) 재구성 필터
* **Iterative 알고리즘**: OSEM/OS-SART/TV regularization 등(빌드/랩핑 옵션에 따라 노출 가변)

---

## 2) 전형적 FDK 파이프라인 (스켈레톤)

```python
import itk

# 1) 투영 기하 구성
geom = itk.ThreeDCircularProjectionGeometry.New()

# 예: 각도/거리 설정 (실제 값으로 대체)
# for i, ang_deg in enumerate(angles_deg):
#     geom.AddProjection(itk.D(ang_deg), itk.D(SID), itk.D(SDD))
#     geom.SetSourceOffsetsX(i, src_offset_x); geom.SetSourceOffsetsY(i, src_offset_y)
#     geom.SetProjectionOffsetsX(i, det_offset_x); geom.SetProjectionOffsetsY(i, det_offset_y)

# 2) 투영 로드 (단일 스택 mha/nii 또는 시리즈)
projs = itk.imread("projections.mha", pixel_type=itk.F)  # (nProj, rows, cols) 또는 (z,y,x) 형태

# 3) FDK 재구성
fdk = itk.FDKConeBeamReconstructionFilter.New()
fdk.SetInput(projs)
fdk.SetGeometry(geom)

# 출력 볼륨 메타 설정(필요 시)
# fdk.SetOutputOrigin([ox, oy, oz])
# fdk.SetOutputSpacing([sx, sy, sz])
# fdk.SetOutputSize([nx, ny, nz])

fdk.Update()
vol = fdk.GetOutput()

# 4) 저장
itk.imwrite(vol, "recon_fdk.nii.gz")
```

---

## 3) 투영 기하(Geometry) 빠른 설정 패턴

```python
geom = itk.ThreeDCircularProjectionGeometry.New()

# 공통 파라미터
SID = 1000.0  # source-to-isocenter distance [mm]
SDD = 1500.0  # source-to-detector distance [mm]

# 각도 루프 예시
import math
n = 360
for i in range(n):
    ang_deg = i * (360.0 / n)
    geom.AddProjection(itk.D(ang_deg), itk.D(SID), itk.D(SDD))
    # 필요 시 디텍터/소스 오프셋, 픽셀 간격, 회전 중심 이동 등 세부 조정:
    # geom.SetProjectionOffsetsX(i, 0.0); geom.SetProjectionOffsetsY(i, 0.0)
    # geom.SetSourceOffsetsX(i, 0.0);      geom.SetSourceOffsetsY(i, 0.0)
```

---

## 4) 투영 읽기(Readers)

```python
# (A) 단일 파일 스택(MetaImage/NIfTI 등)
projs = itk.imread("projections.mha", itk.F)

# (B) 시리즈 폴더(예: PNG/JPG/DICOM) -> 스택
from glob import glob
files = sorted(glob("projs_dir/*.png"))  # 또는 DICOM: *.dcm
projs = itk.imread(files, itk.F)  # ImageSeriesReader 경로 자동 사용
```

> 팁: **정확한 투영 순서**(각도 순) 보장 위해 파일명/정렬 규칙을 명확히 하세요.
> DICOM의 경우 angle metadata가 있는지 확인하고, 필요 시 별도 정렬 로직을 준비합니다.

---

## 5) 출력 볼륨 영역 정의 (중요)

FDK/Iterative 모두 **출력 볼륨의 사이즈/스페이싱/오리진**을 적절히 지정해야 올바른 FOV와 스케일을 얻습니다.

```python
fdk.SetOutputSpacing([0.5, 0.5, 0.5])  # mm
fdk.SetOutputOrigin([-64.0, -64.0, -64.0])
fdk.SetOutputSize([256, 256, 256])
```

---

## 6) Iterative 재구성(개념형)

> 실제 클래스명은 빌드/래핑 옵션에 따라 다를 수 있습니다. 아래는 **패턴** 예시입니다.

```python
# 예시: OS-SART + TV regularization (환경에 있는지 확인 필요)
# has_ossart = hasattr(itk, "OSSARTConeBeamReconstructionFilter")
# if has_ossart:
#     rec = itk.OSSARTConeBeamReconstructionFilter.New()
#     rec.SetInput(projs)
#     rec.SetGeometry(geom)
#     rec.SetNumberOfIterations(10)
#     rec.SetRelaxation(0.2)
#     rec.SetTVWeight(0.005)     # Total Variation 가중치
#     # 출력 볼륨 메타 설정 동일
#     rec.Update()
#     vol = rec.GetOutput()
```

---

## 7) 필수 전·후처리 (권장)

```python
# (A) 플랫필드/다크필드 보정(있는 경우)
# projs_corr = flatdark_correction(projs, flat, dark)

# (B) 로그 변환 (I0/I) -> 선형 감쇠
# projs_log = -itk.log(projs_corr + eps)

# (C) 소규모 scatter 보정/기하 미스매치 보정(옵션)
# projs_scatter = scatter_correction(projs_log, ...)

# (D) 결과 볼륨 클리핑/필터링
# vol = itk.ClampImageFilter[type(vol), type(vol)].New(Input=vol, Bounds=[lo, hi]).GetOutput()
```

---

## 8) NumPy 왕복 & 시각화 샘플

```python
import numpy as np

arr = itk.array_from_image(vol)      # (z,y,x)
mid = arr.shape[0] // 2
mid_slice = arr[mid, :, :]

# (선택) matplotlib로 미리보기
# import matplotlib.pyplot as plt
# plt.imshow(mid_slice, cmap="gray"); plt.show()

# 다시 ITK 이미지로
vol2 = itk.image_from_array(arr)
vol2.CopyInformation(vol)  # spacing/origin/direction 복사
```

---

## 9) 트러블슈팅 체크리스트

* **클래스가 없다?** → `pip show itk-rtk` / RTK 파이썬 래핑 빌드 여부 확인.
* **흰 화면/노이즈** → 출력 영역(FOV)·spacing·origin·size가 실제 기하와 불일치 가능성.
* **유령/블러** → 각도 순서, SID/SDD, detector offset/spacing, 중심정렬, 로그 변환 여부 재점검.
* **밴딩/링 아티팩트** → 게인 보정, 평탄화, 시노그램 도메인 보정 고려.
* **메모리 부족** → 투영 크기/개수, 출력 볼륨 해상도 축소로 테스트.

---

## 10) 미니 실전 예제 3종

### (A) 최소 FDK (기하 값이 이미 알 때)

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

### (B) 시리즈 로드 + 로그 변환 스케치

```python
from glob import glob
import itk, numpy as np

files = sorted(glob("projs/*.png"))
projs = itk.imread(files, itk.F)

arr = itk.array_from_image(projs)
arr = -np.log(np.clip(arr, 1e-6, None))   # 간이 로그 변환
projs_log = itk.image_from_array(arr); projs_log.CopyInformation(projs)

# 이후 FDK/Iterative 투입
```

### (C) 결과 볼륨 가운데 슬라이스 바로 저장

```python
import itk, numpy as np

vol = itk.imread("recon_fdk.nii.gz", itk.F)
arr = itk.array_from_image(vol)
mid = arr.shape[0]//2
sl = arr[mid]
itk.imwrite(itk.image_from_array(sl.astype("float32")), "mid_slice.mha")
```

---

## 11) 에이전트 프롬프트 팁

* “**입력: 투영 묶음, 기하 파라미터 → 출력: 볼륨**” 식으로 목표를 명확히.
* 파일명 패턴, 각도 배열, SID/SDD, 픽셀 간격을 **숫자**로 제공하면 코드 안정성이 높아집니다.
* 실패 시: *출력 해상도/영역 먼저 줄여* 빠르게 디버깅 → 이후 정확 파라미터 반영.
