# “풀 API 리스트(색인)” 자동 생성 스크립트
# 설치된 환경에서 import가 가능한 **공개 심볼을 광범위하게 긁어서** Markdown 인덱스를 만듭니다.  
# - ITK: `dir(itk)`를 걸되, **이름에 ‘rtk’ 포함 시 RTK 색인으로도 분류**  
# - VTK: `vtk` 모듈의 `vtk*` 클래스/함수 패턴 수집  
# - RTK: 독립 모듈이 있다면(`import rtk`) 별도로도 색인 시도

import re, inspect, os, sys
from textwrap import shorten

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs_api_index")
os.makedirs(OUT_DIR, exist_ok=True)

def write_md(path, title, sections):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for sec_title, items in sections:
            f.write(f"## {sec_title}\n\n")
            for name, kind, where in items:
                f.write(f"- `{name}`  _{kind}_ ({where})\n")
            f.write("\n")

def collect_module_api(mod, name_filter=None, where_name=""):
    items = []
    for name in sorted(dir(mod)):
        if name_filter and not name_filter(name):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        kind = None
        if inspect.isclass(obj):
            kind = "class"
        elif inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
            kind = "func"
        elif inspect.ismodule(obj):
            kind = "module"
        else:
            # large wrapped types in ITK/VTK will often be “type” or “object”
            kind = type(obj).__name__
        items.append((name, kind, where_name))
    return items

# ---------- ITK ----------
itk_sections = []
try:
    import itk
    # 전체
    itk_all = collect_module_api(itk, where_name="itk")
    # rtk 키워드 포함(랩핑 환경에 따라 다름)
    rtk_like = [(n,k,w) if "rtk" in n.lower() else None for (n,k,w) in itk_all]
    rtk_like = [x for x in rtk_like if x]

    # 그룹핑 예: 대문자/소문자 패턴으로 roughly class/func 구분
    def by_kind(items, target): return [x for x in items if x[1] == target]
    itk_sections.append(("All (itk)", itk_all[:2000]))  # 너무 길어지는 것 방지용 슬라이스
    write_md(os.path.join(OUT_DIR, "itk_api_index.md"), "ITK Python API Index", itk_sections)

    if rtk_like:
        write_md(os.path.join(OUT_DIR, "rtk_api_index.md"), "RTK (from ITK) API Index",
                 [("Symbols containing 'rtk'", rtk_like)])
except Exception as e:
    print("[WARN] ITK not indexed:", e, file=sys.stderr)

# ---------- VTK ----------
vtk_sections = []
try:
    import vtk
    vtk_all = collect_module_api(vtk, name_filter=lambda s: s.startswith("vtk"))
    # 모듈별 자세한 분류(선택): vtkmodules 이용
    try:
        from vtkmodules import all as vtk_allmods
        vtk_mod_items = []
        for mname in dir(vtk_allmods):
            if mname.startswith("_"): continue
            try:
                m = getattr(vtk_allmods, mname)
                if inspect.ismodule(m):
                    vtk_mod_items.extend(collect_module_api(m, name_filter=lambda s: s.startswith("vtk"),
                                                           where_name=f"vtkmodules.{mname}"))
            except Exception:
                pass
        if vtk_mod_items:
            vtk_sections.append(("vtkmodules/*", vtk_mod_items[:2000]))
    except Exception:
        pass

    vtk_sections.insert(0, ("vtk top-level", vtk_all[:2000]))
    write_md(os.path.join(OUT_DIR, "vtk_api_index.md"), "VTK Python API Index", vtk_sections)
except Exception as e:
    print("[WARN] VTK not indexed:", e, file=sys.stderr)

print("Done. Wrote Markdown indexes to", OUT_DIR)
