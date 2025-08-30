"""
Generate lightweight Markdown API indexes for ITK/VTK (and RTK-like symbols).

This script scans importable public symbols and writes simple lists under docs_api_index/.
- ITK: enumerate dir(itk). Also classify names containing 'rtk' into an RTK index.
- VTK: collect names starting with 'vtk' from the top-level vtk module. Optionally index vtkmodules/*.
"""

import inspect
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs_api_index")
os.makedirs(OUT_DIR, exist_ok=True)


def write_md(path: str, title: str, sections):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for sec_title, items in sections:
            f.write(f"## {sec_title}\n\n")
            for name, kind, where in items:
                f.write(f"- `{name}`  _{kind}_ ({where})\n")
            f.write("\n")


def collect_module_api(mod, name_filter=None, where_name: str = ""):
    items = []
    for name in sorted(dir(mod)):
        if name_filter and not name_filter(name):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if inspect.isclass(obj):
            kind = "class"
        elif inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
            kind = "func"
        elif inspect.ismodule(obj):
            kind = "module"
        else:
            # Large wrapped types in ITK/VTK often appear as generic Type/Object
            kind = type(obj).__name__
        items.append((name, kind, where_name))
    return items


# ---------- ITK ----------
itk_sections = []
try:
    import itk
    # All
    itk_all = collect_module_api(itk, where_name="itk")
    # RTK-like entries
    rtk_like = [(n, k, w) for (n, k, w) in itk_all if "rtk" in n.lower()]

    itk_sections.append(("All (itk)", itk_all[:2000]))  # limit to keep files manageable
    write_md(os.path.join(OUT_DIR, "itk_api_index.md"), "ITK Python API Index", itk_sections)

    if rtk_like:
        write_md(
            os.path.join(OUT_DIR, "rtk_api_index.md"),
            "RTK (from ITK) API Index",
            [("Symbols containing 'rtk'", rtk_like)],
        )
except Exception as e:
    print("[WARN] ITK not indexed:", e, file=sys.stderr)


# ---------- VTK ----------
vtk_sections = []
try:
    import vtk
    vtk_all = collect_module_api(vtk, name_filter=lambda s: s.startswith("vtk"), where_name="vtk")
    # Optional per-module details: use vtkmodules if available
    try:
        from vtkmodules import all as vtk_allmods

        vtk_mod_items = []
        for mname in dir(vtk_allmods):
            if mname.startswith("_"):
                continue
            try:
                m = getattr(vtk_allmods, mname)
                if inspect.ismodule(m):
                    vtk_mod_items.extend(
                        collect_module_api(
                            m,
                            name_filter=lambda s: s.startswith("vtk"),
                            where_name=f"vtkmodules.{mname}",
                        )
                    )
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

