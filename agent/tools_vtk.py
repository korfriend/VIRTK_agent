def vtk_render_snapshot(volume_path: str, out_png: str) -> dict:
    import vtk
    # Load via vtkImageReader or ITK→VTK bridge
    # Configure renderer/camera and take an offscreen snapshot
    ...
    return {"png": out_png}

