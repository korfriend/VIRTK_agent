def vtk_render_snapshot(volume_path: str, out_png: str) -> dict:
    import vtk
    # vtkImageReader / ITK→VTK 브리지 등으로 로드
    # 렌더러/카메라 세팅 후 offscreen snapshot 저장
    ...
    return {"png": out_png}
