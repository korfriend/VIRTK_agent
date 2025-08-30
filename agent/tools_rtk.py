def rtk_fdk(projections_dir: str, geometry_xml: str, out_mha: str,
            spacing=(1.0, 1.0, 1.0), size=None, crop_roi=None) -> dict:
    """
    Reconstruct a CBCT volume from projections using RTK.

    Returns: {"volume_path": out_mha, "spacing": [...], "size": [...]}.

    Implementation options:
      1) If RTK CLI is available: invoke FDK via subprocess (whitelisted paths only).
      2) Otherwise: use ITK/RTK Python APIs if bindings are available.
      3) Validate outputs (existence) and headers (spacing, size).
    """
    ...

