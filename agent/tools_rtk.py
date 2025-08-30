def rtk_fdk(projections_dir: str, geometry_xml: str, out_mha: str,
            spacing=(1.0,1.0,1.0), size=None, crop_roi=None) -> dict:
    """
    returns: {"volume_path": out_mha, "spacing": [...], "size":[...]}
    구현 전략:
      1) RTK CLI 있으면 subprocess로 rtkfdktype 호출 (화이트리스트 경로만)
      2) 또는 ITK/RTK 파이프라인 바인딩이 가능한 환경이면 파이썬 API 사용
      3) 산출물 존재/헤더(스페이싱, 사이즈) 검증
    """
    ...
