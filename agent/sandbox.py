import pathlib

# 허용할 폴더들 (필요에 따라 조정)
WHITELIST_DIRS = [
    "./data",
    "./outputs",
    "./tmp"
]

# 허용할 확장자들
ALLOWED_EXTS = {".mha", ".nii", ".nrrd", ".png", ".jpg", ".tif"}


def guard_path(path: str) -> str:
    """
    path가 허용된 디렉토리와 확장자인지 확인.
    """
    p = pathlib.Path(path).resolve()

    # 1. 화이트리스트 경로 검사
    allowed = False
    for base in WHITELIST_DIRS:
        if str(p).startswith(str(pathlib.Path(base).resolve())):
            allowed = True
            break
    if not allowed:
        raise PermissionError(f"Access to {p} is not allowed (not in whitelist)")

    # 2. 확장자 검사
    if p.suffix and p.suffix.lower() not in ALLOWED_EXTS:
        raise PermissionError(f"Extension {p.suffix} is not allowed")

    return str(p)


def guard_args(args: dict) -> dict:
    """
    dict 안의 값 중 파일 경로가 있으면 검사 후 안전한 경로만 반환.
    """
    checked = {}
    for k, v in args.items():
        if isinstance(v, str) and ("/" in v or "\\" in v):  # 경로일 가능성
            checked[k] = guard_path(v)
        else:
            checked[k] = v
    return checked
