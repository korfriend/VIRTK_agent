import pathlib

# Allowed directories (adjust as needed)
WHITELIST_DIRS = [
    "./data",
    "./outputs",
    "./tmp"
]

# Allowed file extensions
ALLOWED_EXTS = {".mha", ".nii", ".nrrd", ".png", ".jpg", ".tif"}


def guard_path(path: str) -> str:
    """Ensure the path is within allowed directories and has an allowed extension."""
    p = pathlib.Path(path).resolve()

    # 1) Check whitelist directories
    allowed = False
    for base in WHITELIST_DIRS:
        if str(p).startswith(str(pathlib.Path(base).resolve())):
            allowed = True
            break
    if not allowed:
        raise PermissionError(f"Access to {p} is not allowed (not in whitelist)")

    # 2) Check file extension
    if p.suffix and p.suffix.lower() not in ALLOWED_EXTS:
        raise PermissionError(f"Extension {p.suffix} is not allowed")

    return str(p)


def guard_args(args: dict) -> dict:
    """Validate any file path values in a dict and return safe absolute paths."""
    checked = {}
    for k, v in args.items():
        if isinstance(v, str) and ("/" in v or "\\" in v):  # likely a path
            checked[k] = guard_path(v)
        else:
            checked[k] = v
    return checked



