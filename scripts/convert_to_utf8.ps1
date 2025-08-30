# Convert repository text files to UTF-8 (no BOM)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/convert_to_utf8.ps1 [-WhatIf]

param(
  [switch]$WhatIf
)

$extensions = @(
  '.md','.py','.txt','.json','.yml','.yaml','.ini','.cfg','.toml',
  '.gitignore','.gitattributes','.ps1','.bat','.sh','.csv'
)

function Is-TextFile($path) {
  $ext = [System.IO.Path]::GetExtension($path)
  return $extensions -contains $ext.ToLower()
}

$root = Get-Location
$files = Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object { Is-TextFile $_.FullName }

Write-Host "Converting $($files.Count) files to UTF-8 (no BOM)..."

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

foreach ($f in $files) {
  try {
    # Read with .NET default encoding (system code page). If decoding fails, skip.
    $text = [System.IO.File]::ReadAllText($f.FullName)
  } catch {
    Write-Warning "Skip (read failed): $($f.FullName)"
    continue
  }
  if ($WhatIf) {
    Write-Host "Would convert: $($f.FullName)"
  } else {
    try {
      [System.IO.File]::WriteAllText($f.FullName, $text, $utf8NoBom)
      Write-Host "Converted: $($f.FullName)"
    } catch {
      Write-Warning "Skip (write failed): $($f.FullName)"
    }
  }
}

Write-Host "Done."

