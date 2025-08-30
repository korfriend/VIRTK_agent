param(
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'

Write-Host "PowerShell version:" $PSVersionTable.PSVersion
Write-Host "Profile path:" $PROFILE

$profileDir = Split-Path -Parent $PROFILE
if (-not (Test-Path $profileDir)) {
    Write-Host "Creating profile directory:" $profileDir
    New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
}

$desired = @'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
'@

$needWrite = $true
if (Test-Path $PROFILE) {
    $existing = Get-Content -Raw -Encoding UTF8 -Path $PROFILE
    if ($existing -like "*OutputEncoding*UTF8*" -and $existing -like "*InputEncoding*UTF8*") {
        $needWrite = $false
    }
}

if ($Apply) {
    if ($needWrite) {
        Write-Host "Appending UTF-8 settings to profile..."
        Add-Content -Encoding UTF8 -Path $PROFILE -Value "`n# UTF-8 defaults (added by ps_profile_setup.ps1)" 
        Add-Content -Encoding UTF8 -Path $PROFILE -Value $desired
        Write-Host "Done. Open a new PowerShell window to load the profile."
    } else {
        Write-Host "Profile already contains UTF-8 settings. No changes made."
    }
} else {
    Write-Host "Preview only. Use -Apply to write the following to your profile:"
    Write-Host "-----"
    Write-Host $desired
    Write-Host "-----"
}

# Guidance for '&&' behavior
Write-Host "\n'&&' support:"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    Write-Host "This session supports '&&' (PowerShell 7+)."
} else {
    Write-Host "PowerShell 5.1 doesn't support '&&'. Use: cmd1; if ($?) { cmd2 }"
}

# Hint to install PowerShell 7 (no install performed here)
if (-not (Get-Command pwsh -ErrorAction SilentlyContinue)) {
    Write-Host "\nPowerShell 7 (pwsh) not found. To install (manual):"
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host " winget install --id Microsoft.PowerShell -e"
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host " choco install powershell -y"
    } else {
        Write-Host " Download portable ZIP: https://github.com/PowerShell/PowerShell/releases"
    }
}

Write-Host "\nQuick test:"
Write-Host "UTF-8 test: Hangul output OK"
Write-Host ok1; if ($?) { Write-Host ok2 }
