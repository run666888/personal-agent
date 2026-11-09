param(
  [string]$Name = "PersonalAgentApp"
)

$ErrorActionPreference = "Stop"

function Test-FileLocked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return $false
  }

  try {
    $stream = [System.IO.File]::Open($Path, "Open", "ReadWrite", "None")
    $stream.Close()
    return $false
  } catch {
    return $true
  }
}

$effectiveName = $Name
$distDir = Join-Path $PSScriptRoot "dist"
$targetExe = Join-Path $distDir "$Name.exe"

if (Test-FileLocked -Path $targetExe) {
  $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $effectiveName = "$Name-$timestamp"
  Write-Host "Existing EXE is in use: $targetExe"
  Write-Host "Building a new copy instead: $effectiveName.exe"
}

python -m pip install pyinstaller
pyinstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name $effectiveName `
  --collect-data certifi `
  agent_app.py

Write-Host "Build completed: $(Join-Path $distDir "$effectiveName.exe")"
