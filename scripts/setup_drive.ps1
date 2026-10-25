<#
.SYNOPSIS
  Builds the portable ClinicalLingo drive layout from scratch (Windows host).

.DESCRIPTION
  Downloads every runtime piece, provisions them in a short-path staging folder on the
  local disk (flash drives are slow for many small writes and pip breaks on long paths),
  then copies the result next to the repo in ONE sequential pass.

  Final layout (DriveRoot = parent folder of this repo):
    <DriveRoot>\ClinicalLingo\      this repo
    <DriveRoot>\win\python\         Python 3.13 embeddable + requirements
    <DriveRoot>\win\ollama\         Ollama Windows zip release (no installer)
    <DriveRoot>\linux\python\       python-build-standalone tarball (extracted by linux/setup.sh)
    <DriveRoot>\linux\ollama\       Ollama linux-amd64 .tar.zst   (extracted by linux/setup.sh)
    <DriveRoot>\linux\wheels\       manylinux wheels for requirements.txt
    <DriveRoot>\linux\setup.sh
    <DriveRoot>\models\             OLLAMA_MODELS: phi3:mini + all-minilm
    <DriveRoot>\home\               Ollama identity key (redirected HOME)
    <DriveRoot>\env.cmd, start-ollama.cmd, run.cmd, ollama.cmd, env.sh, start-ollama.sh, run.sh

  Nothing is installed on the host. The staging folder is deleted at the end.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_drive.ps1
#>
param(
  [string]$OllamaVersion = "v0.40.11",
  [string]$PythonVersion = "3.13.17",
  [string]$PbsRelease    = "20261021",
  [string]$Stage         = (Join-Path $env:TEMP "clstage"),
  [switch]$SkipLinux
)
$ErrorActionPreference = "Stop"
$Repo      = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DriveRoot = (Resolve-Path (Join-Path $Repo "..")).Path
Write-Host "Repo: $Repo`nDrive root: $DriveRoot`nStaging: $Stage"

New-Item -ItemType Directory -Force -Path "$Stage\downloads","$Stage\win\python","$Stage\win\ollama","$Stage\models","$Stage\home","$Stage\tmp","$Stage\pip-cache" | Out-Null
$dl = "$Stage\downloads"

function Get-File($url, $dest) {
  if (Test-Path $dest) { Write-Host "  have $(Split-Path $dest -Leaf)"; return }
  Write-Host "  downloading $(Split-Path $dest -Leaf)"
  Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
}

Write-Host "== 1. downloads"
Get-File "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" "$dl\python-embed.zip"
Get-File "https://bootstrap.pypa.io/get-pip.py" "$dl\get-pip.py"
Get-File "https://github.com/ollama/ollama/releases/download/$OllamaVersion/ollama-windows-amd64.zip" "$dl\ollama-windows-amd64.zip"
Get-File "https://github.com/ollama/ollama/releases/download/$OllamaVersion/sha256sum.txt" "$dl\sha256sum.txt"
if (-not $SkipLinux) {
  $pbs = "cpython-$PythonVersion+$PbsRelease-x86_64-unknown-linux-gnu-install_only.tar.gz"
  Get-File "https://github.com/astral-sh/python-build-standalone/releases/download/$PbsRelease/$pbs" "$dl\$pbs"
  Get-File "https://github.com/astral-sh/python-build-standalone/releases/download/$PbsRelease/SHA256SUMS" "$dl\SHA256SUMS"
  Get-File "https://github.com/ollama/ollama/releases/download/$OllamaVersion/ollama-linux-amd64.tar.zst" "$dl\ollama-linux-amd64.tar.zst"
}

Write-Host "== 2. verify Ollama checksums"
$sums = Get-Content "$dl\sha256sum.txt"
foreach ($f in @("ollama-windows-amd64.zip") + $(if ($SkipLinux) { @() } else { @("ollama-linux-amd64.tar.zst") })) {
  $expected = ($sums | Where-Object { $_ -match [regex]::Escape($f) }) -split '\s+' | Select-Object -First 1
  $actual = (Get-FileHash "$dl\$f" -Algorithm SHA256).Hash.ToLower()
  if ($expected -ne $actual) { throw "checksum mismatch for $f" }
  Write-Host "  $f OK"
}

Write-Host "== 3. Windows Python (embeddable) + requirements"
Expand-Archive -Force "$dl\python-embed.zip" "$Stage\win\python"
$pth = Get-ChildItem "$Stage\win\python\python3*._pth" | Select-Object -First 1
(Get-Content $pth.FullName) -replace '^#import site', 'import site' | Set-Content $pth.FullName -Encoding ascii
$py = "$Stage\win\python\python.exe"
$env:PIP_CACHE_DIR = "$Stage\pip-cache"
& $py "$dl\get-pip.py" --no-warn-script-location -q
& $py -m pip install -q --no-warn-script-location -r "$Repo\requirements.txt"

Write-Host "== 4. Ollama (zip release, portable)"
Expand-Archive -Force "$dl\ollama-windows-amd64.zip" "$Stage\win\ollama"

Write-Host "== 5. pull models into staging (serve with redirected paths)"
$env:OLLAMA_MODELS = "$Stage\models"; $env:OLLAMA_HOST = "127.0.0.1:11434"; $env:OLLAMA_TMPDIR = "$Stage\tmp"
$env:USERPROFILE = "$Stage\home"; $env:HOME = "$Stage\home"
$serve = Start-Process -FilePath "$Stage\win\ollama\ollama.exe" -ArgumentList "serve" -PassThru -WindowStyle Hidden
try {
  $ok = $false
  for ($i = 0; $i -lt 60 -and -not $ok; $i++) { Start-Sleep 1; try { Invoke-RestMethod "http://127.0.0.1:11434/api/tags" | Out-Null; $ok = $true } catch {} }
  if (-not $ok) { throw "Ollama did not start" }
  & "$Stage\win\ollama\ollama.exe" pull all-minilm
  & "$Stage\win\ollama\ollama.exe" pull phi3:mini
} finally { Stop-Process -Id $serve.Id -Force -ErrorAction SilentlyContinue }

if (-not $SkipLinux) {
  Write-Host "== 6. Linux runtime (archives + wheels; extracted later by linux/setup.sh on a Linux host)"
  New-Item -ItemType Directory -Force -Path "$Stage\linux\python","$Stage\linux\ollama","$Stage\linux\wheels" | Out-Null
  Copy-Item "$dl\$pbs" "$Stage\linux\python\"
  (Get-Content "$dl\SHA256SUMS" | Where-Object { $_ -match [regex]::Escape($pbs) }) | Set-Content "$Stage\linux\python\SHA256SUMS" -Encoding ascii
  Copy-Item "$dl\ollama-linux-amd64.tar.zst" "$Stage\linux\ollama\"
  Copy-Item "$dl\sha256sum.txt" "$Stage\linux\ollama\ollama-sha256sum.txt"
  & $py -m pip download -q -r "$Repo\requirements.txt" -d "$Stage\linux\wheels" --only-binary=:all: `
      --python-version 3.13 --implementation cp --abi cp313 --abi abi3 --abi none `
      --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --platform any
  Copy-Item "$Repo\scripts\linux-setup.sh" "$Stage\linux\setup.sh"
}

Write-Host "== 7. launchers"
foreach ($f in "env.cmd","start-ollama.cmd","run.cmd","ollama.cmd","env.sh","start-ollama.sh","run.sh") { Copy-Item "$Repo\scripts\$f" "$Stage\$f" }

Write-Host "== 8. copy staging -> $DriveRoot (one sequential pass; slow on flash drives)"
robocopy $Stage $DriveRoot /E /NFL /NDL /NJH /NJS /NP /R:2 /W:2 /XD downloads pip-cache tmp | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }

Write-Host "== 9. clean staging"
Remove-Item -Recurse -Force $Stage
Write-Host "Done. Next: $DriveRoot\start-ollama.cmd  then  $DriveRoot\run.cmd doctor"
