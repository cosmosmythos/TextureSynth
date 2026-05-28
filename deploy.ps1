# TextureSynth — rebuild and deploy to Blender
# Run from the DESIGNER folder: .\deploy.ps1

$ErrorActionPreference = "Stop"

$BLENDER_PYTHON = "C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe"
$BUILD_DIR      = "$PSScriptRoot\build"
$SITE_PACKAGES  = "$env:APPDATA\Blender Foundation\Blender\5.0\extensions\.local\lib\python3.11\site-packages"

Write-Host "`n=== Step 1: Configure (if needed) ===" -ForegroundColor Cyan
if (-not (Test-Path "$BUILD_DIR\CMakeCache.txt")) {
    cmake -S $PSScriptRoot -B $BUILD_DIR `
        -DBUILD_PYTHON_BINDINGS=ON `
        -DPython3_EXECUTABLE="$BLENDER_PYTHON" `
        -DCMAKE_BUILD_TYPE=Release
}

Write-Host "`n=== Step 2: Build texturesynth_core ===" -ForegroundColor Cyan
cmake --build $BUILD_DIR --config Release --target texturesynth_core

Write-Host "`n=== Step 3: Find built .pyd ===" -ForegroundColor Cyan
$pyd = Get-ChildItem "$BUILD_DIR" -Recurse -Filter "texturesynth_core*.pyd" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $pyd) { Write-Error "No .pyd found in build output."; exit 1 }
Write-Host "Found: $($pyd.FullName)  [$($pyd.Length) bytes]"

Write-Host "`n=== Step 4: Remove conflicting bundled DLLs (if any) ===" -ForegroundColor Cyan
@("msvcp140*", "vcruntime140*", "concrt140*", "vcomp140*") | ForEach-Object {
    Get-ChildItem $SITE_PACKAGES -Filter $_ -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item $_.FullName -Force
        Write-Host "  Removed: $($_.Name)"
    }
}

Write-Host "`n=== Step 5: Deploy .pyd to site-packages ===" -ForegroundColor Cyan
$dest = Join-Path $SITE_PACKAGES "texturesynth_core.pyd"
Copy-Item $pyd.FullName $dest -Force
Write-Host "Deployed: $dest"

Write-Host "`n=== Done. Restart Blender and test. ===" -ForegroundColor Green
