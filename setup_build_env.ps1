# ============================================================
# setup_build_env.ps1 — ROCm/HIP build env for AMD GPUs on Windows
# Auto-detects the installed GPU architecture. Works on any AMD
# card whose gfx target is supported by the installed ROCm SDK.
# Override detection by setting $env:GPU_ARCHS before running.
# ============================================================

# --- Launch VS Dev Shell if not already in one (build process needs link.exe) ---
if (-not $env:VSCMD_VER) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $devShell = Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1"
        if ($vsPath -and (Test-Path $devShell)) { 
            & $devShell -Arch amd64 -SkipAutomaticLocation | Out-Null 
        }
        else { 
            Write-Warning "VS with C++ tools not found via vswhere. Run from a VS Developer PowerShell manually." -ForegroundColor Red
        }
    } else {
        Write-Warning "vswhere not found. Run from a VS Developer PowerShell manually." -ForegroundColor Red
    }
}

if (-not $env:VSCMD_VER) {
    Write-Warning "VS Dev Shell not active (VSCMD_VER unset). link.exe may be unavailable; the build will likely fail at link time."  -ForegroundColor Red
}

# --- ROCm environment ---
rocm-sdk init
$env:ROCM_ROOT = (rocm-sdk path --root)
$env:ROCM_HOME = $env:ROCM_ROOT
$env:ROCM_PATH = $env:ROCM_ROOT
$env:PATH = "$env:ROCM_ROOT\lib\llvm\bin;$env:ROCM_ROOT\bin;$env:PATH"
$env:CC  = "clang-cl"
$env:CXX = "clang-cl"
$env:DISTUTILS_USE_SDK = "1"
$env:BUILD_TARGET = "rocm"
Remove-Item Env:\ROCM_DEVICE_LIB_ROOT -ErrorAction SilentlyContinue

# --- GPU arch: respect override, else auto-detect the installed GPU ---
if (-not $env:GPU_ARCHS) {
    $detected = $null
    try {
        $detected = (python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])" 2>$null).Trim()
    } catch {}

    if ($detected -match '^gfx\d+$') {
        # sanity-check the SDK can build for it
        $supported = (rocm-sdk targets 2>$null) -split ';' | ForEach-Object { $_.Trim() }
        if ($supported -contains $detected) {
            $env:GPU_ARCHS = $detected
        } else {
            Write-Warning "Detected $detected but the ROCm SDK targets are: $($supported -join ', '). Building for $detected anyway." -ForegroundColor Red
            $env:GPU_ARCHS = $detected
        }
    } else {
        $env:GPU_ARCHS = "gfx1030"
        Write-Warning "Could not auto-detect GPU arch; defaulting to gfx1030. Set `$env:GPU_ARCHS to override."
    }
}

# --- Display final status to the user ---
Write-Host "ROCm build env ready | GPU_ARCHS=$env:GPU_ARCHS | ROCM_ROOT=$env:ROCM_ROOT" -ForegroundColor Green