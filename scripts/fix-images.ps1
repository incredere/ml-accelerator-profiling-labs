# Safe helper script to move PNG files into the repo's `images/` directory
# - Idempotent: can be run multiple times without causing harm
# - Preserves an existing "images" file by renaming it to a timestamped backup
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\fix-images.ps1

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$imagesPath = Join-Path $repo 'images'
$backupPrefix = 'images_file_backup_'

# If a non-directory file named 'images' exists, back it up by renaming
if (Test-Path $imagesPath -PathType Leaf) {
    $ts = Get-Date -Format yyyyMMdd_HHmmss
    $bakName = "$backupPrefix$ts"
    Rename-Item -Path $imagesPath -NewName $bakName -ErrorAction Stop
    Write-Output "Renamed file 'images' -> $bakName"
}

# Ensure images directory exists
if (-not (Test-Path $imagesPath -PathType Container)) {
    New-Item -ItemType Directory -Path $imagesPath | Out-Null
    Write-Output "Created directory: $imagesPath"
} else {
    Write-Output "Directory exists: $imagesPath"
}

# Move PNG files from repo root into images/ (skip if destination exists)
$pngs = Get-ChildItem -Path $repo -File -Filter '*.png' -ErrorAction SilentlyContinue
if (-not $pngs -or $pngs.Count -eq 0) {
    Write-Output 'No PNG files found in repo root. Nothing to move.'
    exit 0
}

$moved = 0
foreach ($f in $pngs) {
    $dest = Join-Path $imagesPath $f.Name
    if (Test-Path $dest) {
        Write-Output "Skipping: destination exists -> $dest"
        continue
    }
    Move-Item -Path $f.FullName -Destination $imagesPath -ErrorAction Stop
    Write-Output "Moved: $($f.Name) -> $imagesPath"
    $moved++
}

Write-Output "Moved files count: $moved"
exit 0

