# add_level3.ps1
param(
    [string]$RepoPath = ".",
    [string]$OutputsPath = "."
)
$ErrorActionPreference = "Stop"
function Fail($m){ Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
function Info($m){ Write-Host "-> $m" -ForegroundColor Cyan }
function OK($m)  { Write-Host "OK  $m" -ForegroundColor Green }

$RepoPath = (Resolve-Path $RepoPath).Path
if (-not (Test-Path (Join-Path $RepoPath ".git"))) { Fail "Not a git repo: $RepoPath" }

$required = @("transformer_profiling.ipynb","Level3_README.md","gpt2.pt.trace.json","main_README_level3_section.md")
foreach ($f in $required) {
    if (-not (Test-Path (Join-Path $OutputsPath $f))) { Fail "Missing: $OutputsPath\$f" }
}
OK "Source files located"

$levelDir = Join-Path $RepoPath "Level3_transformer_profiling"
if (-not (Test-Path $levelDir)) { New-Item -ItemType Directory -Path $levelDir | Out-Null }

Copy-Item -Force (Join-Path $OutputsPath "transformer_profiling.ipynb") (Join-Path $levelDir "transformer_profiling.ipynb")
Copy-Item -Force (Join-Path $OutputsPath "gpt2.pt.trace.json")          (Join-Path $levelDir "gpt2.pt.trace.json")
Copy-Item -Force (Join-Path $OutputsPath "Level3_README.md")            (Join-Path $levelDir "README.md")
OK "Copied notebook, trace, and README into Level3_transformer_profiling/"

$mainReadme = Join-Path $RepoPath "README.md"
if (-not (Test-Path $mainReadme)) { Fail "No README.md in repo root." }
$readmeText = Get-Content $mainReadme -Raw
if ($readmeText -match "Level 3") {
    Info "Main README already references 'Level 3' — skipping patch."
} else {
    $snippet = Get-Content (Join-Path $OutputsPath "main_README_level3_section.md") -Raw
    $snippet = [regex]::Replace($snippet, '^<!--.*?-->\s*', '', 'Singleline')
    $patched = $readmeText.TrimEnd() + "`r`n`r`n" + $snippet.TrimEnd() + "`r`n"
    Set-Content -Path $mainReadme -Value $patched -NoNewline
    OK "Appended Level 3 section to main README.md"
}

Push-Location $RepoPath
try {
    git add "Level3_transformer_profiling" "README.md" | Out-Null
    OK "Staged changes"
    Write-Host ""
    git status --short
    Write-Host ""
    Write-Host 'Next: review in Source Control (Ctrl+Shift+G), then commit + push.' -ForegroundColor Yellow
} finally { Pop-Location }
