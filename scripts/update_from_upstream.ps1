# Bring this personal fork up to date with upstream FreeVideo, only when you run it.
# Shows what is new, asks before merging, keeps your own changes on top, and stops
# cleanly (nothing changed) if the merge would conflict.
$ErrorActionPreference = 'Stop'
$Upstream = 'https://github.com/FlashML-org/FreeVideo.git'
$ForkRepository = 'whiteak/FreeVideo-AI-'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

function Fail($message) { Write-Host ''; Write-Host $message -ForegroundColor Red; exit 1 }
function Invoke-Git { & git.exe @args; if ($LASTEXITCODE -ne 0) { throw "git $($args -join ' ') failed" } }
function Version($ref) {
    try { (& git show "${ref}:freevideo_engine/release_notes.json" 2>$null | ConvertFrom-Json).product_version } catch { '?' }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail 'Git is not installed or not on PATH.' }
if ((& git rev-parse --is-inside-work-tree 2>$null) -ne 'true') { Fail "$Root is not a Git checkout." }
if ((& git rev-parse --is-shallow-repository) -eq 'true') { Fail 'This checkout is shallow. Run: git fetch --unshallow origin' }
if (& git status --porcelain --untracked-files=no) {
    Fail 'You have uncommitted changes. Commit or stash them first, then run this again.'
}
$branch = & git rev-parse --abbrev-ref HEAD
if ($branch -ne 'main') { Fail "You're on branch '$branch'. Switch to main first: git checkout main" }

if (-not (& git remote 2>$null | Select-String -SimpleMatch -Quiet 'upstream')) {
    Write-Host "Adding upstream remote: $Upstream"
    Invoke-Git remote add upstream $Upstream
}
Write-Host 'Checking upstream for new changes...'
Invoke-Git fetch --quiet upstream main
Invoke-Git fetch --quiet origin main

$incoming = @(& git log --oneline --no-merges 'HEAD..upstream/main')
$current = Version 'HEAD'; $latest = Version 'upstream/main'
if ($incoming.Count -eq 0) { Write-Host "Already up to date with upstream (version $current)." -ForegroundColor Green; exit 0 }

Write-Host ''
Write-Host "Your fork: version $current    Upstream: version $latest    New upstream commits: $($incoming.Count)"
Write-Host ''
$incoming | Select-Object -First 25 | ForEach-Object { Write-Host "  $_" }
if ($incoming.Count -gt 25) { Write-Host "  ...and $($incoming.Count - 25) more" }
try {
    $notes = (& git show 'upstream/main:freevideo_engine/release_notes.json' | ConvertFrom-Json).en
    if ($notes.summary) { Write-Host ''; Write-Host "What upstream says is new in ${latest}:"; Write-Host "  $($notes.summary)" }
} catch { }
Write-Host ''
$answer = Read-Host 'Merge these into your fork now? [y/N]'
if ($answer -notmatch '^(y|yes)$') { Write-Host 'Nothing changed.'; exit 0 }

$before = & git rev-parse HEAD
& git merge --no-edit --no-ff -m "Merge upstream FreeVideo $latest" upstream/main
if ($LASTEXITCODE -ne 0) {
    $conflicts = & git diff --name-only --diff-filter=U
    & git merge --abort
    Fail ("The merge conflicts with your changes in:`n  " + ($conflicts -join "`n  ") +
          "`nNothing was changed. Ask Claude to do this merge, or resolve it by hand.")
}

# Your update protection must survive the merge.
$updater = Get-Content -LiteralPath 'freevideo_engine\launcher_update.py' -Raw
if ($updater -notmatch [regex]::Escape("REPOSITORY = '$ForkRepository'")) {
    & git reset --hard $before
    Fail 'Upstream changed the updater so your fork would follow their releases again. The merge was undone; ask Claude to merge by hand.'
}

Write-Host ''
Write-Host "Merged. Your fork is now on upstream version $latest with your changes on top." -ForegroundColor Green
$changed = & git diff --name-only $before HEAD -- pyproject.toml constraints freevideo_engine/bootstrap_versions.json
if ($changed) {
    Write-Host ''
    Write-Host 'Upstream changed the Python packages. Before starting FreeVideo, run setup again:' -ForegroundColor Yellow
    Write-Host '  setup.cmd --reuse-models <the model folder you installed with>'
}
Write-Host 'Restart FreeVideo to use the new version.'
Write-Host ''
$push = Read-Host 'Push the merged main to your GitHub fork too? [y/N]'
if ($push -match '^(y|yes)$') { Invoke-Git push origin main; Write-Host 'Pushed.' -ForegroundColor Green }
