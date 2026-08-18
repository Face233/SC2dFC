[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^E\d{4}$')]
    [string]$ExperimentId,

    [string]$RunId,

    [string]$Destination
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$catalogPath = Join-Path $repositoryRoot "reports/checkpoint_catalog.json"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required."
}
if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    throw "Git LFS is required. Install it, then run: git lfs install"
}
if (-not (Test-Path -LiteralPath $catalogPath)) {
    throw "Checkpoint catalog not found: $catalogPath"
}

$catalog = Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json
$matches = @($catalog.checkpoints | Where-Object {
    $_.experiment_id -eq $ExperimentId -and ([string]::IsNullOrEmpty($RunId) -or $_.run_id -eq $RunId)
})

if ($matches.Count -eq 0) {
    throw "No archived checkpoint matches experiment '$ExperimentId' and run '$RunId'."
}
if ($matches.Count -gt 1) {
    $matches | Select-Object experiment_id, run_id, archive_path | Format-Table | Out-String | Write-Error
    throw "More than one run matches. Re-run with -RunId."
}

$checkpoint = $matches[0]
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path (Split-Path -Parent $repositoryRoot) "SC2dFC-checkpoints"
}

git -C $repositoryRoot fetch $catalog.archive_remote $catalog.archive_branch
if ($LASTEXITCODE -ne 0) { throw "Unable to fetch archive branch." }

if (-not (Test-Path -LiteralPath $Destination)) {
    # Suppress LFS smudge here: only the requested checkpoint is downloaded below.
    git -C $repositoryRoot -c filter.lfs.smudge= -c filter.lfs.required=false worktree add --detach $Destination "$($catalog.archive_remote)/$($catalog.archive_branch)"
    if ($LASTEXITCODE -ne 0) { throw "Unable to create checkpoint worktree: $Destination" }
}

$worktreeRoot = (git -C $Destination rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Destination exists but is not a Git worktree: $Destination"
}

git -C $worktreeRoot lfs pull $catalog.archive_remote --include=$checkpoint.archive_path
if ($LASTEXITCODE -ne 0) { throw "Unable to download checkpoint through Git LFS." }

$checkpointPath = Join-Path $worktreeRoot $checkpoint.archive_path
if (-not (Test-Path -LiteralPath $checkpointPath)) {
    throw "Git LFS completed without creating the expected file: $checkpointPath"
}
if ((Get-Item -LiteralPath $checkpointPath).Length -ne [int64]$checkpoint.size_bytes) {
    throw "Downloaded checkpoint size does not match the catalog: $checkpointPath"
}

Write-Host "Checkpoint ready: $checkpointPath"
