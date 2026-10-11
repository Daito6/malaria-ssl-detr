<#
.SYNOPSIS
    Runs a queue of commands sequentially (Windows PowerShell 5.1).

.DESCRIPTION
    Queue file: one "<run name> | <command>" per line; blank lines and lines starting with # are skipped.
    Commands run from the repository root, in the current environment (activate medthesis first).

    - The output of each command is shown and appended to runs/<run name>/queue.log.
    - A failing command (non-zero exit code) is logged and the queue continues with the next one.
    - Commands already logged as "QUEUE DONE" in their queue.log are skipped, so an interrupted queue
      can simply be started again.
    - scripts/pretrain.py and scripts/train_detr.py commands get --resume when
      runs/<run name>/checkpoints/last.pt exists (interrupted training continues from the last epoch).
      The run name must therefore be the run directory the command writes to.

.EXAMPLE
    conda activate medthesis
    powershell -ExecutionPolicy Bypass -File scripts/run_queue.ps1 -Queue configs/queues/phase3_sweep.txt
#>
param(
    [Parameter(Mandatory = $true)][string]$Queue,
    [switch]$DryRun   # only print what would run (with --resume decisions), run nothing
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

function Write-Log([string]$Path, [string]$Text) {
    Add-Content -Path $Path -Value $Text -Encoding UTF8
}

$entries = @()
foreach ($line in Get-Content -Path $Queue -Encoding UTF8) {
    $t = $line.Trim()
    if ($t -eq "" -or $t.StartsWith("#")) { continue }
    $parts = $t.Split("|", 2)
    if ($parts.Count -ne 2) { throw "bad queue line (expected '<run name> | <command>'): $t" }
    $entries += [pscustomobject]@{ Name = $parts[0].Trim(); Command = $parts[1].Trim() }
}

$summary = @()
$i = 0
foreach ($e in $entries) {
    $i++
    $runDir = Join-Path $root "runs\$($e.Name)"
    $log = Join-Path $runDir "queue.log"
    $cmd = $e.Command
    $tag = "[$i/$($entries.Count)] $($e.Name)"

    if ((Test-Path $log) -and (Select-String -Path $log -SimpleMatch -Quiet -Pattern "QUEUE DONE: $cmd (")) {
        Write-Host "$tag already done, skipping: $cmd" -ForegroundColor DarkGray
        $summary += [pscustomobject]@{ Step = $i; Run = $e.Name; Status = "skipped (done)"; Command = $cmd }
        continue
    }
    $isTrain = $cmd -match "scripts[/\\](pretrain|train_detr)\.py"
    if ($isTrain -and ($cmd -notmatch "--resume") -and (Test-Path (Join-Path $runDir "checkpoints\last.pt"))) {
        $cmd = "$cmd --resume"
    }
    if ($DryRun) {
        Write-Host "$tag would run: $cmd"
        $summary += [pscustomobject]@{ Step = $i; Run = $e.Name; Status = "dry run"; Command = $cmd }
        continue
    }

    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $start = Get-Date
    Write-Host "$tag START: $cmd" -ForegroundColor Cyan
    Write-Log $log "===== $($start.ToString('s')) QUEUE START: $cmd"
    $ErrorActionPreference = "Continue"
    # cmd /c merges stderr into stdout without PowerShell wrapping it into error records
    cmd /c "$cmd 2>&1" | ForEach-Object { Write-Host $_; Write-Log $log $_ }
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    if ($code -eq 0) {
        # the logged command is the queue line (without an added --resume), so a rerun skips it
        Write-Log $log "===== $((Get-Date).ToString('s')) QUEUE DONE: $($e.Command) ($mins min)"
        Write-Host "$tag DONE in $mins min" -ForegroundColor Green
        $status = "done ($mins min)"
    } else {
        Write-Log $log "===== $((Get-Date).ToString('s')) QUEUE FAILED (exit $code): $cmd ($mins min)"
        Write-Host "$tag FAILED (exit $code) after $mins min, continuing" -ForegroundColor Red
        $status = "FAILED (exit $code)"
    }
    $summary += [pscustomobject]@{ Step = $i; Run = $e.Name; Status = $status; Command = $cmd }
}

Write-Host ""
$summary | Format-Table -AutoSize -Wrap | Out-String -Width 200 | Write-Host
