<#
.SYNOPSIS
    Waits until the phase 6 stage 2 queue is finished and the GPU is free, then runs the phase 7 screening queue
    (Windows PowerShell 5.1).

.DESCRIPTION
    Started by the user in a PowerShell window outside Claude Code, with `conda activate medthesis`, only after
    reports/phase7_plan.md is approved. It runs ONLY the -Queue file (micro-screening); full runs are not part of it.

    Every -PollSeconds it checks, and logs the reason when it keeps waiting:
      1. every step of -WaitQueue is logged "QUEUE DONE" in its runs/<run name>/queue.log (a FAILED step keeps the
         wrapper waiting: rerun that queue, the done steps are skipped);
      2. no `scripts/train_detr.py` process and no other `run_queue.ps1` is running;
      3. the GPU is free: no python process in nvidia-smi's compute-app list, utilization < -MaxUtil %,
         free memory >= -MinFreeGiB.
    All checks must pass on -StableChecks consecutive polls (queues have short gaps between steps); then
    scripts/run_queue.ps1 -Queue <Queue> is started once and its exit is logged.

    Before waiting it verifies that the medthesis env is active and that every `--config` file of -Queue exists;
    otherwise it stops at once (nothing is started).

    Log: runs/phase7_screening/wait_then_run.log (plus the per-run queue.log files written by run_queue.ps1).

.EXAMPLE
    conda activate medthesis
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/wait_then_run.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/wait_then_run.ps1 -DryRun   # one check, no run
#>
param(
    [string]$WaitQueue = "configs/queues/phase6_stage2.txt",
    [string]$Queue = "configs/queues/phase7_screening.txt",
    [double]$MinFreeGiB = 12.0,   # joint SSL peak 10.75 GiB (phase 5); set to the v2 smoke-run peak + margin
    [int]$MaxUtil = 30,           # % GPU utilization above which the GPU counts as busy
    [int]$PollSeconds = 300,
    [int]$StableChecks = 2,
    [switch]$DryRun               # check once, log the state, run run_queue.ps1 -DryRun; start nothing
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$logDir = Join-Path $root "runs\phase7_screening"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "wait_then_run.log"

function Log([string]$Text) {
    $line = "$((Get-Date).ToString('s')) $Text"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding UTF8
}

function Read-Queue([string]$Path) {
    $entries = @()
    foreach ($line in Get-Content -Path $Path -Encoding UTF8) {
        $t = $line.Trim()
        if ($t -eq "" -or $t.StartsWith("#")) { continue }
        $parts = $t.Split("|", 2)
        if ($parts.Count -ne 2) { throw "bad queue line in ${Path}: $t" }
        $entries += [pscustomobject]@{ Name = $parts[0].Trim(); Command = $parts[1].Trim() }
    }
    return $entries
}

# reasons why the wait queue is not finished (empty = finished)
function Get-QueueReasons([object[]]$Entries) {
    $notDone = @()
    $failed = @()
    foreach ($e in $Entries) {
        $log = Join-Path $root "runs\$($e.Name)\queue.log"
        if ((Test-Path $log) -and (Select-String -Path $log -SimpleMatch -Quiet -Pattern "QUEUE DONE: $($e.Command) (")) {
            continue
        }
        $notDone += $e.Name
        if ((Test-Path $log) -and (Select-String -Path $log -SimpleMatch -Quiet -Pattern "QUEUE FAILED")) {
            $failed += $e.Name
        }
    }
    $reasons = @()
    if ($notDone.Count -gt 0) {
        $first = $notDone[0]
        $reasons += "$WaitQueue`: $($notDone.Count)/$($Entries.Count) steps not done (next: $first)"
    }
    if ($failed.Count -gt 0) {
        $u = ($failed | Select-Object -Unique) -join ", "
        $reasons += "QUEUE FAILED marker in the log of not-done runs: $u (if the queue has stopped, rerun $WaitQueue; done steps are skipped)"
    }
    return $reasons
}

function Get-ProcessReasons {
    $reasons = @()
    $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine }
    $train = @($procs | Where-Object { $_.CommandLine -match "scripts[/\\]train_detr\.py" })
    if ($train.Count -gt 0) { $reasons += "train_detr.py running (pid $($train[0].ProcessId))" }
    $queues = @($procs | Where-Object { $_.CommandLine -match "run_queue\.ps1" })
    if ($queues.Count -gt 0) { $reasons += "run_queue.ps1 running (pid $($queues[0].ProcessId))" }
    return $reasons
}

function Get-GpuReasons {
    $reasons = @()
    $apps = @(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
    if ($LASTEXITCODE -ne 0) { return @("nvidia-smi failed (exit $LASTEXITCODE)") }
    # on Windows (WDDM) every desktop app is listed; only python processes count as GPU jobs
    $py = @($apps | Where-Object { $_ -match "python[w]?\.exe" })
    if ($py.Count -gt 0) { $reasons += "python process on the GPU: $($py[0].Trim())" }
    $q = (nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | Select-Object -First 1)
    $v = $q.Split(",")
    $freeGiB = [math]::Round([double]$v[0].Trim() / 1024, 2)
    $util = [int]$v[1].Trim()
    if ($util -ge $MaxUtil) { $reasons += "GPU utilization $util% >= $MaxUtil%" }
    if ($freeGiB -lt $MinFreeGiB) { $reasons += "free GPU memory $freeGiB GiB < $MinFreeGiB GiB" }
    return $reasons
}

# ---- preflight: environment and the screening queue's configs (stop at once, start nothing)
if ($env:CONDA_DEFAULT_ENV -ne "medthesis") {
    Log "ABORT: conda env 'medthesis' is not active (CONDA_DEFAULT_ENV='$env:CONDA_DEFAULT_ENV'); run 'conda activate medthesis' first"
    exit 1
}
foreach ($p in @($WaitQueue, $Queue)) {
    if (-not (Test-Path $p)) { Log "ABORT: queue file not found: $p"; exit 1 }
}
$waitEntries = Read-Queue $WaitQueue
$missing = @()
foreach ($e in (Read-Queue $Queue)) {
    $m = [regex]::Match($e.Command, "--config\s+(\S+)")
    if ($m.Success -and -not (Test-Path $m.Groups[1].Value)) { $missing += $m.Groups[1].Value }
}
if ($missing.Count -gt 0) {
    Log "ABORT: configs of $Queue missing: $(($missing | Select-Object -Unique) -join ', ')"
    exit 1
}
Log "waiting for $WaitQueue, then $Queue (MinFreeGiB $MinFreeGiB, MaxUtil $MaxUtil%, poll $PollSeconds s, stable $StableChecks)"

# ---- wait
$ok = 0
$lastReason = ""
while ($true) {
    $reasons = @(Get-QueueReasons $waitEntries) + @(Get-ProcessReasons) + @(Get-GpuReasons)
    if ($reasons.Count -eq 0) {
        $ok++
        Log "all checks passed ($ok/$StableChecks)"
        $lastReason = ""
        if ($ok -ge $StableChecks -or $DryRun) { break }
    } else {
        $ok = 0
        $reason = $reasons -join "; "
        if ($reason -ne $lastReason -or $DryRun) { Log "waiting: $reason" } else { Log "still waiting (same reason)" }
        $lastReason = $reason
        if ($DryRun) { break }
    }
    Start-Sleep -Seconds $PollSeconds
}

# ---- run the screening queue once
$runner = Join-Path $root "scripts\run_queue.ps1"
if ($DryRun) {
    Log "dry run: run_queue.ps1 -DryRun (nothing is started)"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $runner -Queue $Queue -DryRun
    exit 0
}
Log "START: run_queue.ps1 -Queue $Queue"
& powershell -NoProfile -ExecutionPolicy Bypass -File $runner -Queue $Queue
Log "END: run_queue.ps1 exit $LASTEXITCODE (per-step status in runs/<run name>/queue.log)"
