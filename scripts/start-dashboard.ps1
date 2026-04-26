param(
    [switch]$NoBuild,
    [switch]$Logs,
    [switch]$NoAudioBridge,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
$dashboardUrl = "http://localhost:7860"
$healthUrl = "$dashboardUrl/api/health"
$audioBridgeScript = Join-Path $root "scripts\voicebot_audio_bridge.py"
$venvPython = Join-Path $root "venv\Scripts\python.exe"
$bridgePython = if (Test-Path $venvPython) { $venvPython } else { "python" }

Write-Host "Starting Docker services for the voicebot dashboard..."

$composeArgs = @("compose", "up", "-d")
if (-not $NoBuild) {
    $composeArgs += "--build"
}

Push-Location $root
try {
    & docker @composeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed with exit code $LASTEXITCODE"
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $openedBrowser = $false

    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Host "Dashboard is ready at $dashboardUrl"
                Start-Process $dashboardUrl
                $openedBrowser = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }

    if (-not $openedBrowser) {
        throw "Timed out waiting for dashboard at $dashboardUrl"
    }

    if (-not $NoAudioBridge) {
        Write-Host "Starting local audio bridge for automatic voice playback..."
        Start-Process $bridgePython -ArgumentList @($audioBridgeScript) -WorkingDirectory $root
    }

    if ($Logs) {
        Write-Host "Streaming docker compose logs. Press Ctrl+C to stop."
        & docker compose logs -f
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose logs failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Write-Host "Docker services are running in the background."
        Write-Host "Use 'docker compose down' to stop them."
    }
}
finally {
    Pop-Location
}
