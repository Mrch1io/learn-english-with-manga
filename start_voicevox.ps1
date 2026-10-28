param(
    [string]$EnginePath = (Join-Path $PSScriptRoot 'runtime\voicevox\windows-directml\run.exe'),
    [ValidateRange(1, 65535)][int]$Port = 50021
)
$ErrorActionPreference = 'Stop'
$endpoint = "http://127.0.0.1:$Port/version"
$mutex = [System.Threading.Mutex]::new($false, "Local\AI-Manga-Translator-VOICEVOX-$Port")
$ownsMutex = $false
try {
    try { $ownsMutex = $mutex.WaitOne(5000) }
    catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) {
        throw 'Another VOICEVOX launcher is still checking the engine. Please try connecting again shortly.'
    }
    try {
        $request = [System.Net.WebRequest]::Create($endpoint)
        $request.Proxy = $null
        $request.Timeout = 2000
        $response = $request.GetResponse()
        $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
        try { $version = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose(); $response.Dispose() }
        if (-not ($version -is [string]) -or [string]::IsNullOrWhiteSpace($version)) {
            throw 'The local /version endpoint did not return a VOICEVOX version string.'
        }
        Write-Host "VOICEVOX is already available at http://127.0.0.1:$Port (version $version)."
        exit 0
    } catch {
        # 本地连接超时不能证明引擎已启动，还需检查进程和端口。
        $probeFailure = $_.Exception.Message
    }
    if (-not (Test-Path -LiteralPath $EnginePath -PathType Leaf)) {
        throw "VOICEVOX engine not found: $EnginePath. Download the official Windows GPU engine; see README.md."
    }
    $engine = (Resolve-Path -LiteralPath $EnginePath).Path
    $logDirectory = Join-Path $PSScriptRoot 'data\voicevox'
    if ($Port -ne 50021) { $logDirectory = Join-Path $logDirectory $Port }
    $processFile = Join-Path $logDirectory 'process.json'
    if (Test-Path -LiteralPath $processFile -PathType Leaf) {
        # 冷启动可能尚未监听；核对程序路径，避免把复用的 PID 当作引擎。
        $savedProcess = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
        $recordedId = 0
        if ([int]::TryParse([string]$savedProcess.pid, [ref]$recordedId) -and $recordedId -gt 0) {
            $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedId"
            if ($null -ne $existing) {
                if (-not $existing.ExecutablePath) {
                    throw "Cannot verify the executable for recorded VOICEVOX PID $recordedId. No additional engine was started."
                }
                if ([string]::Equals([IO.Path]::GetFullPath($existing.ExecutablePath), $engine, [StringComparison]::OrdinalIgnoreCase)) {
                    Write-Host "VOICEVOX PID $recordedId is still initializing. Reusing it; please wait for the voice catalog. Logs: $logDirectory"
                    exit 0
                }
            }
        }
    }
    # 独占绑定也能发现已占用但尚未监听的端口。
    $portProbe = [System.Net.Sockets.Socket]::new([System.Net.Sockets.AddressFamily]::InterNetwork,
        [System.Net.Sockets.SocketType]::Stream, [System.Net.Sockets.ProtocolType]::Tcp)
    try {
        $portProbe.ExclusiveAddressUse = $true
        $portProbe.Bind([System.Net.IPEndPoint]::new([System.Net.IPAddress]::Loopback, $Port))
    } catch {
        throw "Port $Port is unavailable, and its VOICEVOX version could not be verified. No additional engine was started. Details: $probeFailure"
    } finally {
        $portProbe.Dispose()
    }
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $process = Start-Process -FilePath $engine -ArgumentList '--host','127.0.0.1','--port',"$Port",'--use_gpu','--output_log_utf8','--disable_mutable_api' -WorkingDirectory (Split-Path $engine) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'stdout.log') -RedirectStandardError (Join-Path $logDirectory 'stderr.log')
    @{ pid = $process.Id; executable = $engine } | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8
    Write-Host "Started GPU VOICEVOX (PID $($process.Id)). In the reader, click Connect. Logs: $logDirectory"
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
