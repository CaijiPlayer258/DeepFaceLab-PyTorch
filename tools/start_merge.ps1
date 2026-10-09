# MergeStudio 一键启动 + 自动加载
param(
    [string]$VideoPath = "F:\DFL-PyTorch\workspace\SONE-705_[4K]-00.38.17.445-01.10.46.792.mkv",
    [string]$ModelName = "Anna",
    [int]$Port = 8001
)
$ErrorActionPreference = 'Stop'
$env:APPDATA = [Environment]::GetFolderPath('ApplicationData')
Set-Location 'F:\DFL-PyTorch'

$msProc = Start-Process -FilePath 'F:\DFL-PyTorch\python\python.exe' -ArgumentList '-m','MergeStudio' -WorkingDirectory 'F:\DFL-PyTorch' -WindowStyle Hidden -PassThru
Write-Host "MergeStudio PID: $($msProc.Id)"

$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 2
    try { $null = Invoke-WebRequest -Uri "http://127.0.0.1:${Port}/" -UseBasicParsing -TimeoutSec 3; $ready = $true; break } catch { }
}
if (-not $ready) { Write-Error 'MergeStudio 启动超时'; exit 1 }
Write-Host "MergeStudio 就绪 (端口 $Port)"

$openBody = '{"path": "F:\\DFL-PyTorch\\workspace"}'
$openFile = "$env:TEMP\_ms_open.json"
[IO.File]::WriteAllText($openFile, $openBody, (New-Object System.Text.UTF8Encoding($false)))
Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/api/project/open" -Method POST -ContentType 'application/json' -InFile $openFile | Out-Null
Write-Host 'Workspace 已打开'

$modelBody = "{`"name`": `"$ModelName`"}"
$modelFile = "$env:TEMP\_ms_model.json"
[IO.File]::WriteAllText($modelFile, $modelBody, (New-Object System.Text.UTF8Encoding($false)))
Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/api/models/load" -Method POST -ContentType 'application/json' -InFile $modelFile | Out-Null
Write-Host "模型 $ModelName 已加载"

Start-Process "http://127.0.0.1:${Port}"
Write-Host "浏览器已打开 http://127.0.0.1:${Port} — 翻帧调参即可"