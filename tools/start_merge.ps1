# MergeStudio 一键启动（自动加载由服务端 __main__.py 完成: workspace→model→video）
param(
    [int]$Port = 8002
)
$ErrorActionPreference = 'Stop'
$env:APPDATA = [Environment]::GetFolderPath('ApplicationData')
Set-Location 'F:\DFL-PyTorch'

$msProc = Start-Process -FilePath 'F:\DFL-PyTorch\python\python.exe' -ArgumentList '-m','MergeStudio' -WorkingDirectory 'F:\DFL-PyTorch' -WindowStyle Hidden -PassThru
Write-Host "MergeStudio PID: $($msProc.Id)"

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try { $null = Invoke-WebRequest -Uri "http://127.0.0.1:${Port}/" -UseBasicParsing -TimeoutSec 3; $ready = $true; break } catch { }
}
if (-not $ready) { Write-Error 'MergeStudio 启动超时'; exit 1 }
Write-Host "MergeStudio 就绪 (端口 $Port) — 浏览器由服务端自动打开，页面会自动同步加载状态"

# 等自动加载完成（workspace → model → video），最多 120s
for ($i = 0; $i -lt 60; $i++) {
    try {
        $st = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/api/status" -TimeoutSec 5
        if (-not $st.autoload.configured) { Write-Host '无 .merge_auto.json 自动加载配置（手动模式）'; break }
        if ($st.autoload.stage -eq 'done') {
            Write-Host "自动加载完成: 模型=$($st.model.name)  视频=$($st.video.name)  帧数=$($st.video.total_frames)"
            break
        }
        if ($st.autoload.stage -eq 'error') { Write-Warning "自动加载出错: $($st.autoload.detail)"; break }
    } catch { }
    Start-Sleep -Seconds 2
}
