# DeepFaceLab-Torch AV1/10bit 代理生成脚本
# 用途：AV1/VP9/10bit 源在切脸时软解只有 ~10fps（瓶颈），先转 H.264 硬解友好代理，
#       切脸速度可提升到 25-40 fps（3060 的 NVDEC 本身支持 AV1 硬解，但 OpenCV 不走硬件）
# 用法：powershell -File make_proxy.ps1 -InputVideo "E:\xx.mkv"
#       或把视频拖到 make_proxy.bat 上
# 输出：<源目录>\<同名>_proxy.mp4（H.264 8bit，音频原样复制，帧数与源 1:1 对应，
#       因此代理切出的 aligned 按帧号仍能配回原视频给 MergeStudio 用）
param(
    [Parameter(Mandatory=$true)][string]$InputVideo,
    [string]$OutDir = "",
    [int]$Cq = 19,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'

$ffmpeg  = Join-Path $PSScriptRoot 'ffmpeg.exe';  if(-not (Test-Path $ffmpeg)) { $ffmpeg  = 'ffmpeg' }
$ffprobe = Join-Path $PSScriptRoot 'ffprobe.exe'; if(-not (Test-Path $ffprobe)) { $ffprobe = 'ffprobe' }

$codec = (& $ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt -of csv=p=0:s=x $InputVideo)
Write-Host "源编码: $codec"

if($codec -notmatch 'av1|vp9|10le|10be' -and -not $Force){
    Write-Host '源不是 AV1/VP9/10bit，切脸解码不是瓶颈，无需代理。'
    exit 0
}

$stem = [IO.Path]::GetFileNameWithoutExtension($InputVideo)
$dir  = if($OutDir){ $OutDir } else { Split-Path $InputVideo -Parent }
$out  = Join-Path $dir ($stem + '_proxy.mp4')

Write-Host "NVDEC 硬解 + NVENC 硬编转代理中..."
& $ffmpeg -y -hide_banner -loglevel warning -hwaccel cuda -i $InputVideo -c:v h264_nvenc -preset p4 -rc vbr -cq $Cq -b:v 0 -pix_fmt yuv420p -c:a copy $out

if($LASTEXITCODE -eq 0 -and (Test-Path $out)){
    $mb = [math]::Round((Get-Item $out).Length/1MB,1)
    Write-Host "代理完成: $out ($mb MB)"
    Write-Host "切脸时把 -i 指向这个 _proxy.mp4 即可；MergeStudio 合成仍用原视频。"
} else {
    Write-Error "转码失败 (exit=$LASTEXITCODE)"
    exit 1
}