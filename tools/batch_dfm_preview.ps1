# batch_dfm_preview.ps1 — 逐个 dfm 出 33 帧预览结果到各自文件夹（一层结构：<名>\result_20261010）
# 幂等：夹内已有 30+ 张则跳过。AMP 双输入(morph)模型自动传 morph=1.0。
# 有效性由 dfm_test_merge.py 的 self-check 保证（输出=输入即退出码 2，判 FAIL）。
$ErrorActionPreference = 'Continue'
$dfmRoot = 'D:\Deepfake\模型\DFM'
$logPath = 'F:\DFL-PyTorch\logs\dfm_batch_20261010.log'
$py      = 'F:\DFL-PyTorch\python\python.exe'
$tool    = 'F:\DFL-PyTorch\tools\dfm_test_merge.py'
$frames  = 'F:\dfl\workspace\测试集\data_dst'
$lmks    = 'F:\dfl\workspace\测试集\data_dst\aligned_lmks'

$env:APPDATA = "$env:USERPROFILE\AppData\Roaming"
$env:LOCALAPPDATA = "$env:USERPROFILE\AppData\Local"
$env:PYTHONIOENCODING = 'utf-8'

function Write-Log([string]$msg) {
    Add-Content -Path $logPath -Value "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg -Encoding UTF8
}

Set-Content -Path $logPath -Value '' -Encoding UTF8

$dirs = Get-ChildItem $dfmRoot -Directory | Sort-Object Name
$todo = 0; $skip = 0; $ok = 0; $fail = 0
foreach ($d in $dirs) {
    $dfm = Get-ChildItem $d.FullName -Filter *.dfm -File | Select-Object -First 1
    if (-not $dfm) { Write-Log "SKIP_NODFM $($d.Name)"; $skip++; continue }
    $outDir = Join-Path $d.FullName 'result_20261010'
    if ((Test-Path $outDir) -and ((Get-ChildItem $outDir -File -ErrorAction SilentlyContinue).Count -ge 30)) {
        Write-Log "SKIP_DONE $($d.Name)"; $skip++; continue
    }
    $todo++
    # 探测是否 AMP 双输入（含 morph_value）→ 传 morph=1.0
    $probe = & $py -c "import onnxruntime as ort,sys; s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider']); print(1 if any('morph' in i.name for i in s.get_inputs()) else 0)" $dfm.FullName 2>$null
    $morphFlag = if ("$probe".Trim() -eq '1') { '1.0' } else { '0.5' }
    Write-Log "START $($d.Name) $($dfm.Name) morph=$morphFlag"
    $out = & $py $tool $dfm.FullName $frames $lmks $outDir 20 80 $morphFlag 2>&1
    $exitCode = $LASTEXITCODE
    $nOut = (Get-ChildItem $outDir -File -ErrorAction SilentlyContinue).Count
    $selfFail = ($out | Select-String -Pattern 'FAIL-SELF-CHECK') -ne $null
    if ($exitCode -eq 0 -and $nOut -ge 30 -and -not $selfFail) {
        Write-Log "OK $($d.Name) n=$nOut"; $ok++
    } else {
        $tail = ($out | Select-Object -Last 3) -join ' | '
        Write-Log "FAIL $($d.Name) exit=$exitCode n=$nOut :: $tail"; $fail++
    }
}
Write-Log "BATCH_DONE ok=$ok fail=$fail skip=$skip todo=$todo"
