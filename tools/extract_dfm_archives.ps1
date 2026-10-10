# extract_dfm_archives.ps1
# 逐包建夹解压 D:\Deepfake\模型\DFM 下的压缩包，防止内部重名 dfm 互相覆盖。
# 幂等：目标夹已有 dfm 则跳过；日志：F:\DFL-PyTorch\logs\dfm_extract_20261010.log
$ErrorActionPreference = 'Continue'
$dfmDir  = 'D:\Deepfake\模型\DFM'
$seven   = 'C:\Program Files\7-Zip\7z.exe'
$logPath = 'F:\DFL-PyTorch\logs\dfm_extract_20261010.log'

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $logPath -Value $line -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null
Set-Content -Path $logPath -Value '' -Encoding UTF8

$archives = Get-ChildItem $dfmDir -File | Where-Object { $_.Extension -in '.7z','.rar' } | Sort-Object Name
Write-Log ("Found {0} archives" -f $archives.Count)

$ok = 0; $fail = 0; $skip = 0
foreach ($arc in $archives) {
    try {
        $base = [IO.Path]::GetFileNameWithoutExtension($arc.Name)
        $pw = $null
        if ($base -match '^(.*?)\s*密码\s*([A-Za-z0-9]+)\s*$') { $base = $Matches[1]; $pw = $Matches[2] }
        $dest = Join-Path $dfmDir $base
        $existingDfm = Get-ChildItem $dest -Filter *.dfm -File -ErrorAction SilentlyContinue
        if ($existingDfm) { $skip++; Write-Log "SKIP (已有dfm) $($arc.Name)"; continue }
        if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Path $dest | Out-Null }
        if ($pw) { Write-Log "START $($arc.Name) -> $dest (带密码)" } else { Write-Log "START $($arc.Name) -> $dest" }
        $argList = @('x', $arc.FullName, "-o$dest", '-y', '-bd')
        if ($pw) { $argList += "-p$pw" } else { $argList += '-p' }
        $out = & $seven @argList 2>&1
        $code = $LASTEXITCODE
        if ($code -eq 0 -or $code -eq 1) {
            # 展平：若解压出唯一子目录且根层无文件，则上提内容
            $rootFiles = Get-ChildItem $dest -File
            $subDirs   = Get-ChildItem $dest -Directory
            if (-not $rootFiles -and $subDirs.Count -eq 1) {
                Get-ChildItem $subDirs[0].FullName -Force | Move-Item -Destination $dest -Force
                Remove-Item $subDirs[0].FullName -Force
                Write-Log "FLATTEN $($subDirs[0].Name) 上提"
            }
            # 单 dfm 且名不同则统一改名为 <夹名>.dfm
            $dfms = Get-ChildItem $dest -Filter *.dfm -File
            if ($dfms.Count -eq 1 -and $dfms[0].BaseName -ne $base) {
                Rename-Item $dfms[0].FullName -NewName "$base.dfm"
                Write-Log "RENAME -> $base.dfm"
            }
            $n = (Get-ChildItem $dest -Filter *.dfm -File).Count
            if ($n -ge 1) { Write-Log "OK $($arc.Name) dfm_count=$n"; $ok++ }
            else { Write-Log "FAIL $($arc.Name) 解压成功但无dfm"; $fail++ }
        } else {
            $tail = ($out | Select-Object -Last 4) -join ' | '
            Write-Log "FAIL $($arc.Name) exit=$code :: $tail"
            $fail++
        }
    } catch {
        Write-Log "FAIL $($arc.Name) exception=$($_.Exception.Message)"
        $fail++
    }
}

# 散装 dfm 归位
$looseJ = Join-Path $dfmDir '鞠婧祎384.dfm'
$folderJ = Join-Path $dfmDir '鞠婧祎384'
if ((Test-Path $looseJ) -and (Test-Path $folderJ)) {
    $inner = Get-ChildItem $folderJ -Filter *.dfm -File
    if ($inner.Count -ge 1) {
        $hLoose = (Get-FileHash $looseJ -Algorithm SHA256).Hash
        $dup = $false
        foreach ($f in $inner) { if ((Get-FileHash $f.FullName -Algorithm SHA256).Hash -eq $hLoose) { $dup = $true } }
        if ($dup) {
            Add-Type -AssemblyName Microsoft.VisualBasic
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($looseJ, 'OnlyErrorDialogs', 'SendToRecycleBin')
            Write-Log "DEDUP 散装鞠婧祎384.dfm 与解压件相同 -> 回收站"
        } else {
            Move-Item $looseJ (Join-Path $folderJ '鞠婧祎384_散装.dfm')
            Write-Log "KEEP 散装与解压件不同 -> 鞠婧祎384_散装.dfm"
        }
    }
}
$loose6 = Join-Path $dfmDir '女6.dfm'
if (Test-Path $loose6) {
    $f6 = Join-Path $dfmDir '女6'
    if (-not (Test-Path $f6)) { New-Item -ItemType Directory -Path $f6 | Out-Null }
    Move-Item $loose6 (Join-Path $f6 '女6.dfm')
    Write-Log "MOVED 女6.dfm -> 女6\"
}

Write-Log "ALL_DONE ok=$ok fail=$fail skip=$skip"
