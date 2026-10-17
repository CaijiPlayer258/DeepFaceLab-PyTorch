# 完整生命周期 Demo：起训 → 备份 → 改参 → 预览 → 收敛判断 → 收工
# 消息流形态（无命令过程），数据来自演进式 mock（--port 6790）
param([int]$Port = 6790)
$py = 'F:\DFL-PyTorch\python\python.exe'
Set-Location 'F:\DFL-PyTorch'

function Api($path, $method = 'GET', $body = $null) {
    if ($method -eq 'GET') { return Invoke-RestMethod -Uri "http://127.0.0.1:$Port$path" -TimeoutSec 15 }
    else { return Invoke-RestMethod -Uri "http://127.0.0.1:$Port$path" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 15 }
}
function Trend($h) {
    $n = $h.Count; $seg = [Math]::Min(48, [int]($n / 4))
    $sN = ($h[($n-$seg)..($n-1)] | Measure-Object -Property {$_[0]} -Average).Average
    $sP = ($h[([Math]::Max(0,$n-2*$seg))..($n-$seg-1)] | Measure-Object -Property {$_[0]} -Average).Average
    $dN = ($h[($n-$seg)..($n-1)] | Measure-Object -Property {$_[1]} -Average).Average
    $dP = ($h[([Math]::Max(0,$n-2*$seg))..($n-$seg-1)] | Measure-Object -Property {$_[1]} -Average).Average
    $sd = [math]::Round(($sN-$sP)/$sP*100, 1); $dd = [math]::Round(($dN-$dP)/$dP*100, 1)
    @{ s = "$([math]::Round($h[-1][0],4))（近段 $($sd)%）"; d = "$([math]::Round($h[-1][1],4))（近段 $($dd)%）"; n = $n }
}
function You($t)    { Write-Host "`n👤 你" -ForegroundColor Cyan; Write-Host "  $t" -ForegroundColor White }
function Bot($t)    { Write-Host "🤖 Agent" -ForegroundColor Green; Write-Host $t -ForegroundColor Gray }
function Div()      { Write-Host ("─" * 56) -ForegroundColor DarkGray }
function Stage($i)  { $script:loss = Api "/loss-history?stage=$i" }   # 推进训练时间

Write-Host "════════ 完整生命周期 Demo：从起训到收工 ════════" -ForegroundColor Yellow

# ── ① 起训 ──
Stage 0
You '用 anna 的 PNG 脸集开训，dst 用昨天切好的那批 JPG，接着上次的 Anna 模型续训'
Bot @"
好，都齐了：
  · src：anna 4702 张（PNG 脸图，直接进，不用转格式）
  · dst：17561 张（切脸 JPG）
  · 模型：Anna_SAEHD 续训（上次 iter 166.5 万，权重完整带上）

已在 GPU 上开跑（后台独立进程，掉线不影响），开了远程控制通道。
每 4 小时给你推一次报告，有异常随时喊你。
"@
Div

# ── ② 定时汇报①：早期 ──
$t = Trend $script:loss
You '(4 小时后 · agent 定时推送)'
Bot @"
📊 4h 汇报 · Anna_SAEHD

src $($t.s)    dst $($t.d)

前期正常下降，离收敛还早，不用管。
📎 [报告图：loss 曲线 + 两列预览]
"@
Div

# ── ③ 中途备份 ──
You '练到一半我要出门，先备份个存档点'
Bot @"
已打备份 ✅（自动备份间隔是 500 iter 一个存档点，
这次是手动额外打了一个，模型 + 优化器状态都在。
随时可以从这个点回滚。）
"@
Div

# ── ④ 改参：中期调整 ──
Stage 3
$t = Trend $script:loss
You '看下现在什么情况，降得有点慢了吧'
Bot @"
📊 $($t.n) 条记录
src $($t.s)    dst $($t.d)

下降变缓了（过了快速下降段）。两个选择：
  ① 保持——中段本来就是慢降，正常
  ② 降 lr（3.1e-05 → 1.5e-05）——收敛后期常这么做，压细节
另外建议把「随机遮挡」关了：你的素材遮挡不多，
这增强现在边际收益小，还拖点速度。
"@
You '行，都按你说的来'
Api '/update-model-options' 'POST' '{"password":"caiji","lr":1.5e-05,"random_occlusion":false}' | Out-Null
$opts = Api '/current-model-options'
Bot @"
已改（训练不停）：
  · lr 3.10e-05 → $($opts.lr.ToString('0.00e+00'))
  · 随机遮挡 开 → 关
半小时后我看 loss 反应再向你确认。
"@
Div

# ── ⑤ 看预览 ──
You '发两张脸来看看效果'
Bot @"
📎 预览（两列手机版）

从合并预览列看：正脸的五官已经稳了，侧脸还有点糊，
边缘接缝在这个阶段正常。
（预览是模型实时出的，不是成片质量；最终看 MergeStudio 合成。）
"@
Div

# ── ⑥ 收敛判断 ──
Stage 7
$t = Trend $script:loss
You '(又挂了一阵) 差不多了吧？'
Bot @"
📊 $($t.n) 条记录
src $($t.s)    dst $($t.d)

两条 loss 都走平了——继续挂收益很小，我的建议：收工。
预览侧脸细节这两小时也没再变。
要看最后一版预览确认下吗？还是直接保存退出？
"@
You '直接收吧'
Bot @"
收到。已保存并优雅退出 ✅
  · 模型权重 + 优化器状态已落盘（约 3.4GB）
  · 下次可直接续训；也可以直接拿去 MergeStudio 合成出片
全程 2 次改参、1 次手动备份，无异常中断。
👋
"@

Write-Host "`n════════ Demo 结束 ════════" -ForegroundColor Yellow
