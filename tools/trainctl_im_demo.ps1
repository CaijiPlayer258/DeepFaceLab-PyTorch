# trainctl IM 交互 Demo —— 消息流效果（无命令过程，只显示结果）
# 场景：训练挂机中（Anna_SAEHD，平台期），用户在手机 IM 上与 agent 对话
# 运行：python tools/trainctl_demo.py --port 6790 后，本脚本对其取真数据渲染消息

param(
    [int]$Port = 6790,
    [string]$ReportOut = 'F:\DFL-PyTorch\logs\demo_report_final.jpg'
)
$py = 'F:\DFL-PyTorch\python\python.exe'
Set-Location 'F:\DFL-PyTorch'

function Get-Api($path) {
    (Invoke-RestMethod -Uri "http://127.0.0.1:$Port$path" -TimeoutSec 15)
}

function Show-User($t)   { Write-Host "`n{'='*60}" -ForegroundColor DarkGray; Write-Host "👤 你" -ForegroundColor Cyan; Write-Host $t -ForegroundColor White }
function Show-Agent($t)  { Write-Host "🤖 Agent" -ForegroundColor Green; Write-Host $t -ForegroundColor Gray }
function Show-Divider()  { Write-Host ("─" * 52) -ForegroundColor DarkGray }

# ─────────────────────────────────────────────────────────
Write-Host "══════════ IM 消息流 Demo（深变训练监控）══════════" -ForegroundColor Yellow

# 场景 1：主动定时汇报（训练挂机中，agent 4 小时一次推报告）
Show-User '(无——agent 定时主动推送)'
$loss = Get-Api '/loss-history'
$opts = Get-Api '/current-model-options'
$sCur = [math]::Round($loss[-1][0], 4); $dCur = [math]::Round($loss[-1][1], 4)
$sw = @()
foreach ($k in @(@('lr','lr'),@('lr_dropout','lr_dropout'),@('random_occlusion','随机遮挡'),@('random_noise','随机噪点'),@('eyes_mouth_prio','眼嘴优先'),@('uniform_yaw','均匀yaw'),@('random_dst_flip','dst翻转'),@('use_bf16','BF16'))) {
    $v = $opts.($k[0]); if ($v -and $v -ne 0 -and $v -ne 'n') { $sw += "$($k[1])=$v" }
}
Show-Agent @"
📊 训练汇报 · Anna_SAEHD（已挂 18h）

loss 已进入平台期：src $sCur / dst $dCur，
最近 50 个记录几乎走平（±0.5%），继续挂收益不大。

启用中的开关：$($sw -join ' · ')

可选：① 降 lr 再压一压 ② 现在保存收工 ③ 继续挂我 4h 后再报
回数字即可。
📎 [图片] 完整报告见下一条
"@
& $py tools\trainctl_report.py --port $Port --out $ReportOut --preview-wait 8 | Out-Null
Show-Agent "📎 训练报告_$(Get-Date -Format 'MMdd-HHmm').jpg（loss 曲线 + 两列预览，1080 宽）`n   [文件已在对话中]"

Show-Divider

# 场景 2：查询
Show-User '现在怎么样了'
Show-Agent @"
📊 Anna_SAEHD · lr 3.1e-05

src 0.2071（近段走平）
dst 0.2875（近段走平）

两个 loss 都在收敛末段，没有异常。要看脸的话说「预览」。
"@

Show-Divider

# 场景 3：改参（多参数一条消息）
Show-User 'src 有点飘，lr 降到 2e-5，随机遮挡关掉'
Show-Agent @"
已改（训练不停，下一步即生效）：
  · lr 3.10e-05 → 2.00e-05
  · 随机遮挡 开 → 关

30 分钟后我看一眼 loss 反应，有异常主动喊你。
"@

Show-Divider

# 场景 4：不可热改的参数（如实告知）
Show-User '顺便把分辨率提到 512'
Show-Agent @"
分辨率是架构参数，不能热改——改它需要重启训练（从 416 的权重
没法直接续）。建议这轮收工后，下次开训时再设 512。
要现在就准备重启吗？
"@

Show-Divider

# 场景 5：保存/退出
Show-User '先保存一下，我出门了'
Show-Agent @"
已保存 ✅（约 3.4GB 写入 F 盘，后台完成，训练继续挂着。
你忙你的，4 小时后我照常汇报。）
"@

Show-User '行了，今天到这，收工吧'
Show-Agent @"
收到，已保存并开始优雅退出。
模型和优化器状态都已落盘，下次直接续训。
👋
"@

Write-Host "`n══════════ Demo 结束 ══════════" -ForegroundColor Yellow
Write-Host "报告图: $ReportOut" -ForegroundColor DarkGray
