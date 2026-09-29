#!/usr/bin/env python3
"""
page_settings - 运行时参数调整页面（模块）
提供 HTML 模板，供 page_trainer 导入使用。
"""
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DFL 参数设置</title>
	<style>
		@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;550;600&display=swap');
		*{box-sizing:border-box;margin:0;padding:0}
		::-webkit-scrollbar{width:6px;height:6px}
		::-webkit-scrollbar-track{background:transparent}
		::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:3px;transition:background .15s}
		::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.14)}
		*{scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.08) transparent}
		::selection{background:rgba(91,91,214,.35);color:#fff}
		body{background:#0a0a0b;color:rgba(255,255,255,.8);font-family:'Inter',-apple-system,sans-serif;font-size:13px;font-weight:450;min-height:100vh;padding:20px;-webkit-font-smoothing:antialiased}
		header{display:flex;align-items:center;gap:10px;height:40px;padding:0 14px;background:#0d0d0e;border-bottom:1px solid rgba(255,255,255,.06);flex-shrink:0;margin:-20px -20px 16px}
		header h1{font-size:13px;font-weight:550;color:rgba(255,255,255,.8);letter-spacing:-.01em}
		.back-btn{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);border-radius:5px;color:rgba(255,255,255,.45);text-decoration:none;font-size:11px;font-weight:500;transition:all .12s}
		.back-btn:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.65)}
		main{max-width:720px;margin:0 auto}
		.card{background:#0d0d0e;border:1px solid rgba(255,255,255,.06);border-radius:8px;padding:14px;margin-bottom:12px}
		.card h2{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:rgba(255,255,255,.25);margin-bottom:10px}
		.note{font-size:11px;color:rgba(255,255,255,.25);margin:-6px 0 10px}
		.row{display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.03)}
		.row:last-child{border-bottom:none}
		.lbl{font-size:12px;color:rgba(255,255,255,.55)}
		.val{font-size:12px;color:rgba(255,255,255,.75);font-weight:500}
		.ctrl{display:flex;align-items:center;gap:6px}
		.ctrl input[type=text]{width:72px;padding:3px 6px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.65);font-size:11px;font-family:inherit;text-align:center;outline:none;transition:border-color .12s}
		.ctrl input[type=text]:focus{border-color:rgba(91,91,214,.4)}
		.ctrl select{padding:3px 6px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.65);font-size:11px;font-family:inherit;outline:none;transition:border-color .12s}
		.ctrl select:focus{border-color:rgba(91,91,214,.4)}
		.range-ctrl{display:flex;align-items:center;gap:4px}
		.range-ctrl input{width:48px!important}
		.range-ctrl .sep{color:rgba(255,255,255,.2);font-size:11px}
		.toggle{position:relative;display:inline-block;width:30px;height:18px;cursor:pointer}
		.toggle input{opacity:0;width:0;height:0}
		.slider{position:absolute;inset:0;background:rgba(255,255,255,.08);border-radius:9px;transition:background .2s}
		.slider:before{content:'';position:absolute;left:2px;bottom:2px;width:14px;height:14px;background:rgba(255,255,255,.3);border-radius:50%;transition:all .2s}
		.toggle input:checked+.slider{background:rgba(91,91,214,.5)}
		.toggle input:checked+.slider:before{transform:translateX(12px);background:#8b8be6}
		select{padding:3px 8px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:5px;color:rgba(255,255,255,.65);font-size:12px;font-family:inherit;outline:none;transition:border-color .12s}
		select:focus{border-color:rgba(91,91,214,.4)}
		.actions{display:flex;gap:8px;justify-content:center;margin:16px 0}
		button{padding:5px 14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:6px;color:rgba(255,255,255,.55);cursor:pointer;font-size:11px;font-family:inherit;font-weight:500;transition:all .12s}
		button:hover{background:rgba(255,255,255,.07);color:rgba(255,255,255,.75)}
		button:disabled{opacity:.3;cursor:default}
		.btn-apply{background:linear-gradient(135deg,#5b5bd6,#8b5cf6)!important;color:#fff!important;border:none!important}
		.btn-apply:hover{opacity:.9!important;box-shadow:0 2px 8px rgba(91,91,214,.25)!important}
		.btn-reset{background:rgba(255,255,255,.04)!important;color:rgba(255,255,255,.45)!important}
		.btn-reset:hover{background:rgba(255,255,255,.07)!important;color:rgba(255,255,255,.65)!important}
		.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);z-index:1000;display:none;align-items:center;justify-content:center}
		.modal-overlay.open{display:flex}
		.modal-box{min-width:320px;max-width:420px;background:#0d0d0e;border:1px solid rgba(255,255,255,.08);border-radius:10px;box-shadow:0 16px 48px rgba(0,0,0,.5);padding:20px;text-align:center}
		.modal-box h2{font-size:14px;font-weight:550;color:rgba(255,255,255,.8);margin-bottom:8px}
		.modal-box p{font-size:12px;color:rgba(255,255,255,.45);margin-bottom:16px}
		.modal-box input{width:100%;padding:8px 12px;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.08);border-radius:6px;color:rgba(255,255,255,.75);font-size:13px;font-family:inherit;text-align:center;outline:none;transition:border-color .12s}
		.modal-box input:focus{border-color:rgba(91,91,214,.4);box-shadow:0 0 0 2px rgba(91,91,214,.08)}
		.modal-box .modal-err{font-size:12px;color:#ef4444;margin-top:8px;display:none}
		.modal-box .modal-actions{display:flex;gap:8px;margin-top:16px}
		.modal-box .modal-actions button{flex:1;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;font-family:inherit;border:none;transition:opacity .12s}
		#toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);padding:8px 20px;border-radius:6px;font-size:12px;font-weight:500;z-index:2000;opacity:0;transition:opacity .3s;pointer-events:none}
		#toast.show{opacity:1}
		#toast.ok{background:rgba(91,91,214,.2);color:#8b8be6;border:1px solid rgba(91,91,214,.25)}
		#toast.err{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.2)}
		@media(max-width:600px){body{padding:12px}header{margin:-12px -12px 12px}}
	</style>
</head>
<body>
<header>
  <a class="back-btn" href="/Trainer">&larr; 返回</a>
  <h1>运行时参数调整</h1>
</header>

<!-- Password modal (hidden until Apply clicked) -->
<div class="modal-overlay" id="pwd-modal">
  <div class="modal-box">
    <h2>&#128274; 需要密码确认</h2>
    <p>修改参数需要输入密码</p>
    <input id="pwd-input" type="password" placeholder="请输入密码" autocomplete="off">
    <div class="modal-err" id="pwd-err">密码错误，请重试</div>
    <div class="modal-actions">
      <button class="btn-reset" onclick="closePwdModal()">取消</button>
      <button class="btn-apply" onclick="confirmPassword()">确认</button>
    </div>
  </div>
</div>

<main>
  <div id="loading" style="color:#888;font-size:13px;padding:40px;text-align:center">正在加载设置…</div>
  <div id="content" style="display:none">
    <div class="card">
      <h2>模型架构</h2>
      <div class="note">仅作展示，不可修改</div>
      <div id="arch-fields"></div>
    </div>
    <div class="card">
      <h2>训练选项</h2>
      <div class="note">仅作展示，不可修改</div>
      <div id="train-fields"></div>
    </div>
    <div class="card">
      <h2>样本增强</h2>
      <div class="note">可动态调整</div>
      <div id="sp-fields"></div>
    </div>
    <div class="card">
      <h2>输出配置</h2>
      <div class="note">可动态调整</div>
      <div id="oc-fields"></div>
    </div>
    <div class="card">
      <h2>训练参数</h2>
      <div class="note">可动态调整</div>
      <div id="tr-fields"></div>
    </div>
    <div class="card">
      <h2>加载器</h2>
      <div class="note">可动态调整</div>
      <div id="ld-fields"></div>
    </div>
    <div class="card">
      <h2>保存与备份</h2>
      <div class="note">可动态调整</div>
      <div id="sv-fields"></div>
    </div>
    <div class="actions">
      <button class="btn-reset" onclick="loadAll()">重置表单</button>
      <button class="btn-apply" id="btn-apply" onclick="onApply()">应用参数</button>
    </div>
    <div id="last-refresh" style="text-align:center;font-size:11px;color:#444;padding:0 0 14px">等待加载...</div>
  </div>
</main>
<div id="toast"></div>

<script>
// ========== Field Definitions ==========
const ARCH_FIELDS = [
  ['resolution', '分辨率', v => v + 'px'],
  ['face_type', '人脸类型', null],
  ['archi', 'AE 架构', null],
  ['ae_dims', 'AE 维度', null],
  ['e_dims', '编码器维度', null],
  ['d_dims', '解码器维度', null],
  ['d_mask_dims', '解码器 Mask 维度', null],
];

const TRAIN_FIELDS = [
  ['adabelief', 'AdaBelief 优化器', v => v ? '&#10003;' : '&#10007;'],
  ['clipgrad', '梯度裁剪', v => v ? '&#10003;' : '&#10007;'],
  ['use_bf16', 'BF16 混合精度', v => v ? '&#10003;' : '&#10007;'],
  ['models_opt_on_gpu', '优化器在 GPU', v => v ? '&#10003;' : '&#10007;'],
  ['gan_power', 'GAN 强度', v => parseFloat(v).toFixed(4)],
  ['gan_patch_size', 'GAN 块大小', null],
  ['gan_dims', 'GAN 维度', null],
  ['true_face_power', 'True Face 强度', v => parseFloat(v).toFixed(4)],
  ['use_fast_generator', '快速生成器', v => v ? '&#10003;' : '&#10007;'],
  ['pretrain', '预训练模式（已停用，强制关闭）', () => '&#10007;'],
  ['gradient_checkpointing', '梯度检查点', v => v ? '&#10003;' : '&#10007;'],
];

const TR_FIELDS = [
  ['tr', 'lr', 'float', '学习率'],
  ['tr', 'lr_cos', 'int', '余弦退火周期'],
  ['tr', 'lr_dropout', 'select', '学习率 Dropout', {n:'关闭', y:'开启', cpu:'CPU'}],
  ['tr', 'random_src_flip', 'toggle', 'SRC 随机翻转'],
  ['tr', 'random_dst_flip', 'toggle', 'DST 随机翻转'],
  ['tr', 'random_hsv_power', 'float', '随机 HSV 强度'],
  ['tr', 'face_style_power', 'float', '人脸风格强度'],
  ['tr', 'bg_style_power', 'float', '背景风格强度'],
  ['tr', 'vgg_perceptual_power', 'float', 'VGG感知损失'],
  ['tr', 'masked_training', 'toggle', '遮罩区域训练'],
  ['tr', 'blur_out_mask', 'toggle', '羽化遮罩外围'],
  ['tr', 'eyes_mouth_prio', 'toggle', '嘴眼优先'],
  ['tr', 'uniform_yaw', 'toggle', 'Yaw 均匀分布'],
  ['tr', 'pretrain_single_decoder', 'toggle', '预训练单解码器'],
];

const SP_FIELDS = [
  ['sp', 'rotation_range', 'range', '旋转范围'],
  ['sp', 'scale_range', 'range', '缩放范围'],
  ['sp', 'tx_range', 'range', '水平位移'],
  ['sp', 'ty_range', 'range', '垂直位移'],
];

const OC_FIELDS = [
  ['oc', 'warp', 'toggle', '扭曲'],
  ['oc', 'transform', 'toggle', '变换'],
  ['oc', 'ct_mode', 'select', '色彩迁移', {none:'无', rct:'RCT', mt:'MT(RGB遮罩加权)', lct:'LCT', mkl:'MKL', idt:'IDT', sot:'SOT', sot_s:'SOT-S', lut:'LUT'}],
  ['oc', 'random_occlusion', 'toggle', '随机遮挡'],
  ['oc', 'random_noise', 'toggle', '随机噪点'],
  ['oc', 'random_color_power', 'float', '随机偏色强度'],
  ['oc', 'random_color_alg', 'select', '随机偏色算法', {none:'关闭', direct:'直接（强制正常光照）', rct:'RCT', mt:'MT(遮罩加权)', lct:'LCT'}],
  ['oc', 'edge_enhance_power', 'float', '边缘强化'],
  ['oc', 'preview_grade_alg', 'select', '合并预览调色', {same:'跟随颜色模式', none:'不调色', rct:'RCT', mt:'MT(RGB遮罩加权)', lct:'LCT', mkl:'MKL', idt:'IDT', sot:'SOT', lut:'LUT'}],
];

const LD_FIELDS = [
  ['ld', 'batch_size', 'int', 'Batch Size'],
];

const SV_FIELDS = [
  ['sv', 'crash_threshold', 'float', '崩溃阈值'],
  ['sv', 'backup_interval', 'int', '备份间隔（迭代数）'],
  ['sv', 'max_backups', 'int', '最大备份数'],
];

// ========== State ==========
let _pendingPayload = null;

// ========== Password Modal ==========
function openPwdModal() {
  document.getElementById('pwd-modal').classList.add('open');
  document.getElementById('pwd-input').value = '';
  document.getElementById('pwd-err').style.display = 'none';
  document.getElementById('pwd-input').focus();
}
function closePwdModal() {
  document.getElementById('pwd-modal').classList.remove('open');
  _pendingPayload = null;
}
document.getElementById('pwd-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') confirmPassword();
});

async function confirmPassword() {
  const pwd = document.getElementById('pwd-input').value;
  const err = document.getElementById('pwd-err');
  try {
    const r = await fetch('/check-password', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: pwd}),
    });
    const res = await r.json();
    if (res.ok) {
      const payload = _pendingPayload;
      closePwdModal();
      await doApply(payload, pwd);
    } else {
      err.style.display = 'block';
    }
  } catch(e) {
    err.textContent = '请求失败: ' + e.message;
    err.style.display = 'block';
  }
}

// ========== Build ==========
function buildReadonlyRow(label, valueHtml) {
  const row = document.createElement('div');
  row.className = 'row';
  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = label;
  const val = document.createElement('span');
  val.className = 'val';
  val.innerHTML = valueHtml;
  row.appendChild(lbl);
  row.appendChild(val);
  return row;
}

function buildField(group, key, type, label, opts) {
  const row = document.createElement('div');
  row.className = 'row';
  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = label;
  row.appendChild(lbl);
  const ctrl = document.createElement('div');
  ctrl.className = 'ctrl';
  if (type === 'toggle') {
    const le = document.createElement('label');
    le.className = 'toggle';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.dataset.group = group; cb.dataset.fieldKey = key;
    const sp = document.createElement('span');
    sp.className = 'slider';
    le.appendChild(cb); le.appendChild(sp);
    ctrl.appendChild(le);
  } else if (type === 'select') {
    const sel = document.createElement('select');
    sel.dataset.group = group; sel.dataset.fieldKey = key;
    for (const [val, txt] of Object.entries(opts)) {
      const opt = document.createElement('option');
      opt.value = val; opt.textContent = txt;
      sel.appendChild(opt);
    }
    ctrl.appendChild(sel);
  } else if (type === 'range') {
    const rc = document.createElement('div');
    rc.className = 'range-ctrl';
    const i0 = document.createElement('input'); i0.type = 'text';
    i0.dataset.group = group; i0.dataset.fieldKey = key; i0.dataset.rangeIdx = '0'; i0.placeholder = 'min';
    const sp = document.createElement('span'); sp.className = 'sep'; sp.textContent = '~';
    const i1 = document.createElement('input'); i1.type = 'text';
    i1.dataset.group = group; i1.dataset.fieldKey = key; i1.dataset.rangeIdx = '1'; i1.placeholder = 'max';
    rc.appendChild(i0); rc.appendChild(sp); rc.appendChild(i1);
    ctrl.appendChild(rc);
  } else {
    const inp = document.createElement('input'); inp.type = 'text';
    inp.dataset.group = group; inp.dataset.fieldKey = key; inp.placeholder = key;
    ctrl.appendChild(inp);
  }
  row.appendChild(ctrl);
  return row;
}

function getVal(el) {
  if (el.type === 'checkbox') return el.checked;
  if (el.tagName === 'SELECT') return el.value;
  const raw = el.value.trim();
  const n = Number(raw);
  if (!isNaN(n) && raw !== '') return n;
  return raw;
}

function setVal(el, val) {
  if (el.type === 'checkbox') { el.checked = !!val; return; }
  if (el.tagName === 'SELECT') { el.value = String(val); return; }
  if (typeof val === 'number') { el.value = String(val); return; }
  el.value = val != null ? String(val) : '';
}

function formatVal(v, fmt) {
  if (fmt) return fmt(v);
  if (v === true) return '&#10003;';
  if (v === false) return '&#10007;';
  if (v == null) return '<span style="color:#555">&#8212;</span>';
  return String(v);
}

// ========== Data Loading ==========
async function loadAll() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('content').style.display = 'none';
  try {
    const [modelR, genR] = await Promise.all([
      fetch('/current-model-options'),
      fetch('/current-settings'),
    ]);
    const modelOpts = await modelR.json();
    const genSettings = await genR.json();
    renderAll(modelOpts, genSettings);
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
    var _lr=document.getElementById('last-refresh');if(_lr)_lr.textContent='最后刷新: '+new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('loading').textContent = '加载失败: ' + e.message;
  }
}

function renderAll(modelOpts, genSettings) {
  document.getElementById('arch-fields').innerHTML = '';
  ARCH_FIELDS.forEach(([key, label, fmt]) => {
    document.getElementById('arch-fields').appendChild(buildReadonlyRow(label, formatVal(modelOpts[key], fmt)));
  });
  document.getElementById('train-fields').innerHTML = '';
  TRAIN_FIELDS.forEach(([key, label, fmt]) => {
    document.getElementById('train-fields').appendChild(buildReadonlyRow(label, formatVal(modelOpts[key], fmt)));
  });
  populateEditable('sp-fields', 'sp', SP_FIELDS, genSettings.sample_process_options || {});
  populateEditable('oc-fields', 'oc', OC_FIELDS, genSettings.output_sample_types || {});
  populateEditable('tr-fields', 'tr', TR_FIELDS, modelOpts);
  populateEditable('ld-fields', 'ld', LD_FIELDS, genSettings.loader || {});
  populateEditable('sv-fields', 'sv', SV_FIELDS, modelOpts);
}

function populateEditable(containerId, group, fieldDefs, data) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  fieldDefs.forEach(([g, key, type, label, opts]) => {
    if (g !== group) return;
    const row = buildField(g, key, type, label, opts);
    container.appendChild(row);
    const els = container.querySelectorAll(`[data-group="${group}"][data-field-key="${key}"]`);
    els.forEach(el => {
      const ridx = el.dataset.rangeIdx;
      if (ridx !== undefined) {
        const v = data[key];
        if (Array.isArray(v)) setVal(el, v[parseInt(ridx)]);
      } else {
        setVal(el, data[key]);
      }
    });
  });
}

// ========== Collect ==========
function collectEditable() {
  const result = {};
  const rangeBuf = {};
  document.querySelectorAll('[data-group]').forEach(el => {
    const group = el.dataset.group;
    const key = el.dataset.fieldKey;
    const ridx = el.dataset.rangeIdx;
    if (!group || !key) return;
    if (!result[group]) result[group] = {};
    if (ridx !== undefined) {
      const bk = group + '\t' + key;
      if (!rangeBuf[bk]) rangeBuf[bk] = [];
      rangeBuf[bk][parseInt(ridx)] = getVal(el);
    } else {
      result[group][key] = getVal(el);
    }
  });
  for (const [k, arr] of Object.entries(rangeBuf)) {
    const sep = k.indexOf('\t');
    result[k.slice(0, sep)][k.slice(sep + 1)] = arr;
  }
  return result;
}

// ========== Apply ==========
function onApply() {
  const collected = collectEditable();
  const genPayload = {};
  if (collected.sp) genPayload.sample_process_options = collected.sp;
  if (collected.oc) genPayload.output_sample_types = collected.oc;
  if (collected.ld) genPayload.loader = collected.ld;
  const modelPayload = {};
  if (collected.tr) Object.assign(modelPayload, collected.tr);
  if (collected.sv) Object.assign(modelPayload, collected.sv);
  _pendingPayload = {genPayload, modelPayload};
  openPwdModal();
}

async function doApply(payload, pwd) {
  const {genPayload, modelPayload} = payload;
  const btn = document.getElementById('btn-apply');
  btn.disabled = true;
  btn.textContent = '应用中...';
  try {
    let ok = true;
    if (Object.keys(genPayload).length > 0) {
      const r1 = await fetch('/update-settings', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(Object.assign({}, genPayload, {password: pwd})),
      });
      if (!(await r1.json()).ok) ok = false;
    }
    if (Object.keys(modelPayload).length > 0) {
      const r2 = await fetch('/update-model-options', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(Object.assign({}, modelPayload, {password: pwd})),
      });
      if (!(await r2.json()).ok) ok = false;
    }
    showToast(ok ? '参数已应用，将在下一次迭代生效' : '部分参数应用失败', ok ? 'ok' : 'err');
  } catch(e) {
    showToast('请求失败: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = '应用参数';
    _pendingPayload = null;
  }
}

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = type;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

loadAll();
</script>
</body>
</html>"""
