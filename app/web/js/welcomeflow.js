/* 欢迎页背景：流动云海（B8 定稿 shader，纯 WebGL，无依赖） */

const canvas = document.getElementById("wGl");
let started = false, gl = null, raf = 0, t0 = 0, prog = null, uni = {};
let reduce = false;

const VS = "attribute vec2 aPos; varying vec2 vUv;\n" +
  "void main(){ vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }";

const FS = `
precision highp float;
uniform float uTime;
uniform vec2 uRes;
varying vec2 vUv;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
float hash1(float n){ return fract(sin(n)*43758.5453); }
float noise(vec2 p){
  vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1.,0.)),u.x), mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),u.x), u.y);
}
float fbm(vec2 p){
  float v=0.0,a=0.5;
  for(int i=0;i<5;i++){ v+=a*noise(p); p*=2.03; a*=0.5; }
  return v;
}
vec2 toP(){ return (vUv - 0.5) * vec2(uRes.x/uRes.y, 1.0); }
vec3 tri(float t){
  t = fract(t) * 3.0;
  vec3 c0 = vec3(0.40, 0.90, 1.00);
  vec3 c1 = vec3(0.12, 0.34, 1.00);
  vec3 c2 = vec3(0.56, 0.30, 0.98);
  if (t < 1.0) return mix(c0, c1, t);
  if (t < 2.0) return mix(c1, c2, t - 1.0);
  return mix(c2, c0, t - 2.0);
}

float density(vec2 p, float t){
  vec2 q;
  q.x = fbm(p * 0.85 + vec2(0.0, t * 0.30));
  q.y = fbm(p * 0.85 + vec2(t * 0.26, 0.0));
  vec2 r;
  r.x = fbm(p + 3.5 * q + vec2(1.7, 9.2) + t * 0.14);
  r.y = fbm(p + 3.5 * q + vec2(8.3, 2.8) - t * 0.12);
  float f = fbm(p * 2.5 + 4.55 * r);
  return clamp(f, 0.0, 1.0);
}

vec3 shade(vec2 p, float t){
  float d = density(p, t);
  float s = pow(d, 0.40);
  float k = 0.0 + d * 0.55 + 0.08 * sin(t * 0.40);
  vec3 tint = tri(k);
  vec3 col = tint * (0.12 + s * 2.50);
  col += vec3(1.0) * pow(s, 6.0) * 0.22;
  col += tri(k + 0.30) * pow(s, 3.0) * 0.30;
  return col;
}

void main(){
  vec2 p = toP();
  float t = uTime * 0.40;
  p *= 1.0 + 0.035 * sin(t * 0.4);

  vec3 c = shade(p, t);
  c += shade(p + vec2(0.005, 0.0), t);
  c += shade(p - vec2(0.005, 0.0), t);
  c += shade(p + vec2(0.0, 0.005), t);
  c += shade(p - vec2(0.0, 0.005), t);
  c /= 5.0;

  vec3 back = shade(p * 0.55 + vec2(0.0, 0.10), t * 0.45) * 0.35;
  c = c + back * (1.0 - c);

  c = mix(c, c * vec3(0.92, 0.88, 1.10), smoothstep(0.35, 0.0, c.g));
  c = mix(c, c * vec3(0.88, 1.02, 1.08), smoothstep(0.55, 1.2, c.g));

  float d = density(p, t);
  float vig = smoothstep(1.45, 0.05, length(p * vec2(0.9, 1.0)));
  float a = clamp(0.10 + pow(d, 2.80) * 0.95, 0.0, 1.0) * (0.30 + 0.70 * vig);

  float g = hash1(dot(vUv * uRes, vec2(1.0, 1.0)) + fract(uTime) * 91.7) - 0.5;
  c += g * 0.022;
  a += g * 0.010;

  gl_FragColor = vec4(c * a, a);
}
`;

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.error("welcome bg shader: " + gl.getShaderInfoLog(sh));
    return null;
  }
  return sh;
}

function resize() {
  if (!gl) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const w = canvas.clientWidth || window.innerWidth;
  const h = canvas.clientHeight || window.innerHeight;
  canvas.width = Math.max(1, Math.floor(w * dpr));
  canvas.height = Math.max(1, Math.floor(h * dpr));
  gl.viewport(0, 0, canvas.width, canvas.height);
}

function frame() {
  raf = requestAnimationFrame(frame);
  gl.uniform1f(uni.time, (performance.now() - t0) / 1000);
  gl.uniform2f(uni.res, canvas.clientWidth, canvas.clientHeight);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

export function startWelcome3D() {
  window.__w3dRunning = true;
  if (!canvas) return;
  if (started) {
    // 之前初始化过、但循环被 stop() 掐掉了（退出登录后回到欢迎页就是这种情况）：
    // 直接续上 requestAnimationFrame 就行，不用重建 GL 资源和 shader。
    if (!raf) raf = requestAnimationFrame(frame);
    return;
  }
  started = true;

  try {
    gl = canvas.getContext("webgl", { antialias: false, alpha: false });
  } catch (e) { gl = null; }
  if (!gl) { canvas.style.display = "none"; return; }

  const vs = compile(gl.VERTEX_SHADER, VS);
  const fs = compile(gl.FRAGMENT_SHADER, FS);
  if (!vs || !fs) { canvas.style.display = "none"; return; }

  prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { canvas.style.display = "none"; return; }
  gl.useProgram(prog);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "aPos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  uni.time = gl.getUniformLocation(prog, "uTime");
  uni.res = gl.getUniformLocation(prog, "uRes");

  resize();
  window.addEventListener("resize", resize);

  reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // 起始相位随机：shader 只由 uTime 驱动，每次都从 0 开始的话，
  // 每次打开的动画完全一样（用户会觉得"背景总是同一个样子"）。
  // 随机跳进 0~60 秒的任意相位，画面就每次不同。
  t0 = performance.now() - Math.random() * 60000;
  if (reduce) {
    gl.uniform1f(uni.time, 8.0);
    gl.uniform2f(uni.res, canvas.clientWidth, canvas.clientHeight);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  } else {
    frame();
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      cancelAnimationFrame(raf); raf = 0;
    } else if (!reduce && !raf) {
      t0 = performance.now() - 8 * 1000;
      frame();
    }
  });
}

export function stopWelcome3D() {
  cancelAnimationFrame(raf);
  raf = 0;
  window.__w3dRunning = false;
}
