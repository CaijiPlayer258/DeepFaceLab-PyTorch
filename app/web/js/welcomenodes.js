/* 变体 B：神经网络星图 —— 节点缓慢漂移，连线随距离淡出，颜色沿链接流动 */
import * as THREE from "../vendor/three.module.min.js";

const canvas = document.getElementById("wGl");
let started = false, renderer, scene, camera, pts, lines, uniforms = {};
let raf = 0, t0 = 0;
const COUNT = 120;
const nodes = [];
const edges = [];

const VERT = `
  attribute float aBright;
  attribute float aT;
  varying float vBright;
  varying float vT;
  uniform float uTime;
  void main() {
    vBright = aBright;
    vT = aT;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = (2.2 + aBright * 3.4) * (12.0 / -mv.z);
  }
`;

const FRAG = `
  precision highp float;
  uniform float uTime;
  varying float vBright;
  varying float vT;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r = length(d);
    if (r > 0.5) discard;
    float soft = smoothstep(0.5, 0.05, r);
    float flow = vT + sin(uTime * 0.6 + vT * 8.0) * 0.15;
    vec3 cyan = vec3(0.45, 0.92, 1.0);
    vec3 blue = vec3(0.16, 0.40, 1.0);
    vec3 violet = vec3(0.66, 0.32, 1.0);
    vec3 col = mix(blue, cyan, smoothstep(0.2, 0.9, flow));
    col = mix(col, violet, smoothstep(0.5, 0.0, flow) * 0.9);
    float a = soft * (0.25 + vBright * 0.85);
    gl_FragColor = vec4(col * a, a);
  }
`;

const LINE_VERT = `
  attribute float aT;
  varying float vT;
  void main() {
    vT = aT;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const LINE_FRAG = `
  precision highp float;
  uniform float uTime;
  varying float vT;
  void main() {
    float flow = fract(vT * 1.4 - uTime * 0.12);
    float pulse = smoothstep(0.6, 1.0, flow) * 0.9 + 0.12;
    vec3 cyan = vec3(0.42, 0.88, 1.0);
    vec3 violet = vec3(0.55, 0.35, 1.0);
    vec3 col = mix(violet, cyan, flow);
    gl_FragColor = vec4(col * pulse, pulse * 0.55);
  }
`;

function build() {
  const g = new THREE.Group();
  const rand = (s) => { let x = Math.sin(s * 127.1) * 43758.5453; return x - Math.floor(x); };

  for (let i = 0; i < COUNT; i++) {
    const r = 2.4 + rand(i * 3.1) * 3.6;
    const a = rand(i * 7.7) * Math.PI * 2;
    nodes.push({
      base: new THREE.Vector3(Math.cos(a) * r, (rand(i * 11.3) - 0.5) * 5.2, Math.sin(a) * r * 0.35),
      phase: rand(i * 5.5) * Math.PI * 2,
      bright: 0.25 + rand(i * 13.9) * 0.75
    });
  }

  const pos = new Float32Array(COUNT * 3);
  const br = new Float32Array(COUNT);
  const tt = new Float32Array(COUNT);
  nodes.forEach((n, i) => {
    pos[i * 3] = n.base.x; pos[i * 3 + 1] = n.base.y; pos[i * 3 + 2] = n.base.z;
    br[i] = n.bright;
    tt[i] = (n.base.y + 3) / 6;
  });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("aBright", new THREE.BufferAttribute(br, 1));
  geo.setAttribute("aT", new THREE.BufferAttribute(tt, 1));

  uniforms.uTime = { value: 0 };
  pts = new THREE.Points(geo, new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, uniforms,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
  }));
  g.add(pts);

  // 近邻连线
  const lp = [];
  const lt = [];
  for (let i = 0; i < COUNT; i++) {
    for (let j = i + 1; j < COUNT; j++) {
      const d = nodes[i].base.distanceTo(nodes[j].base);
      if (d < 2.1) {
        lp.push(nodes[i].base.x, nodes[i].base.y, nodes[i].base.z,
                nodes[j].base.x, nodes[j].base.y, nodes[j].base.z);
        lt.push(0, 1);
      }
    }
  }
  const lgeo = new THREE.BufferGeometry();
  lgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(lp), 3));
  lgeo.setAttribute("aT", new THREE.BufferAttribute(new Float32Array(lt), 1));
  lines = new THREE.LineSegments(lgeo, new THREE.ShaderMaterial({
    vertexShader: LINE_VERT, fragmentShader: LINE_FRAG, uniforms,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
  }));
  g.add(lines);
  return g;
}

function resize() {
  const w = canvas.clientWidth || innerWidth, h = canvas.clientHeight || innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function loop() {
  raf = requestAnimationFrame(loop);
  const t = (performance.now() - t0) / 1000;
  uniforms.uTime.value = t;
  pts.rotation.y = t * 0.06;
  lines.rotation.y = pts.rotation.y;
  renderer.render(scene, camera);
}

export function startWelcome3D() {
  if (started || !canvas) return;
  started = true;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }); }
  catch (e) { canvas.style.display = "none"; return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0, 14);

  const g = build();
  g.rotation.z = -0.08;
  scene.add(g);

  resize();
  addEventListener("resize", resize);
  t0 = performance.now();
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) { renderer.render(scene, camera); } else loop();

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { cancelAnimationFrame(raf); raf = 0; }
    else if (!reduce && !raf) { t0 = performance.now() - uniforms.uTime.value * 1000; loop(); }
  });
}

export function stopWelcome3D() { cancelAnimationFrame(raf); raf = 0; }
