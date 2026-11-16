#!/usr/bin/env node
/* 校验 app/web/js/app/*.js 拼接后仍是合法 JS。
 *
 * 为什么需要它：外层 IIFE 跨文件（分片单独 node --check 必然报错），
 * 所以改完任何一个分片，跑这个脚本 —— 它按文件名顺序拼接后交给 vm.Script 编译，
 * 语法错会带行号（行号是拼接后的，再按下面打印的分片偏移换算回具体文件）。
 *
 *   node app/tools/js/check_bundle.js
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const dir = path.join(__dirname, "..", "..", "web", "js", "app");
const files = fs.readdirSync(dir).filter(f => f.endsWith(".js")).sort();
let src = "";
const spans = [];
for (const f of files) {
  const body = fs.readFileSync(path.join(dir, f), "utf8");
  const from = src.split("\n").length;
  src += body + "\n";
  spans.push({ f, from, to: src.split("\n").length - 1 });
}
try {
  new vm.Script(src, { filename: "app.bundle.js" });
  console.log("OK  " + files.length + " 个分片 · " + src.split("\n").length + " 行 · 语法通过");
} catch (e) {
  const m = /app\.bundle\.js:(\d+)/.exec(String(e.stack || ""));
  const line = m ? Number(m[1]) : 0;
  const hit = spans.find(s => line >= s.from && line <= s.to);
  console.error("FAIL " + e.message);
  if (hit) console.error("     拼接行 " + line + " → " + hit.f + " 第 " + (line - hit.from + 1) + " 行");
  process.exit(1);
}
