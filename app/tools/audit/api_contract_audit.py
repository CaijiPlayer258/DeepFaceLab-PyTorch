# -*- coding: utf-8 -*-
r"""前端 fetch ↔ 后端 OpenAPI 契约对账（常驻审计）。

为什么要有它
  后端 132 个 operation、前端 116 个 (method, path) —— 差值不是"少写了几个接口"，
  而是**协议漂移**：前端照着一份想象中的 API 写按钮，后端改了名 / 删了路由，
  界面上表现为 404 或静默失败。以前靠一次手工审计（app/docs/33-API-CONTRACT.md）
  才发现，之后没人守着就会再漂。这个脚本把那次审计变成可重跑的命令。

四张清单
  A  前端调了、后端没有                  ← **非空即退出码 1**（这就是要挡的漂移）
  B  后端有、前端从没调                  ← 标注三类：验收脚本用 / 资源加载用 / 真死
  C  同路径、method 对不上               ← 非空即退出码 1
  D  前端有 fetch 但 URL 静态解不出来    ← 非空即退出码 2（新的间接调用会被揪出来，
                                          而不是像以前那样被悄悄漏掉）

判据
  · 后端口径：import server.app → app.openapi()（**运行时真值**，不是 AST 扫源码），
    再用 ast 数一遍 @router.* / @app.* 装饰器交叉验证；两边不等 = 有动态注册，
    直接退出码 3，免得后面几轮都在错口径上跑。
  · 前端口径：扫 app/web 下的 fetch(...) 与 new EventSource(...)（跳过 vendor/）
    - 顶层 + 切分：字面量段原样留、表达式段记 {x}；
    - 模板串的插值段同样记 {x}；
    - 局部 var 赋值、模块级常量（var API = "/api/…"）、函数形参回溯调用点都追；
    - 包装函数（face_db.js 的 call/post/del、mask-editor 的 request/get/post、
      app.js 的 trainCtl …）按"函数体里的 fetch 位置"做形参替换后再解析。
  · 匹配规则：段数必须相等；前端的 {x} **只能**匹配后端的参数段，不能吞字面量段
    （否则 /api/merge/stage/{x} 会把 stage/status 一起吞掉）；字面量逐字相等。
  · 资源加载：<img src> / <video src> 这类也算"前端在用"，另记一份"任意 /api 字面量"
    集合，用来把 B 清单里"只被资源加载用"的标出来。

用法（仓库根）
    runtime\server\Scripts\python.exe app\tools\audit\api_contract_audit.py
    ... --quiet           只打 A / C / D（CI 用）
    ... --json out.json   顺手落一份机器可读的

退出码
    0 = A / C / D 全空
    1 = A 或 C 非空（协议漂移）
    2 = D 非空（有静态解不出的 fetch，去补 Resolver 规则）
    3 = 后端自检失败（openapi 与 ast 对不上，或 import 起不来）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VERBS = ("get", "post", "put", "delete", "patch", "head", "options")

# 前端目录里不当"前端代码"看的（第三方库 / 图标 / 字体）
SKIP_PARTS = ("vendor", "fonts", "img")


# ============================================================
# 0) 路径
# ============================================================
def _roots() -> tuple:
    """(仓库根, app 目录)。跟 tools/audit/i18n_audit.py 同一套兜底。"""
    here = Path(__file__).resolve()
    for root in (here.parents[3], here.parents[2]):
        if (root / "app" / "web" / "index.html").is_file():
            return root, root / "app"
    return here.parents[3], here.parents[2]


ROOT, APP = _roots()
WEB = APP / "web"


# ============================================================
# 1) 后端口径
# ============================================================
def _ast_op_count() -> int:
    """ast 数 @router.<verb>(…) / @app.<verb>(…)。"""
    import ast
    n = 0
    for p in (APP / "server").rglob("*.py"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) \
                        and dec.func.attr.lower() in VERBS:
                    n += 1
    return n


def backend_ops() -> tuple:
    """({(METHOD, path)}, ast 计数)。"""
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import server.app as srvapp                       # noqa: WPS433 —— 故意运行时 import
    spec = srvapp.app.openapi()
    ops = set()
    for path, item in (spec.get("paths") or {}).items():
        for k in item:
            if k.lower() in VERBS:
                ops.add((k.upper(), path))
    return ops, _ast_op_count()


# ============================================================
# 2) JS 预处理
# ============================================================
QUOTES = "'\"\x60"


def mask_js(text: str) -> str:
    """注释与字符串内容换成空格（长度 / 换行不变）。

    这样在遮罩文本里搜 fetch( / , / + 不会撞上字符串里的同名子串，
    取参数时仍然读原文。模板串里的插值段对拼 URL 没意义，整体当字面量。
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                i += 2
        elif c in QUOTES:
            q = c
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    out[i] = out[i + 1] = " "
                    i += 2
                    continue
                if text[i] == q:
                    i += 1
                    break
                if text[i] != "\n":
                    out[i] = " "
                i += 1
        elif c == "/" and _regex_here(text, i):
            # 正则字面量：里面的引号/斜杠/花括号都不算 JS 语法。
            # 不认它的话 /[^\s"']/ 里的引号会把后面的代码全吞成字符串，
            # 函数体花括号对不上，整份文件的解析全歪（踩过）。
            out[i] = " "
            i += 1
            in_cls = False
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:
                    out[i] = out[i + 1] = " "
                    i += 2
                    continue
                if ch == "[":
                    in_cls = True
                elif ch == "]":
                    in_cls = False
                elif ch == "/" and not in_cls:
                    out[i] = " "
                    i += 1
                    break
                elif ch == "\n":
                    break
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


_REGEX_PREV = set("(,=:[!&|?{};+-*%~^<>")


def _regex_here(text: str, i: int) -> bool:
    """text[i] == '/' → 它是正则字面量的开头吗（而不是除号）。"""
    j = i - 1
    while j >= 0 and text[j] in " \t\r\n":
        j -= 1
    if j < 0:
        return True
    if text[j] in _REGEX_PREV:
        return True
    head = text[max(0, j - 8):j + 1]
    return bool(re.search(r"\b(return|typeof|case|in|of|delete|void|new)$", head))


def _match_paren(masked: str, i: int) -> int:
    depth, n = 0, len(masked)
    while i < n:
        c = masked[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top(masked: str, start: int, end: int, seps: str) -> list:
    """按顶层分隔符切 [start, end)。"""
    out, depth, s = [], 0, start
    for i in range(start, end):
        c = masked[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c in seps and depth == 0:
            out.append((s, i))
            s = i + 1
    out.append((s, end))
    return out


def call_args(text: str, masked: str, open_paren: int) -> list:
    """text[open_paren] == '(' → 实参表达式原文列表。"""
    close = _match_paren(masked, open_paren)
    if close < 0:
        return []
    return [text[s:e].strip() for s, e in _split_top(masked, open_paren + 1, close, ",")
            if text[s:e].strip()]


def _stmt_end(masked: str, s: int) -> int:
    """从 s 起找语句结束（顶层 ; / 换行 / 顶层闭括号）。"""
    depth, e = 0, s
    while e < len(masked):
        c = masked[e]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif c in ";\n" and depth == 0:
            break
        e += 1
    return e


def _brace_end(masked: str, bs: int) -> int:
    """masked[bs] == '{' → 配对的 '}'。"""
    depth, i = 0, bs
    while i < len(masked):
        if masked[i] == "{":
            depth += 1
        elif masked[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(masked) - 1


def _calls_of(jf, name: str) -> list:
    """name 的所有调用点（左括号下标），排除函数定义处与自身函数体。"""
    body = jf.funcs[name]["body"]
    out = []
    # (?<![\w$.]) —— 别把 Array.prototype.slice.call(…) 当成包装函数 call
    for m in re.finditer(r"(?<![\w$.])" + re.escape(name) + r"\s*\(", jf.masked):
        if body[0] <= m.start() <= body[1]:
            continue
        if re.search(r"\bfunction\s+$", jf.masked[:m.start()]):
            continue
        out.append(m.end() - 1)
    return out


VERB_SET = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}


def _verb_in(args: list) -> str:
    """实参里有没有一个裸的 HTTP 动词字面量（mask-editor 的 request('POST', path)）。"""
    for a in args:
        sm = _STR_RE.match(a.strip())
        if sm and _unquote(a.strip()).upper() in VERB_SET:
            return _unquote(a.strip()).upper()
    return ""


def _split_or(masked: str, expr: str) -> list:
    """顶层 || 切开（空字符串兜底写法 a || b 的两边都算候选）。"""
    out, depth, s, i = [], 0, 0, 0
    while i < len(masked) - 1:
        c = masked[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "|" and masked[i + 1] == "|" and depth == 0:
            out.append(expr[s:i].strip())
            s = i + 2
            i += 1
        i += 1
    out.append(expr[s:].strip())
    return [x for x in out if x] or [expr]


def _split_ternary(expr: str):
    """顶层 a ? b : c → (a, b, c)；不是三元就 None。"""
    mk = mask_js(expr)
    depth, q = 0, -1
    for i, c in enumerate(mk):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "?" and depth == 0:
            q = i
            break
    if q < 0:
        return None
    depth = 0
    for i in range(q + 1, len(mk)):
        c = mk[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ":" and depth == 0:
            return (expr[:q].strip(), expr[q + 1:i].strip(), expr[i + 1:].strip())
        elif c == "?" and depth == 0:
            return None                     # 嵌套三元，不处理
    return None


_STR_RE = re.compile(r"""^(['"\x60])((?:\\.|(?!\1)[\s\S])*)\1$""")
_IDENT_RE = re.compile(r"^[A-Za-z_$][\w$]*$")
_CALL_HEAD_RE = re.compile(r"^([A-Za-z_$][\w$]*)\s*\(")
_METHOD_RE = re.compile(r"""\bmethod\s*:\s*(['"])([A-Za-z]+)\1""")


def _unquote(lit: str) -> str:
    b = lit[1:-1]
    b = b.replace("\\'", "'").replace('\\"', '"')
    if lit[0] == "\x60":
        b = re.sub(r"\$\{[^}]*\}", "{}", b)            # 模板串插值段 → 洞
    return b


def clean_path(tpl: str) -> str:
    """砍 query / hash，从第一个 /api 起算，丢弃外链与非 /api 路径。"""
    t = (tpl or "").split("#", 1)[0].split("?", 1)[0]
    t = re.sub(r"^https?://[^/]+", "", t)
    k = t.find("/api")
    if k < 0:
        return ""
    t = re.sub(r"/{2,}", "/", t[k:])
    # "…/videos" + (dir ? "?dir=" + x : "") 会拼出 /api/merge/videos{}。
    # 这种粘在字面量后面、前面不是 / 的洞属于查询串，去掉。
    t = re.sub(r"(?<=[^/{])\{\}", "", t)
    return t.rstrip("/")


# ============================================================
# 3) 每个文件的模型
# ============================================================
class JsFile:
    def __init__(self, path: Path, rel: str):
        self.path = path
        self.rel = rel
        self.text = path.read_text(encoding="utf-8", errors="replace")
        self.masked = mask_js(self.text)
        self.consts = self._consts()
        self.funcs = self._funcs()

    def line_of(self, pos: int) -> int:
        return self.text.count("\n", 0, pos) + 1

    def _consts(self) -> dict:
        """只管**顶层** var NAME = "…" / A + B（函数里的局部变量交给 local_assign）。"""
        out = {}
        depth = 0
        depth_at = []
        for ch in self.masked:
            depth_at.append(depth)
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        # depth<=1：前端文件普遍被 (function(){…})() 包着，常量在 depth=1。
        # 值必须是字面量或含 + 的拼接（var path = mk(...) 这种不算），
        # 免得多行同名变量顶掉别人。
        for m in re.finditer(r"\b(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=\s*", self.masked):
            if depth_at[m.start()] > 1:
                continue
            expr = self.text[m.end():_stmt_end(self.masked, m.end())].strip()
            if expr and (_STR_RE.match(expr) or "+" in expr):
                out.setdefault(m.group(1), expr)
        return out

    def _funcs(self) -> dict:
        """name → {params, body:(start,end)}，只管 function NAME(…) 声明式。"""
        out = {}
        for m in re.finditer(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", self.masked):
            op = m.end() - 1
            close = _match_paren(self.masked, op)
            if close < 0:
                continue
            bs = self.masked.find("{", close)
            if bs < 0:
                continue
            depth, i = 0, bs
            while i < len(self.masked):
                if self.masked[i] == "{":
                    depth += 1
                elif self.masked[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            out[m.group(1)] = {
                "params": call_args(self.text, self.masked, op),
                "body": (bs, i),
            }
        return out


# ============================================================
# 4) 解析引擎
# ============================================================
class Resolver:
    MAX_DEPTH = 12
    MAX_ALTS = 48          # 一个表达式最多展开多少种拼法（截断了就是漏报）

    def __init__(self):
        self.unresolved = []

    @staticmethod
    def subst(expr: str, params: list, args: list) -> str:
        """形参名换成实参文本（只用于取 method / 拼字面量）。"""
        if not params:
            return expr
        out = expr
        for i, p in enumerate(params):
            if i < len(args):
                out = re.sub(r"\b" + re.escape(p) + r"\b",
                             lambda _m, a=args[i]: "(" + a + ")", out)
        return out

    def local_assign(self, jf: JsFile, pos: int, ident: str):
        """pos 所在的**同一层函数**里，ident 最近一次 var/let/const 赋值。

        只搜 jf.masked[:pos] 会捞到别的函数里的同名变量：app.js 里 var path
        出现过四次，trainCtl 的形参 path 会被 7100 行那个 svg 的 path 顶掉。
        """
        bs = self.scope_at(jf, pos)[2][0]
        best = None
        for m in re.finditer(r"\b(?:var|let|const)\s+" + re.escape(ident) + r"\s*=\s*",
                             jf.masked[bs:pos]):
            best = m
        if not best:
            return None
        s = bs + best.end()
        return jf.text[s:_stmt_end(jf.masked, s)].strip()

    def enclosing_func(self, jf: JsFile, pos: int):
        best = None
        for name, f in jf.funcs.items():
            bs, be = f["body"]
            if bs <= pos <= be and (best is None or bs > best[1]["body"][0]):
                best = (name, f)
        return best

    def sink_of(self, jf: JsFile, name: str):
        """→ (url_expr, opts_expr, 位置) 或 None。"""
        f = jf.funcs[name]
        bs, be = f["body"]
        for pat in (r"\bfetch\s*\(", r"\bnew\s+EventSource\s*\("):
            m = re.search(pat, jf.masked[bs:be])
            if not m:
                continue
            op = bs + m.end() - 1
            args = call_args(jf.text, jf.masked, op)
            return (args[0] if args else "{}",
                    ", ".join(args[1:]) if len(args) > 1 else "", op)
        return None

    # ---- 表达式 → [(模板, 方法提示)] --------------------------------------
    def resolve(self, jf: JsFile, pos: int, expr: str, depth: int = 0) -> list:
        """模板里的 {} 是洞；第二个元素是"这条分支上看到的方法"（可能为空）。

        方法必须**跟着分支走**：同一个 fetch(url, opts) 里的 url 若来自包装函数的
        形参，不同调用点带的方法不同（face_db 的 call 一族就是 POST/DELETE/GET
        三种都走同一条 fetch）。整条链只取一个方法会张冠李戴，
        平白造出 C 清单的假阳性。
        """
        expr = (expr or "").strip()
        if not expr or depth > self.MAX_DEPTH:
            return [("{}", "")]

        if _STR_RE.match(expr):
            return [(_unquote(expr), "")]

        tri = _split_ternary(expr)
        if tri:
            return (self.resolve(jf, pos, tri[1], depth + 1)
                    + self.resolve(jf, pos, tri[2], depth + 1))[:self.MAX_ALTS]

        cm = _CALL_HEAD_RE.match(expr)
        if cm:
            if cm.group(1) in jf.funcs:
                return self.through_func(jf, pos, cm.group(1), expr, depth)
            return [("{}", "")]                 # encodeURIComponent(…) 之类

        if _IDENT_RE.match(expr):
            loc = self.local_assign(jf, pos, expr)      # 同层局部变量优先
            if loc:
                return self.resolve(jf, pos, loc, depth + 1)
            if expr in jf.consts:
                return self.resolve(jf, pos, jf.consts[expr], depth + 1)
            return self.through_param(jf, pos, expr, depth)

        mk = mask_js(expr)
        # a || b：两边都可能被用到，都收（faceThumb 就是 thumb_url || API+"/thumb"）
        ors = _split_or(mk, expr)
        if len(ors) > 1:
            out = []
            for piece in ors:
                out.extend(self.resolve(jf, pos, piece, depth + 1))
            return out[:self.MAX_ALTS] or [("{}", "")]
        spans = _split_top(mk, 0, len(expr), "+")
        if len(spans) > 1:
            acc = [("", "")]
            for s, e in spans:
                subs = self.resolve(jf, pos, expr[s:e].strip(), depth + 1) or [("{}", "")]
                nxt = []
                for a, ha in acc:
                    for b, hb in subs:
                        nxt.append((a + b, ha or hb))
                acc = nxt[:self.MAX_ALTS]
            return acc
        return [("{}", "")]

    def through_func(self, jf: JsFile, pos: int, name: str, expr: str, depth: int) -> list:
        mk = mask_js(expr)
        op = mk.find("(")
        args = call_args(expr, mk, op) if op >= 0 else []
        f = jf.funcs[name]
        sink = self.sink_of(jf, name)
        if sink is None:
            ret = re.search(r"\breturn\b", jf.masked[f["body"][0]:f["body"][1]])
            if not ret:
                return [("{}", "")]
            s = f["body"][0] + ret.end()
            target = jf.text[s:_stmt_end(jf.masked, s)].strip()
        else:
            target = sink[0]
        return self.substitute(jf, pos, target, f["params"], args, depth)

    def substitute(self, jf, pos, expr, params, args, depth) -> list:
        mk = mask_js(expr)
        acc = [("", "")]
        for s, e in _split_top(mk, 0, len(expr), "+"):
            piece = expr[s:e].strip()
            if piece in params:
                idx = params.index(piece)
                piece = args[idx] if idx < len(args) else "{}"
            subs = self.resolve(jf, pos, piece, depth + 1) or [("{}", "")]
            nxt = []
            for a, ha in acc:
                for b, hb in subs:
                    nxt.append((a + b, ha or hb))
            acc = nxt[:self.MAX_ALTS]
        return acc

    def through_param(self, jf: JsFile, pos: int, ident: str, depth: int) -> list:
        """裸标识符是某层函数的形参 → 回到它的每个调用点，代进实参再看一遍。

        匿名函数表达式（window.X = function (p) {…}）回溯不到调用点，
        直接留洞 —— 只要最后的 URL 还落在 /api 上就不算 D。
        """
        name, params, _body = self.scope_at(jf, pos)
        if not name or ident not in params:
            return [("{}", "")]
        f = jf.funcs[name]
        idx = params.index(ident)
        out = []
        for m in _calls_of(jf, name):
            args = call_args(jf.text, jf.masked, m)
            if idx >= len(args):
                continue
            hint = self.method_of(jf, m, args[idx + 1] if idx + 1 < len(args) else "")
            if not hint:
                hint = _verb_in(args)
            for tpl, h in self.resolve(jf, m, args[idx], depth + 1):
                out.append((tpl, h or hint))
        return out or [("{}", "")]

    # ---- 一个 fetch / EventSource 调用点 → [(method, path)] --------------
    def ops_at(self, jf: JsFile, pos: int, callee: str, args: list) -> list:
        url = args[0] if args else "{}"
        paths = self.resolve(jf, pos, url)
        base = "GET" if callee == "EventSource" else \
            self.method_of(jf, pos, args[1] if len(args) > 1 else "")
        opts_expr = (args[1] if len(args) > 1 else "").strip()
        if _IDENT_RE.match(opts_expr):
            # opts 是个形参 = 每个调用点方法不同（face_db 的 call 一族）。
            # 这时再用"第一个调用点的方法"当兜底，会造出 POST /api/merge/faces
            # 这种根本不存在的组合 —— 兜底一律作废，让每条分支自带方法。
            _n, _ps, _b = self.scope_at(jf, pos)
            if opts_expr in _ps:
                base = ""
        out = []
        for tpl, hint in paths:
            cp = clean_path(tpl)
            if cp:
                out.append((hint or base or "GET", cp))
        if not out:
            self.unresolved.append((jf.rel, jf.line_of(pos), url.strip()[:70]))
        return out

    def scope_at(self, jf: JsFile, pos: int):
        """pos 所在最近一层函数 → (名字或 "", 形参, (body 起, body 止))。

        匿名函数表达式（window.X = function (p) {…}）也算一层，名字为空。
        最外层返回 ("", [], (0, 文件长))。
        """
        enc = self.enclosing_func(jf, pos)
        if enc:
            return enc[0], enc[1]["params"], enc[1]["body"]
        k = jf.masked.rfind("function", 0, pos)
        while k >= 0:
            op = jf.masked.find("(", k)
            close = _match_paren(jf.masked, op) if op >= 0 else -1
            if close > 0 and close < pos:
                bs = jf.masked.find("{", close)
                if bs > 0 and bs < pos and _brace_end(jf.masked, bs) >= pos:
                    return ("", call_args(jf.text, jf.masked, op),
                            (bs, _brace_end(jf.masked, bs)))
            k = jf.masked.rfind("function", 0, k)
        return "", [], (0, len(jf.text))

    def method_of(self, jf: JsFile, pos: int, opts_text: str, depth: int = 0) -> str:
        """method 藏在 opts 里的哪一层都能挖出来：

        fetch(url, {method:"POST"})                    直给
        var o = {method:"POST"}; fetch(url, o)         局部变量
        function call(p, o) { fetch(p, o) }  ← call(p, {method:"DELETE"})   形参
        mask-editor 的 request: opts 里装着 method 形参 ← post() → request("POST", …)
        """
        txt = (opts_text or "").strip()
        if not txt or depth > self.MAX_DEPTH:
            return ""
        m = _METHOD_RE.search(txt)
        if m:
            return m.group(2).upper()
        # method: <ident> —— 值本身是变量/形参
        mv = re.search(r"\bmethod\s*:\s*([A-Za-z_$][\w$]*)", txt)
        if mv:
            got = self.value_of(jf, pos, mv.group(1), depth)
            if got:
                return got.upper()
        # 整个 opts 就是一个变量
        if _IDENT_RE.match(txt):
            loc = self.local_assign(jf, pos, txt)
            if loc:
                return self.method_of(jf, pos, loc, depth + 1)
            return self.param_method(jf, pos, txt, depth)
        return ""

    def value_of(self, jf: JsFile, pos: int, ident: str, depth: int):
        """标识符 → 它自己的字面量值（局部 var / 常量），取不到就算了。"""
        loc = self.local_assign(jf, pos, ident)
        if loc is None and ident in jf.consts:
            loc = jf.consts[ident]
        if loc is None:
            return self.param_method(jf, pos, ident, depth)
        sm = _STR_RE.match(loc.strip())
        return _unquote(loc.strip()) if sm else ""

    def param_method(self, jf: JsFile, pos: int, ident: str, depth: int) -> str:
        """ident 是外层函数的形参 → 回溯调用点，取实参里的 method。"""
        name, params, _body = self.scope_at(jf, pos)
        if not name or ident not in params:
            return ""
        idx = params.index(ident)
        for op in _calls_of(jf, name):
            args = call_args(jf.text, jf.masked, op)
            if idx >= len(args):
                continue
            got = self.method_of(jf, op, args[idx], depth + 1)
            if got:
                return got
            verb = _verb_in(args)
            if verb:
                return verb
        return ""


# ============================================================
# 5) 扫前端
# ============================================================
def scan_frontend() -> tuple:
    """({(METHOD, path)}, {/api 字面量}, D 清单)。"""
    rv = Resolver()
    ops, literals = set(), set()
    for p in sorted(WEB.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".js", ".html"):
            continue
        rel = p.relative_to(WEB)
        if any(part in SKIP_PARTS for part in rel.parts) or p.name.endswith(".min.js"):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for lit in re.findall(r"""['"\x60](/api[\w\-./{}:]*)['"\x60]""", text):
            literals.add(lit)
        if p.suffix.lower() != ".js":
            continue
        jf = JsFile(p, rel.as_posix())
        for m in re.finditer(r"\bfetch\s*\(|new\s+EventSource\s*\(", jf.masked):
            callee = "EventSource" if "EventSource" in m.group(0) else "fetch"
            ops.update(rv.ops_at(jf, m.start(), callee,
                                 call_args(jf.text, jf.masked, m.end() - 1)))
        # 资源加载也算"前端在用"：img.src = API + "/thumb?…" 这类
        # （/api/dataset/thumb、/api/merge/video、/api/merge/faces/thumb 全走这条路）
        for m in re.finditer(r"\.src\s*=\s*", jf.masked):
            s = m.end()
            expr = jf.text[s:_stmt_end(jf.masked, s)].strip()
            if not expr:
                continue
            for tpl, _h in rv.resolve(jf, m.start(), expr):
                cp = clean_path(tpl)
                if cp:
                    ops.add(("GET", cp))
    return ops, literals, rv.unresolved


# ============================================================
# 6) 比对
# ============================================================
def segs(path: str) -> list:
    return [s for s in path.strip("/").split("/") if s]


def is_param(seg: str) -> bool:
    return seg.startswith("{") and seg.endswith("}")


def match(f_segs: list, b_segs: list) -> bool:
    """前端模板能否对上后端路径模板。"""
    if len(f_segs) != len(b_segs):
        return False
    for f, b in zip(f_segs, b_segs):
        if f == "{}" or is_param(f):
            if not is_param(b):
                return False
        elif f != b:
            return False
    return True


def reconcile(front_ops: set, back_ops: set, front_lits: set) -> dict:
    back_by_path = {}
    for mv, pth in back_ops:
        back_by_path.setdefault(pth, set()).add(mv)

    A, C, matched = [], [], set()
    for mv, fp in sorted(front_ops):
        fs = segs(fp)
        hit = False
        for bp, meths in back_by_path.items():
            if match(fs, segs(bp)):
                hit = True
                if mv in meths:
                    matched.add((mv, bp))
                else:
                    C.append((mv, fp, bp, sorted(meths)))
        if not hit:
            A.append((mv, fp))

    tools = []
    for p in (ROOT / "app" / "tools").rglob("*.py"):
        try:
            tools.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            pass

    B = []
    for mv, bp in sorted(back_ops):
        if (mv, bp) in matched:
            continue
        bs = segs(bp)
        res = any(match(segs(l), bs) for l in front_lits)
        flat = bp.replace("{", "").replace("}", "")
        users = [] if bp == "/" else sorted({n for n, t in tools if bp in t or flat in t})
        B.append((mv, bp, "resource" if res else "none", users))
    return {"A": A, "B": B, "C": C, "matched": len(matched)}


# ============================================================
# 7) 主程序
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="前端 fetch ↔ 后端 OpenAPI 契约对账")
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--quiet", action="store_true", help="只打 A / C / D")
    args = ap.parse_args()

    back_ops, ast_n = backend_ops()
    if ast_n != len(back_ops):
        print("[契约审计] 后端自检失败：openapi %d 个 operation，ast 数到 %d 个装饰器。"
              % (len(back_ops), ast_n))
        print("           说明有动态注册（add_api_route / 运行时拼前缀），本脚本口径要重写。")
        return 3

    front_ops, front_lits, unresolved = scan_frontend()
    res = reconcile(front_ops, back_ops, front_lits)
    n_paths = len({p for _, p in back_ops})

    print("=" * 78)
    print("前端 fetch ↔ 后端 OpenAPI 契约对账")
    print("  后端：%d 个 path 模板 / %d 个 operation（ast 交叉验证 %d ✓）"
          % (n_paths, len(back_ops), ast_n))
    print("  前端：%d 个 (method, path)（去重后）" % len(front_ops))
    print("=" * 78)

    def dump(title, rows, fmt):
        print("")
        print("%s（%d）" % (title, len(rows)))
        if not rows:
            print("  —— 空 ——")
            return
        for r in rows:
            print("  " + fmt(r))

    dump("A · 前端调了、后端没有", res["A"], lambda r: "%-6s %s" % (r[0], r[1]))
    if not args.quiet:
        dump("B · 后端有、前端从没调", res["B"],
             lambda r: "%-6s %-44s %s%s"
                       % (r[0], r[1],
                          "资源加载" if r[2] == "resource" else "无人调用",
                          ("  ← " + ", ".join(r[3])) if r[3] else ""))
    dump("C · 同路径 method 错配", res["C"],
         lambda r: "前端 %-6s %-40s 后端只有 %s" % (r[0], r[1], "/".join(r[3])))
    dump("D · 前端 fetch 的 URL 静态解不出", unresolved,
         lambda r: "%-26s:%-6d %s" % (r[0], r[1], r[2]))

    # 后端 operation 数 = 前端命中数 + B；前端 (method,path) = 命中 + A + C
    ok_id = len(back_ops) - res["matched"] == len(res["B"])
    print("")
    print("恒等式：后端 %d - 命中 %d = B %d  %s（前端 %d = 命中 %d + A %d + C %d）"
          % (len(back_ops), res["matched"], len(res["B"]), "✓" if ok_id else "✗",
             len(front_ops), res["matched"], len(res["A"]), len(res["C"])))

    if args.json:
        Path(args.json).write_text(json.dumps({
            "backend": {"paths": n_paths, "operations": len(back_ops)},
            "frontend": sorted("%s %s" % (m, p) for m, p in front_ops),
            "A": res["A"], "B": res["B"], "C": res["C"], "D": unresolved,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已写：%s" % args.json)

    if unresolved:
        print("\n[契约审计] D 非空：有 fetch 的 URL 静态解不出来 —— 要么是新写法，"
              "要么是真漏报。去 Resolver 里补规则，别放过。")
        return 2
    if res["A"] or res["C"]:
        print("\n[契约审计] A / C 非空：前端与后端协议漂移了。")
        return 1
    print("\n[契约审计] A / C / D 全空，协议对齐。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
