# 12 · 人脸数据库（后端）

合成工作室的「人脸数据库」：一个**合成项目**（一个视频）一个库。前端下一步只消费
这里定义的 JSON，不需要知道磁盘布局。

实现：

| 文件 | 职责 |
| --- | --- |
| `app/server/_faces.py` | 存储层：index、文件锁、原子写、组、回收站、缩略图、DFM 输入尺寸 |
| `app/server/routes_faces.py` | HTTP 层：`/api/merge/faces/*`，参数校验 + 复用检测/对齐内核 |
| `app/server/app.py` | 注册路由；`_no_cache` 中间件对缩略图放行缓存头 |

## 磁盘布局

```
<workspace>/merge/<视频名>/
    _faces/
        index.json          # 唯一清单
        index.json.bak      # 上一次成功写入的清单（只留一份）
        f_0001.png          # 统一尺寸 PNG（512 或该 DFM 的输入尺寸）
        f_0002.png
        _trash/             # 回收站：trashed=true 的文件搬到这里
        _thumb/             # 128×128 JPEG 缩略图缓存
        .lock               # 跨进程文件锁
```

## 数据模型

```json
{
  "version": 1,
  "next_face": 7,
  "next_group": 3,
  "groups": [
    {"id": "g1", "dfm": "张元英", "name": "张元英", "size": 384},
    {"id": "g2", "dfm": "阿証-416", "name": "阿証-416", "orphan": true, "size": 416}
  ],
  "faces": [
    {
      "id": "f_0001",
      "file": "f_0001.png",
      "group": "g1",
      "source": {"video": "xx.mp4", "frame": 1234},
      "added_at": 1794063308.806,
      "trashed": false
    }
  ]
}
```

约定：

* `face.file` 只存**基名**。`trashed=false` 时在 `_faces/`，
  `trashed=true` 时在 `_faces/_trash/` —— trash / restore 只翻转 `trashed`
  并搬文件，`id` / `file` 不变，前端不用重新对 id。
* `face.group` 可以是空串：刚提取、还没投放的脸 = **未分组**。
* `group.id` 一旦分配永不回收。`dfm` 是组的身份（大小写变体复用同一个
  id，只更新显示名）。不再选中的组如果还有脸，`orphan=true` 保留；没脸才消失。
* `group.size` 是 DFM 的 ONNX 输入边长（384 / 416 / 512…）。找不到
  `workspace/model/<dfm>.dfm` 时按 512；前端组头直接显示 `size×size`。
* `next_face` / `next_group` 是单调计数器（删除后 id 也不复用）。
* 组/脸只存元数据，**不复制文件**：跨组投放只改 `face.group`。

## API

所有接口都挂 `/api/merge/faces`。成功是 `{ok:true, ...}`；失败是
`{ok:false, error, detail}` + 对应 HTTP 状态码。`project` 是项目目录
（绝对路径，或相对 4.1 仓库根）。

### GET `/api/merge/faces?project=<项目目录>`

返回整库：

```json
{
  "ok": true,
  "project": "D:/.../workspace/merge/xx",
  "groups": [{"id": "g1", "dfm": "张元英", "name": "张元英", "orphan": false, "size": 384}],
  "faces": [{"id": "f_0001", "file": "f_0001.png", "group": "g1",
             "source": {"video": "xx.mp4", "frame": 1234},
             "added_at": 1794063308.806, "trashed": false,
             "thumb_url": "/api/merge/faces/thumb?project=...&id=f_0001"}],
  "stats": {"total": 1, "active": 1, "trashed": 0, "groups": 1,
            "grouped": 1, "ungrouped": 0, "orphan_groups": 0,
            "orphan_faces": 0, "next_face": 2}
}
```

`faces` 包含回收站里的脸（`trashed=true`），前端自己按 flag 分栏。

### POST `/api/merge/faces/extract`

```json
{"project": "D:/.../merge/xx", "frame": 1234, "dfm": "张元英"}
```

从**当前帧**检测并裁出**所有**人脸，每张一条记录：

* 帧图优先读项目里的 `000000.jpg…`（`frame` 就是帧号）；
  没有帧图就回退到项目里的源视频按帧号 seek；项目目录不存在时**自动创建**
  （只在工作区里建，越界 400），响应里带 `"created": true`。
* 检测复用 `app._get_detector()`；对齐走 **68 点仿射**：检测器给 5 点 → 2D106
  回归 68 点（`routes_merge._frame_landmarks(...)`，只认模型输出）→
  `_align.align_face_68()`（68 对点最小二乘相似变换 + INTER_CUBIC）。
  **拿不到 68 点不退回 5 点**：这张脸跳过，全帧都拿不到时回
  `{"ok": false, "reason": "no_landmarks68"}`。
* 新记录带 `"align": "68"`；老库里没有这个字段的按 5 点对齐看待（尺寸一样，
  对齐略有差异，不回溯改写）。
* 一帧多脸：检测器返回几张就存几张（各自独立对齐），不合并、不取最大。
* 尺寸：在 `workspace/model/<dfm>.dfm` 找到就读 ONNX 输入张量的 H/W
  （384 / 416 / 512…），找不到就 512。
* `dfm` 只认模型名，不能带路径。给了 `dfm` 就自动归到同名组
  （没有就新建）；不给就是未分组。

成功：

```json
{"ok": true, "frame": 1234, "size": 384, "dfm": "张元英", "group": "g1",
 "count": 2, "detected": 2,
 "faces": [{"id": "f_0003", "file": "f_0003.png", "group": "g1",
            "source": {"video": "xx.mp4", "frame": 1234},
            "added_at": 1794063309.1, "trashed": false,
            "thumb_url": "/api/merge/faces/thumb?...",
            "thumb": "data:image/jpeg;base64,...",
            "box": [x1, y1, x2, y2], "score": 0.99}]}
```

**没有人脸不是错误**：HTTP 200 + `{"ok": false, "reason": "no_face", "detected": 0}`。

### POST `/api/merge/faces/add`

点「＋」走这条：**已经知道是哪张脸**，不用再整帧重检测。

```json
{"project": "D:/.../merge/xx",          // 可省略：省略就用默认库 <工作区>/merge
 "kps68": [[x, y], ... 68 个],          // /api/merge/detect 每张脸都带 kps68
 "box": [x1, y1, x2, y2],               // 可选，只进元数据
 "dfm": "张元英", "frame": 1234,
 "source": {"video": "xx.mp4", "frame": 1234}}
```

* 裁剪和 extract 是**同一条** 68 点仿射，所以同一帧同一张脸，两条路落下来的
  PNG 逐像素相同（尺寸也一样）。
* `kps` 只有 5 点 / 不是 68×2 → HTTP 200 + `{"ok": false, "reason":
  "no_landmarks68", "detail": "...", "expect": "68x2"}`，**不落任何文件**。
* 整帧原图来源：项目帧图 > `frame_path` > `frame_b64` > 合成器里打开着的素材。
* `project` 省略 / 目录不存在 → 用默认库 `<工作区>/merge`（`_faces/` +
  `index.json` 自动建），响应 `"created": true`；越界到工作区外一律 400。

```json
{"ok": true, "project": "...", "created": true, "dfm": "张元英", "size": 384,
 "align": "68", "count": 1,
 "faces": [{"id": "f_0001", ..., "kps68": [[x, y], ...]}]}
```

### POST `/api/merge/faces/move`

`{"project": "...", "ids": ["f_0001", "f_0002"], "group": "g2"}`
（`group: ""` = 移回未分组）。只改 `face.group`，文件一个字节都不动。

### POST `/api/merge/faces/trash` / `restore`

`{"project": "...", "ids": ["f_0001"]}`

trash：文件搬到 `_faces/_trash/`，index 里 `trashed=true`（重复 trash 幂等）。
restore：搬回 `_faces/`，`trashed=false`（重复 restore 幂等）。
回收站里的脸仍能取缩略图。

### DELETE `/api/merge/faces`

`{"project": "...", "ids": ["f_0001"]}`

彻底删除：PNG、`_trash/` 里的副本、`_thumb/<id>.jpg`、index 记录一起清掉。

### POST `/api/merge/faces/groups`

`{"project": "...", "dfms": ["张元英", "阿証-416"]}`

同步组（幂等）：同名 dfm 复用 id；不再选的组有脸 → `orphan=true` 保留，
没脸 → 删组。返回和 GET 一样的 `groups / faces / stats`。

### GET `/api/merge/faces/thumb?project=<项目目录>&id=f_0001`

128×128 JPEG。首次生成后缓存到 `_faces/_thumb/<id>.jpg`，之后不再解码原图。
响应带 `Cache-Control: public, max-age=604800, immutable` 和 `ETag`；
带 `If-None-Match` 命中时回 304。

## 错误与安全

* 400：参数不合法；`project` / `ids` / `group` / `dfm` 里出现
  `..`、绝对路径、盘符；`ids` 不是 `f_0001` 这种形态。
* 404：项目目录/人脸 id/组/文件不存在；帧号越界。
* 409：还没有帧也没有源视频；restore 目标已存在。
* 503：等文件锁超时（另一个进程正在长时间写）。
* 500：index.json 坏到连 `index.json.bak` 也读不出来。

所有库内路径都必须 relative_to(`<项目>/_faces`)：index 里的 `file` 只允许
基名，请求里的 id 只允许 `f_\d{4,}`，绝对路径与 `..` 一律 400。

## 并发 / 落盘

* 一个项目一把锁：进程内 `RLock` + 跨进程 `_faces/.lock`
  （Windows `msvcrt.locking` / POSIX `flock`，拿不到重试 15s）。
* 写文件：同目录 `<name>.part` → `os.replace`；index 替换前先把旧内容
  写成 `index.json.bak`。
* 并发 10 个 move：最后 index 一定是完整 JSON，脸数量守恒。

## 验收

临时目录里造项目（不碰 workspace 真实数据）：

```bat
runtime\server\Scripts\python.exe app\tools\test_faces_api.py
```

脚本会打印 PASS/FAIL，退出码 0 = 全过。
