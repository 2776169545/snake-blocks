/* =====================================================================
 * snake_ai.js —— 强化学习蛇策略（移植进游戏本体的彩蛋）
 *
 * 用法：在同源相噬关卡里按 Tab 打开秘笈面板，输入 workbuddy 回车，
 *      AI 即接管方向盘代打。玩家按任意方向键/WASD 即刻夺回，AI 不抢。
 *      再次输入 workbuddy 交还给 AI；输入 off 一并关闭。
 *
 * 依赖两个外接文件（顺序有讲究，见 snake-blocks.html 尾部）：
 *   1) snake_ai_model.js  —— 纯权重（base64 float32），必须先加载
 *   2) snake_ai.js        —— 本文件，纯逻辑
 *
 * 设计约束（都是踩过的坑，动手前先读）：
 *   · 整个文件包在一个 IIFE 里。全局只暴露 window.SNAKE_AI。
 *     原因：游戏逻辑段有 166 个顶层 const，不包 IIFE 一旦撞名就是
 *     SyntaxError，整个文件崩掉，游戏直接白屏。
 *   · 绝不引用任何全局标识符（S/DIRS/step/... 一律不碰），状态只经
 *     attach(SB) 传入的 SB.* 读。这样浏览器和 node 两端能共用同一份代码。
 *   · 动作只通过 SB.setDir(方向字符串) 下发 —— 那是游戏自己的玩家输入
 *     入口（键盘/触摸也都走它），所以 AI 和人类完全同一条代码路径，
 *     游戏规则一个字节都没改，而且玩家随时能按键抢回方向盘。
 *   · 决策必须发生在 step() 内部、队列消费之前（见 attach 里的 step 补丁）。
 *     setDir 是队列制，晚一格就等于走进 biteWarn 刚判危险的那一格 —— 那是
 *     系统性劣化，不是随机劣化。
 *   · 方向序 ACTIONS = ['up','left','down','right']，与训练环境 env-server.js
 *     的 DIRV 顺序一致。⚠ 游戏本体 DIRS 的键序是 up/down/left/right，
 *     两者不同！按 DIRS 键序转会静默错位，AI 变成专往一个方向转。
 *
 * 维度说明：OBS = 380（游戏原生观测）+ 36（训练期增广的 aux）= 416
 * ===================================================================== */
;(function (root) {
'use strict';

/* ---------------------------------------------------------------- 段1 常量 */

/** 观测维度。改这里之前先确认与模型 meta.sizes[0] 一致，否则直接拒绝加载。 */
var OBS_DIM = 416;

/** 动作序 —— 必须与训练环境一致，⚠ 不要改成游戏 DIRS 的键序 */
var ACTIONS = ['up', 'left', 'down', 'right'];

/** 位移向量，与 ACTIONS 同序。0=上 1=左 2=下 3=右 */
var DIRV = [[0, -1], [-1, 0], [0, 1], [1, 0]];

/** 人类按键 → 方向。本地副本，不依赖游戏的 DIRKEY，避免耦合。 */
var HUMAN_KEYS = {
  ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
  w: 'up', s: 'down', a: 'left', d: 'right',
  W: 'up', S: 'down', A: 'left', D: 'right'
};

/* ---------------------------------------------------------------- 段2 前向 */

/**
 * base64 → Float32Array。
 *
 * ⚠ 关键：必须先解到独立的一份 Uint8Array，再 reinterpret 成 Float32Array。
 *   直接 new Float32Array(buf.buffer) 在 node 下会读到错位数据 —— Buffer
 *   的底层 ArrayBuffer 常带 8 字节池偏移（byteOffset ≠ 0）。那种错误不抛异常，
 *   只是算出一堆垃圾 Q 值，排查起来极其痛苦。这里显式从 0 偏移新建缓冲区。
 */
function decodeB64(s, count) {
  var bin = typeof atob === 'function'
    ? atob(s)                                   // 浏览器
    : Buffer.from(s, 'base64').toString('binary');  // node
  var bytes = new Uint8Array(count * 4);
  for (var i = 0; i < bytes.length; i++) bytes[i] = bin.charCodeAt(i) & 0xff;
  // 单独 new ArrayBuffer(n) 保证 byteOffset === 0
  var ab = new ArrayBuffer(bytes.length);
  var u8 = new Uint8Array(ab);
  u8.set(bytes);
  return new Float32Array(ab);
}

var NET = null;   // 载入后的网络 {W:[...], B:[...], sizes, tag}

/**
 * 载入 window.SNAKE_AI_W 权重。返回 true/false，不抛异常。
 * 失败时游戏照常可玩，只是 AI 不可用 —— 彩蛋不该有能力把游戏搞崩。
 */
function loadModel() {
  var m = (root.SNAKE_AI_W || root.window && root.window.SNAKE_AI_W);
  if (!m) return fail('找不到 snake_ai_model.js（权重没加载）');
  if (!m.meta || !m.meta.sizes) return fail('权重 meta 缺失');
  if (m.meta.sizes[0] !== OBS_DIM) {
    return fail('模型输入维度 ' + m.meta.sizes[0] + ' ≠ 代码的 ' + OBS_DIM + '，已拒绝加载');
  }
  var sizes = m.meta.sizes, L = sizes.length - 1, W = [], B = [], l;
  try {
    for (l = 0; l < L; l++) {
      W.push(decodeB64(m['W' + l], sizes[l] * sizes[l + 1]));
      B.push(decodeB64(m['B' + l], sizes[l + 1]));
    }
  } catch (e) {
    return fail('权重解码失败：' + (e && e.message));
  }
  /* 解码自检：复现 Python 侧写下的 probe 值。
     对不上就说明解码路径错了（几乎必然是 buffer 偏移问题），宁可不加载。 */
  if (m.probe && m.probe.W0head && W[0]) {
    for (var i = 0; i < m.probe.W0head.length; i++) {
      if (Math.abs(W[0][i] - m.probe.W0head[i]) > 1e-6) {
        return fail('权重解码对拍失败（W0[' + i + ']: 期望 ' + m.probe.W0head[i] +
                    ' 实得 ' + W[0][i] + '），多半是字节偏移问题');
      }
    }
  }
  NET = { W: W, B: B, sizes: sizes, tag: m.meta.tag || '?', obsDim: sizes[0] };
  buf.h1 = new Float32Array(sizes[1]);
  buf.h2 = new Float32Array(sizes[2]);
  buf.q = new Float32Array(sizes[sizes.length - 1]);
  return true;
}

var loadError = '';
function fail(msg) { loadError = msg; return false; }

/** 前向的常驻中间缓冲，维度在 loadModel 时按模型大小分配 */
var buf = { h1: null, h2: null, q: null };

/**
 * 前向：x（416 维）→ 4 个 Q 值。
 * 布局照 train.py L238-248：Z = X @ W + B，ReLU 只施加在中间层，输出层线性。
 * W[l] 形状 (fan_in, fan_out)，所以累加是 out[j] += x[i] * W[i*out_n + j]。
 *
 * @param {ArrayLike<number>} x 长度必须 ≥ NET.sizes[0]
 * @param {Float32Array} [q] 可复用的输出缓冲（省每步分配）
 * @returns {Float32Array} 长度 4
 */
function forward(x, q) {
  var sizes = NET.sizes;
  var nIn = sizes[0], nHid1 = sizes[1], nHid2 = sizes[2], nOut = sizes[3];
  var W0 = NET.W[0], B0 = NET.B[0], W1 = NET.W[1], B1 = NET.B[1],
      W2 = NET.W[2], B2 = NET.B[2];
  /* 中间层缓冲常驻复用：每 116ms 一次推理，没必要每次 new */
  var h1 = buf.h1, h2 = buf.h2;
  var out = q || buf.q;
  var i, j;
  for (j = 0; j < nHid1; j++) h1[j] = B0[j];
  for (i = 0; i < nIn; i++) {
    var xi = x[i], row = i * nHid1;
    if (xi === 0) continue;              // 稀疏观测，省不少乘加
    for (j = 0; j < nHid1; j++) h1[j] += xi * W0[row + j];
  }
  for (j = 0; j < nHid1; j++) if (h1[j] < 0) h1[j] = 0;    // ReLU

  for (j = 0; j < nHid2; j++) h2[j] = B1[j];
  for (i = 0; i < nHid1; i++) {
    var hi = h1[i]; if (hi === 0) continue;
    var r1 = i * nHid2;
    for (j = 0; j < nHid2; j++) h2[j] += hi * W1[r1 + j];
  }
  for (j = 0; j < nHid2; j++) if (h2[j] < 0) h2[j] = 0;    // ReLU

  for (j = 0; j < nOut; j++) out[j] = B2[j];                // 末层线性，无激活
  for (i = 0; i < nHid2; i++) {
    var gi = h2[i]; if (gi === 0) continue;
    var r2 = i * nOut;
    for (j = 0; j < nOut; j++) out[j] += gi * W2[r2 + j];
  }
  return out;
}

/* ---------------------------------------------------------------- 段3 观测

 * 全部移植自训练侧，逐行对照：
 *   · observe380()  ← duel-train.html L3293-3325（游戏原生观测）
 *   · aux36()       ← env-server.js L409-460 + train.py L186-210（拼接顺序）
 *
 * 这里是移植最容易出错的地方 —— 任何一维拼错都不会报错，只会让 AI 变笨。
 * 三条铁律：
 *   1. 负分块 neg 是**毒**（玩家 shrink 变短），只有 block/rarity 是食物。
 *      早期版本把 neg 也算食物，等于教蛇去吃让自己变短的东西。
 *   2. S.grow 是观测里**没有的隐藏状态**：同一个视觉状态（尾格空着），
 *      grow>0 时走上去是死、grow=0 时是活。必须直接读 S.grow 来定 bodyEnd。
 *   3. 追尾可达的目标是**原尾当空地可穿透**（它下一拍会让位），
 *      但 fatal 检查范围 grow>0 时要含原尾。
 */

/** 7×7 视野的 6 个通道，与 duel-train.html L3309-3319 同序 */
var GRID_N = 7, GRID_HALF = 3;

/** BFS 用的定长工作区。64×64 远大于 20×20 棋盘，够用。 */
var bfsDist = new Int16Array(64 * 64), bfsQ = new Int32Array(64 * 64),
    bfsFirst = new Int8Array(64 * 64);
var bfsDistN = new Int16Array(64 * 64), bfsQN = new Int32Array(64 * 64);
var bfsDistR = new Int16Array(64 * 64), bfsQR = new Int32Array(64 * 64);
var trSeen = new Uint8Array(64 * 64), trQ = new Int32Array(64 * 64);
var cycIdx = new Int16Array(64 * 64);
var cycPath = null;

/**
 * 游戏原生观测：380 维。结构照 duel-train.html L3295-3325，一个字没改。
 *   [0..1]    蛇头归一化坐标
 *   [2..49]   自己身体 24 节 × 2（缺节填 -1,-1）
 *   [50..81]  蚀蛇身体 16 节 × 2
 *   [82..375] 7×7 视野 × 6 通道
 *   [376..379] body.len/40, rlen/40, len/(len+rlen), 饥饿度/5
 */
function observe380(S, out) {
  var v = out || obs.buf380;
  var p = 0, i, k;
  var h = SB.headOf() || { x: 0, y: 0 };
  var W = S.w, H = S.h;
  v[p++] = h.x / W; v[p++] = h.y / H;
  var body = S.body;
  for (i = 0; i < 24; i++) {
    var b = body[i];
    if (b) { v[p++] = b.x / W; v[p++] = b.y / H; }
    else { v[p++] = -1; v[p++] = -1; }
  }
  var r = S.rivals[0];
  var rAlive = !!(r && r.alive);
  for (i = 0; i < 16; i++) {
    var e = rAlive ? r.body[i] : null;
    if (e) { v[p++] = e.x / W; v[p++] = e.y / H; }
    else { v[p++] = -1; v[p++] = -1; }
  }
  /* 7×7 局部视野：6 通道 one-hot，出界/空地/墙/毒/星屑/其它正数。
     ⚠ 每格必须恰好 push 6 个。先按原游戏写法填进临时数组再整体 push，
     顺序 push 极易漏 —— 分支多的时候漏一个不会报错，只会让维度错位。 */
  var o = obs._o;
  for (var gy = -GRID_HALF; gy <= GRID_HALF; gy++) {
    for (var gx = -GRID_HALF; gx <= GRID_HALF; gx++) {
      var x = h.x + gx, y = h.y + gy;
      o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0; o[4] = 0; o[5] = 0;
      if (!SB.inB(x, y)) o[0] = 1;
      else {
        var c = S.cells[SB.idx(x, y)];
        if (!c) o[1] = 1;
        else if (c.k === 'wall') o[2] = 1;
        else if (c.k === 'neg') o[3] = 1;
        else if (c.rarity) o[4] = 1;
        else o[5] = 1;
      }
      for (var kk = 0; kk < 6; kk++) v[p++] = o[kk];
    }
  }
  var rlen = rAlive ? r.body.length : 0;
  v[p++] = S.body.length / 40;
  v[p++] = rlen / 40;
  v[p++] = S.body.length / Math.max(1, S.body.length + rlen);
  /* ⚠ 这一维读的是 r.starve 而不是 r.hunger —— 后者在整个代码库里从未被赋值，
     恒为 0。r.starve 是真实的饥饿计数（游戏 L1443 赋值、L889/L943 读），
     starve>0 会让蚀蛇强制降速到 0.8×，是重要战术信号。 */
  v[p++] = rAlive ? (r.starve || 0) / 5 : 0;
  obs.dim380 = p;
  return p;   /* 应恒为 380 */
}

/** BFS：蛇头 → 最近**正数食物**。neg 不是食物（是毒）。照 env-server.js L77-105。 */
function bfsToFood(S) {
  var h = S.body[0];
  if (!h) return { dist: -1, dir: -1, fx: 0, fy: 0 };
  var W = S.w, H = S.h, cells = S.cells;
  bfsDist.fill(-1); bfsFirst.fill(-1);
  var start = h.y * W + h.x;
  bfsDist[start] = 0;
  var qs = 0, qe = 0, k;
  bfsQ[qe++] = start;
  while (qs < qe) {
    var i = bfsQ[qs++], x = i % W, y = (i / W) | 0, d = bfsDist[i];
    var c = cells[i];
    /* block 含 rarity 星屑，都是涨节 */
    if (i !== start && c && c.k === 'block') return { dist: d, dir: bfsFirst[i], fx: x, fy: y };
    for (k = 0; k < 4; k++) {
      var nx = x + DIRV[k][0], ny = y + DIRV[k][1];
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      var j = ny * W + nx;
      if (bfsDist[j] >= 0) continue;
      var cc = cells[j];
      if (cc && cc.k === 'wall') continue;
      bfsDist[j] = d + 1;
      bfsFirst[j] = (i === start) ? k : bfsFirst[i];
      bfsQ[qe++] = j;
    }
  }
  return { dist: -1, dir: -1, fx: h.x, fy: h.y };
}

/** BFS：蛇头 → 最近 neg（毒）的距离。照 env-server.js L110-135。 */
function bfsToNeg(S) {
  var h = S.body[0];
  if (!h) return -1;
  var W = S.w, H = S.h, cells = S.cells;
  bfsDistN.fill(-1);
  var start = h.y * W + h.x;
  bfsDistN[start] = 0;
  var qs = 0, qe = 0, k;
  bfsQN[qe++] = start;
  while (qs < qe) {
    var i = bfsQN[qs++], x = i % W, y = (i / W) | 0, d = bfsDistN[i];
    var c = cells[i];
    if (i !== start && c && c.k === 'neg') return d;
    for (k = 0; k < 4; k++) {
      var nx = x + DIRV[k][0], ny = y + DIRV[k][1];
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      var j = ny * W + nx;
      if (bfsDistN[j] >= 0) continue;
      var cc = cells[j];
      if (cc && cc.k === 'wall') continue;
      bfsDistN[j] = d + 1;
      bfsQN[qe++] = j;
    }
  }
  return -1;
}

/** BFS：蚀蛇头 → 最近口粮（neg 或 rarity）。照 env-server.js L364-390。 */
function bfsRivalToFood(S) {
  var r = S.rivals[0];
  var h = (r && r.alive) ? r.body[0] : null;
  if (!h) return { dist: -1, fx: 0, fy: 0 };
  var W = S.w, H = S.h, cells = S.cells;
  bfsDistR.fill(-1);
  var start = h.y * W + h.x;
  bfsDistR[start] = 0;
  var qs = 0, qe = 0, k;
  bfsQR[qe++] = start;
  while (qs < qe) {
    var i = bfsQR[qs++], x = i % W, y = (i / W) | 0, d = bfsDistR[i];
    var c = cells[i];
    if (i !== start && c && (c.k === 'neg' || c.rarity)) return { dist: d, fx: x, fy: y };
    for (k = 0; k < 4; k++) {
      var nx = x + DIRV[k][0], ny = y + DIRV[k][1];
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      var j = ny * W + nx;
      if (bfsDistR[j] >= 0) continue;
      var cc = cells[j];
      if (cc && cc.k === 'wall') continue;
      bfsDistR[j] = d + 1;
      bfsQR[qe++] = j;
    }
  }
  return { dist: -1, fx: h.x, fy: h.y };
}

/**
 * 自咬预警：4 个方向各一维，会不会咬到自己。
 * 照 env-server.js L141-154，精确复刻游戏 L2828/L2847 的语义：
 * willGrow = S.grow>0；无信贷时排除旧尾（本拍让位），有信贷时整条身体都算。
 */
function selfBiteWarn(S, out) {
  var r0 = out || obs._bw;
  var h = S.body[0];
  r0[0] = 0; r0[1] = 0; r0[2] = 0; r0[3] = 0;
  if (!h) return r0;
  var willGrow = (S.grow || 0) > 0 ? 1 : 0;
  var n = S.body.length - (willGrow ? 0 : 1);
  for (var d = 0; d < 4; d++) {
    var nx = h.x + DIRV[d][0], ny = h.y + DIRV[d][1];
    for (var i = 0; i < n; i++) {
      if (S.body[i].x === nx && S.body[i].y === ny) { r0[d] = 1; break; }
    }
  }
  return r0;
}

/** 撞蚀蛇预警：4 个方向各一维，会不会撞上蚀蛇身体（含头）。照 env-server.js L230-241。 */
function rivalBiteWarn(S, out) {
  var r0 = out || obs._rb;
  var h = S.body[0], r = S.rivals[0];
  r0[0] = 0; r0[1] = 0; r0[2] = 0; r0[3] = 0;
  if (!h || !r || !r.alive || !r.body.length) return r0;
  for (var d = 0; d < 4; d++) {
    var nx = h.x + DIRV[d][0], ny = h.y + DIRV[d][1];
    for (var i = 0; i < r.body.length; i++) {
      if (r.body[i].x === nx && r.body[i].y === ny) { r0[d] = 1; break; }
    }
  }
  return r0;
}

/**
 * 追尾可达判据（「贴着自己边走」的形式化）。照 env-server.js L172-220。
 * 障碍 = 新头 + 原 body[0..L-2]；目标 = 原尾当**空地**可穿透。
 * fatal 检查范围 bodyEnd = willGrow ? L : L-1。
 * 够得到尾 ⇒ 永远不会被自己围死。这就是把可达性不变量直接喂给网络。
 */
function tailReachOf(S, out) {
  var o = out || obs._tr;
  var h = S.body[0], W = S.w, H = S.h, cells = S.cells;
  var L = S.body.length;
  var willGrow = (S.grow || 0) > 0 ? 1 : 0;
  var tail = S.body[L - 1];
  var bodyEnd = willGrow ? L : (L - 1);
  var minArea = 400, d, i, k;
  for (d = 0; d < 4; d++) o.reach[d] = 0;
  if (!h) { o.minArea = 0; return o; }
  for (d = 0; d < 4; d++) {
    var nx = h.x + DIRV[d][0], ny = h.y + DIRV[d][1];
    if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
    var fatal = false;
    for (i = 0; i < bodyEnd; i++) {
      if (S.body[i].x === nx && S.body[i].y === ny) { fatal = true; break; }
    }
    if (fatal) continue;
    trSeen.fill(0);
    for (i = 0; i <= L - 2; i++) {
      var pp = S.body[i];
      if (trSeen[pp.y * W + pp.x] !== 3) trSeen[pp.y * W + pp.x] = 2;
    }
    trSeen[ny * W + nx] = 2;
    var start = ny * W + nx;
    trSeen[start] = 1;
    var qs = 0, qe = 0, area = 0, hitTail = 0;
    trQ[qe++] = start;
    while (qs < qe) {
      var ii = trQ[qs++], x = ii % W, y = (ii / W) | 0;
      area++;
      if (x === tail.x && y === tail.y) hitTail = 1;
      for (k = 0; k < 4; k++) {
        var ax = x + DIRV[k][0], ay = y + DIRV[k][1];
        if (ax < 0 || ay < 0 || ax >= W || ay >= H) continue;
        var j = ay * W + ax;
        if (trSeen[j] === 1 || trSeen[j] === 2) continue;
        var c = cells[j];
        if (c && c.k === 'wall') { trSeen[j] = 3; continue; }
        trSeen[j] = 1;
        trQ[qe++] = j;
      }
    }
    if (area < minArea) minArea = area;
    o.reach[d] = hitTail ? 1 : 0;
  }
  o.minArea = minArea / 400;
  return o;
}

/**
 * Hamiltonian 环（兜底路线参考）。照 env-server.js L293-332。
 * 前提：墙只放内圈（游戏 buildTerrain L2487），最外圈永无墙 → 外环必成。
 * 外环 + 内层蛇形扩展，构造失败就停在已建成部分（永不空手）。
 * 缓存：每局建一次（墙布局不变）。若 cycPath 为 null 或与当前局不符则重建。
 */
function cycleBuild(S) {
  var W = S.w, H = S.h, cells = S.cells;
  cycIdx.fill(-1);
  var path = obs._cycPath || (obs._cycPath = []);
  path.length = 0;
  var isWall = function (x, y) { var c = cells[y * W + x]; return !!(c && c.k === 'wall'); };
  var push = function (x, y) {
    if (x < 0 || y < 0 || x >= W || y >= H || isWall(x, y)) return false;
    cycIdx[y * W + x] = path.length; path.push(x, y); return true;
  };
  /* ① 外环顺时针（外圈无墙，必定成功） */
  for (var x = 0; x < W; x++) push(x, 0);
  for (var y = 1; y < H; y++) push(W - 1, y);
  for (var x2 = W - 2; x2 >= 0; x2--) push(x2, H - 1);
  for (var y2 = H - 2; y2 >= 1; y2--) push(0, y2);
  /* ② 内层蛇形扩展：要求与当前末端相邻才连上，保持环连续 */
  var inset;
  for (inset = 1; inset < Math.min(W, H) / 2 - 1; inset++) {
    var x0 = inset, y0 = inset, x1 = W - 1 - inset, y1 = H - 1 - inset;
    if (x1 - x0 < 2 || y1 - y0 < 1) break;
    var before = path.length;
    for (var cx = x0; cx <= x1; cx++) {
      var down = ((cx - x0) % 2) === 0;
      for (var kk = 0; kk <= y1 - y0; kk++) {
        var cy = down ? y0 + kk : y1 - kk;
        var li = path.length - 2;
        if (li < 0) continue;
        var lx = path[li], ly = path[li + 1];
        var adj = (Math.abs(lx - cx) + Math.abs(ly - cy)) === 1;
        if (!adj) continue;
        push(cx, cy);
      }
    }
    if (path.length === before) break;
  }
  cycPath = path;
  return path.length / 2;   /* 平铺存 [x,y,x,y,...]，长度是格数的 2 倍 */
}

/**
 * 环上推进方向 one-hot：从头所在环段 → 最近**可吃**食物所在环段。
 * ⚠ 不能无脑走下一节 —— 环上没食物，无脑沿环 = 不吃 = 饿死。所以指向食物方向。
 * 照 env-server.js L336-357。
 */
function cycleHeadOf(S, out) {
  var o = out || obs._ch;
  o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0;
  var h = S.body[0];
  if (!h || !cycPath || !cycPath.length) return o;
  var W = S.w;
  var here = cycIdx[h.y * W + h.x];
  if (here < 0) return o;
  var f = bfsToFood(S);
  var fcell = cycIdx[f.fy * W + f.fx];
  if (fcell < 0) return o;
  var L = cycPath.length / 2;
  var fwd = (fcell - here + L) % L;
  var bwd = (here - fcell + L) % L;
  if (fwd === 0 && bwd === 0) return o;
  var target = (fwd <= bwd) ? (here + 1) % L : (here - 1 + L) % L;
  var ax = cycPath[here * 2], ay = cycPath[here * 2 + 1];
  var bx = cycPath[target * 2], by = cycPath[target * 2 + 1];
  var dx = bx - ax, dy = by - ay;
  for (var d = 0; d < 4; d++) {
    if (DIRV[d][0] === dx && DIRV[d][1] === dy) { o[d] = 1; break; }
  }
  return o;
}

/**
 * 蚀蛇逼近方向（closing）：它上一拍头 → 这一拍头 的位移，与「它→我」方向做点积。
 * +1 = 正朝我冲，-1 = 在远离。⚠ 依赖 prevRHead，aux36 每次调用后更新。
 * 0.8× 常态下它隔拍原地 → closing=0；rush/burst 时每拍都动 → 真实反映冲势。
 */
function rClosingOf(rh, ph) {
  if (!prevRHead || !rh || !ph) return 0;
  var dx = rh.x - prevRHead.x, dy = rh.y - prevRHead.y;
  if (dx === 0 && dy === 0) return 0;
  var len = Math.hypot(dx, dy);
  var tx = ph.x - rh.x, ty = ph.y - rh.y;
  var tl = Math.hypot(tx, ty);
  if (tl === 0) return 0;
  return Math.max(-1, Math.min(1, (dx * tx + dy * ty) / (len * tl)));
}
var prevRHead = null;

/**
 * 训练期增广的 36 维 aux，拼接顺序照 train.py L193-208（一个都不能错位）。
 * 这 36 维是 s5→s9 那一路所有改进的载体：食物 BFS 方向、毒块预警、
 * 自咬预警、蚀蛇饥饿/口粮/rush/烧长/收口、风筝距离与逼近、
 * 追尾可达、口袋深度、Hamiltonian 环方向。
 *
 * @returns {number[]} 长度恒为 36
 */
function aux36(S) {
  var v = obs.buf36;
  var h = S.body[0];
  var W = S.w, H = S.h, i;
  var f = bfsToFood(S);
  var dn = bfsToNeg(S);
  var rf = bfsRivalToFood(S);
  var r = S.rivals[0];
  var rAlive = !!(r && r.alive && r.body[0]);
  var rh = rAlive ? r.body[0] : null;
  var DUEL = SB.DUEL;
  /* 风筝信号：pd≤4 是它的 rush 射程 */
  var pd = (h && rh) ? Math.abs(h.x - rh.x) + Math.abs(h.y - rh.y) : 0;
  var rClosing = rClosingOf(rh, h);
  prevRHead = rh ? { x: rh.x, y: rh.y } : null;
  /* 追尾可达只算一次，minArea 从同一次结果取（省一次 flood fill） */
  var tr = tailReachOf(S);
  /* 门控：围困前兆（≥2 方向追不到尾）或长蛇（≥24 超出观测窗）→ 激活 1，否则软化 0.5。
     不做硬 0，避免短蛇期梯度断裂。 */
  var unsafeDirs = 4 - tr.reach[0] - tr.reach[1] - tr.reach[2] - tr.reach[3];
  var gate = (unsafeDirs >= 2 || S.body.length >= 24) ? 1.0 : 0.5;

  var bw = selfBiteWarn(S), rb = rivalBiteWarn(S), ch = cycleHeadOf(S);
  var p = 0;
  /* 0-2 食物方向与距离。1.5 是「找不到」的哨兵，与真实 dist/20 最大值 1 错开 */
  v[p++] = h ? (f.fx - h.x) / Math.max(W, H) : 0;
  v[p++] = h ? (f.fy - h.y) / Math.max(W, H) : 0;
  v[p++] = f.dist < 0 ? 1.5 : f.dist / 20;
  /* 3-6 BFS 首步方向 one-hot */
  v[p++] = f.dir === 0 ? 1 : 0; v[p++] = f.dir === 1 ? 1 : 0;
  v[p++] = f.dir === 2 ? 1 : 0; v[p++] = f.dir === 3 ? 1 : 0;
  /* 7 毒块预警；8 免剪剩余拍数 */
  v[p++] = dn < 0 ? 1.5 : dn / 20;
  v[p++] = Math.min(S.grow || 0, 8) / 8;
  /* 9-12 自咬预警 */
  for (i = 0; i < 4; i++) v[p++] = bw[i];
  /* 13-14 蚀蛇饥饿： starve 计数 / 距上次进食的进度 */
  v[p++] = rAlive ? Math.min(r.starve || 0, 8) / 8 : 0;
  v[p++] = rAlive ? Math.min(1, Math.max(0, (S.steps - (r.lastEat || 0)) / DUEL.STARVE_EVERY)) : 0;
  /* 15-17 蚀蛇到口粮的距离与方向（抢粮/堵路标尺） */
  v[p++] = rf.dist < 0 ? 1.5 : rf.dist / 20;
  v[p++] = rh ? (rf.fx - rh.x) / Math.max(W, H) : 0;
  v[p++] = rh ? (rf.fy - rh.y) / Math.max(W, H) : 0;
  /* 18-20 机动状态：rush / 烧长进度 / 收口进度（逼它自烧） */
  v[p++] = (rAlive && r.rush) ? 1 : 0;
  v[p++] = (rAlive && r.rush) ? Math.min(1, (r.burn || 0) / DUEL.BURN_EVERY) : 0;
  v[p++] = rAlive ? Math.min(1, (r.choke || 0) / DUEL.TURN_WAIT) : 0;
  /* 21-22 风筝：距离/20（≤0.2 即 rush 射程内）、这一拍是否正冲来 */
  v[p++] = pd / 20;
  v[p++] = rClosing;
  /* 23-26 撞蚀蛇预警 */
  for (i = 0; i < 4; i++) v[p++] = rb[i];
  /* 27-30 追尾可达 ×gate；31 口袋深度（连续量，不门控） */
  for (i = 0; i < 4; i++) v[p++] = tr.reach[i] * gate;
  v[p++] = tr.minArea;
  /* 32-35 沿环走向食物段的方向 ×gate */
  for (i = 0; i < 4; i++) v[p++] = ch[i] * gate;
  obs.dim36 = p;
  return p;   /* 应恒为 36 */
}

/** 合法动作：屏蔽与当前朝向相反的方向。与游戏 setDir L2705 的过滤规则一致。 */
function legalActions(S, out) {
  var o = out || obs._legal;
  var last = S.queue.length ? S.queue[S.queue.length - 1] : S.dir;
  var ld = SB.DIRS[last] || [1, 0];
  for (var d = 0; d < 4; d++) {
    o[d] = !(DIRV[d][0] === -ld[0] && DIRV[d][1] === -ld[1]) ? 1 : 0;
  }
  return o;
}

/** 完整 416 维观测 = 380 原生 + 36 aux（aux 在尾部，train.py L210 的 concatenate） */
function observe(S, out) {
  var v = out || obs.buf416;
  if (!obs.buf380) obs.buf380 = new Float32Array(380);
  if (!obs.buf36) obs.buf36 = new Float32Array(36);
  var n380 = observe380(S, obs.buf380);
  var n36 = aux36(S);
  for (var i = 0; i < n380; i++) v[i] = obs.buf380[i];
  for (var j = 0; j < n36; j++) v[380 + j] = obs.buf36[j];
  obs.dim416 = n380 + n36;
  return v;
}

/** 观测层的常驻缓冲与工作区 */
var obs = {
  buf380: null, buf36: null, buf416: null,
  dim380: 0, dim36: 0, dim416: 0,
  _bw: [0, 0, 0, 0], _rb: [0, 0, 0, 0], _ch: [0, 0, 0, 0], _legal: [0, 0, 0, 0],
  _o: [0, 0, 0, 0, 0, 0],
  _tr: { reach: [0, 0, 0, 0], minArea: 0 },
  _cycPath: null
};

/* ---------------------------------------------------------------- 段4 控制 */

/**
 * 选动作：合法动作掩码 + argmax。照 train.py L365-366。
 * ⚠ 掩码必须参与 —— 非法方向（掉头）Q 值再高也不能选。
 * @returns {number} 动作索引 0..3；全非法时返回 0
 */
function decide(S) {
  var x = observe(S);
  var legal = legalActions(S);
  var q = forward(x);
  if (!q[0] && !q[1] && !q[2] && !q[3]) return 0;
  var best = 0, bestQ = -Infinity, anyLegal = false;
  for (var d = 0; d < 4; d++) {
    if (!legal[d]) continue;
    anyLegal = true;
    if (q[d] > bestQ) { bestQ = q[d]; best = d; }
  }
  if (!anyLegal) return 0;
  return best;
}

/* ---------------------------------------------------------------- 段6 宿主注入 */

var SB = null;   /* 状态只经 SB.* 读，不引用任何全局标识符 */

function attach(sb) {
  SB = sb;
  if (!loadModel()) return false;
  obs.buf416 = new Float32Array(OBS_DIM);
  return true;
}

/** 换局/换关时清缓存：prevRHead 与 Hamiltonian 环都跟具体局绑定 */
function onNewState(S) {
  prevRHead = null;
  cycPath = null;
  obs._cycPath = null;
  lastS = S;
}

var lastS = null;

/**
 * 决策并下发一个方向。必须发生在 step() 内部、队列消费之前。
 * 队列非空时跳过这一拍：那一格是玩家刚按的，让它先走 ——
 * 这与训练语义等价（训练里队列恒空），同时避免 AI 的方向被压到队尾晚一拍。
 */
function beforeStep() {
  var S = SB.S;
  if (!S || S.over || S.win || !S.alive || !S.L || !S.L.duel) return;
  if (S !== lastS) onNewState(S);
  if (!cycPath) cycleBuild(S);
  if (S.queue.length > 0) return;
  var d = decide(S);
  SB.setDir(ACTIONS[d]);
}

/* ---- 浏览器侧：接管/夺回/秘笈/HUD ----
   以下全部靠「打补丁」实现，游戏本体一个字节都不用改。
   可行性依据：classic script 的顶层 function 声明挂在全局对象上，
   外部重新赋值后游戏内部的原调用点会看到新版本（顶层 const 不可重绑，
   所以只包 function）。*/

var ui = {
  mode: 'off',        /* 'off' | 'ai' | 'human' */
  patched: false,     /* 基础补丁是否装上 */
  patchedFns: [],     /* 实际装了哪些补丁，供调试查 */
  submitPatched: false,
  enterBound: false,
  lastAction: -1,
  errors: 0
};

/**
 * 重试绑定 #cheatIn 的 Enter 拦截。
 * 本文件作为 <script src> 加载时位置在 </body> 之前，DOM 里已有该输入框，
 * 通常一次就绑上；但若日后标签被挪到别处，需要重试。这里保守地幂等重试。
 */
function retryEnter() {
  if (ui.enterBound) return true;
  var doc = root.document;
  var el = doc && doc.getElementById && doc.getElementById('cheatIn');
  if (!el || !doc.addEventListener) return false;

  /* ⚠⚠⚠ 必须绑在 **document 的 capture 阶段**，绝对不能绑 #cheatIn 自己。
     这是「输几遍都是关闭」的根因，记在这里免得日后又被改回去：

     游戏在 #cheatIn 上也挂了一个 keydown（snake-blocks.html 的 submitCheat）。
     DOM 规范里，**同一个目标元素上的多个监听器按注册顺序执行，capture 标志
     不产生任何先后优势** —— capture 只在「祖先节点 vs 目标节点」之间才有意义。
     游戏那个监听器先注册（主脚本先执行），所以绑在 #cheatIn 上永远抢不过它：

       ① 游戏先跑 → submitCheat() → matchCheat('workbuddy') 匹配成功，
          但游戏的 applyCheat 里没有这条密令 → 返回 null →
          `st = (r==='on' ? '已开启' : '已关闭')` → 界面渲染成「已关闭」、
          并且把输入框 `inp.value=''` 清空；
       ② 本文件随后才跑 → 读到的是空字符串 → matchCheat 返回 null → 直接 return。
     净效果：AI 一次都没被切换，屏幕上却每次都说「已关闭」。

     stopPropagation 也救不了场：它只挡跨节点传播，挡不住**同一节点上**已注册的
     后续监听器（那需要 stopImmediatePropagation）。所以这里两道一起下。 */
  doc.addEventListener('keydown', function (e) {
    var t = e.target;
    if (!t || t.id !== 'cheatIn') return;
    if (e.key !== 'Enter') return;
    if (!isWorkbuddy(t.value)) return;            /* 其他密令 → 交还原逻辑 */
    e.preventDefault();
    e.stopImmediatePropagation();                 /* 别让原 submitCheat 也跑 */
    runWorkbuddy(t);
  }, true);

  /* 「执行」按钮（#chGo）走的是 onclick = submitCheat，不经过 keydown ——
     必须单独接管，否则点按钮仍然是「已关闭」而且 AI 根本没切换。
     ⚠ 这里**整体覆盖** onclick，不能用 addEventListener 追加：
       追加会让一次点击跑两个处理器（游戏的和我们的），一开一关，净效果又是关。 */
  var go = doc.getElementById('chGo');
  if (go) go.onclick = function () { runWorkbuddy(doc.getElementById('cheatIn')); };

  ui.enterBound = true;
  return true;
}

/** 这个输入值是不是本文件该接管的密令（交给游戏的 submitCheat 只会得到「已关闭」） */
function isWorkbuddy(v) {
  var M = root.SB;
  var c = M && M.matchCheat ? M.matchCheat(v) : null;
  return !!(c && c.key === 'workbuddy');
}

/** 真正切换 AI，并把结果写回面板（Enter 与「执行」按钮共用同一条路径） */
function runWorkbuddy(inp) {
  var r = toggleAI();
  syncCheatList();
  syncHudChip();
  var msg = root.document && root.document.getElementById('chMsg');
  if (msg) {
    msg.textContent = r === 'on'
      ? '✓ 📈 WorkBuddy 托管 已开启（按方向键随时收回）'
      : r === 'blocked'
        ? '✗ 📈 只能在「同源相噬」关卡里代打'
        : '✓ 📈 WorkBuddy 托管 已关闭';
  }
  if (inp) { inp.value = ''; try { inp.focus(); } catch (e) {} }
}

function cheatPanelOpen() {
  try {
    var el = (root.document && root.document.getElementById) && root.document.getElementById('pCheat');
    return !!el && !el.classList.contains('hide');
  } catch (e) { return false; }
}

function flash(msg) {
  try { if (typeof root.flashTips === 'function') root.flashTips(msg); } catch (e) {}
}
function hud() {
  try { if (typeof root.updateHUD === 'function') root.updateHUD(); } catch (e) {}
}

/* ---------- 状态可视化（HUD 芯片 / 秘笈列表标记） --------------------
   ⚠⚠ 为什么这里不用「包函数」补丁：
     updateHUD 与 renderCheatList 都定义在 snake-blocks.html 的
     `if(typeof window!=='undefined'){` **块内**（3702 行起、5318 行才闭合），
     是**块级函数声明**。非严格模式的 Annex B 语义会额外在外层建一份 var 绑定
     并同步给 window，但那是**两个独立绑定** —— 外部改 window.updateHUD
     影响不到块内的原调用点，游戏主循环调 updateHUD() 走的仍是块级绑定，
     补丁永远看不见。
     （对比：step 定义在**逻辑段顶层**，那里没有块，所以包 step 是有效的。）

     改用 MutationObserver 直接盯 DOM：游戏每次重写 #pwrs / #chList 之后，
     我们把状态补回去。这是不侵入游戏本体的唯一可靠做法。 */
function syncHudChip() {
  try {
    var doc = root.document;
    var pw = doc && doc.getElementById('pwrs');
    if (!pw) return;
    var chip = pw.querySelector('[data-wb-chip]');
    var txt = ui.mode === 'ai' ? '📈 AI 托管中'
            : ui.mode === 'human' ? '📈 已让出'
            : (!ui.patched ? '📈 模型未加载' : '');
    if (!txt) { if (chip) chip.remove(); return; }
    if (!chip) {
      chip = doc.createElement('span');
      chip.setAttribute('data-wb-chip', '1');
      chip.style.cssText = 'opacity:.9;font-size:12px;margin-left:6px';
      pw.appendChild(chip);
    }
    /* ⚠ 只在文字真的变了才写：textContent 赋值本身会再触发一次
       MutationObserver，无脑写就成了回调自激。 */
    if (chip.textContent !== txt) chip.textContent = txt;
  } catch (e) {}
}

function syncCheatList() {
  try {
    var doc = root.document;
    var el = doc && doc.getElementById('chList');
    if (!el) return;
    var rows = el.querySelectorAll('.ci');
    for (var i = 0; i < rows.length; i++) {
      var b = rows[i].querySelector('b');
      if (!b || b.textContent.indexOf('workbuddy') < 0) continue;
      var tag = rows[i].querySelector('[data-wb-tag]');
      if (ui.mode === 'ai') {
        rows[i].classList.add('on');
        if (!tag) {
          tag = doc.createElement('font');
          tag.setAttribute('data-wb-tag', '1');
          tag.setAttribute('color', 'var(--gold)');
          rows[i].appendChild(tag);
        }
        if (tag.textContent !== '\u3000[已开启]') tag.textContent = '\u3000[已开启]';
      } else {
        rows[i].classList.remove('on');
        if (tag) tag.remove();
      }
      break;
    }
  } catch (e) {}
}

/** 盯住 HUD 与秘笈列表：游戏每次重写它们，我们就把状态补回去。 */
function watchDom() {
  try {
    if (!root.MutationObserver) return false;
    var doc = root.document;
    var pairs = [['pwrs', syncHudChip], ['chList', syncCheatList]];
    for (var i = 0; i < pairs.length; i++) {
      var el = doc.getElementById(pairs[i][0]);
      if (!el) continue;
      var mo = new root.MutationObserver(pairs[i][1]);
      mo.observe(el, { childList: true, subtree: true, characterData: true });
    }
    syncHudChip();
    return true;
  } catch (e) { return false; }
}

/** 切换 AI 托管开关（秘笈 workbuddy 专用） */
function toggleAI() {
  if (!SB) return 'off';
  if (ui.mode === 'ai') {
    ui.mode = 'off';
    flash('📈 已关闭 AI 托管');
    hud();
    return 'off';
  }
  /* 只在决斗关可用：观测里 16 维蚀蛇相关特征在其他关全是 0，
     跑起来等于半盲，别让人以为 AI 抽风。 */
  var S = SB.S;
  if (S && S.L && !S.L.duel) {
    flash('📈 AI 只能在「同源相噬」里代打');
    /* ⚠ 返回 'blocked' 而不是 'off'：调用方要能区分「这儿不让用」和「刚被关掉」，
       否则面板会把拒绝渲染成「已关闭」—— 那正是本次要修的误导文案。 */
    return 'blocked';
  }
  ui.mode = 'ai';
  onNewState(S);
  if (S) S.cheatUsed = true;      /* 彩蛋局不写存档，不污染进度 */
  flash('📈 AI 接管方向盘 · 按任意方向键收回');
  hud();
  return 'on';
}

/** 玩家按方向键 → 立刻收回方向盘。AI 静默待命，不自动抢回。 */
function handToHuman(why) {
  if (ui.mode !== 'ai') return;
  ui.mode = 'human';
  flash('📈 已让出' + (why ? '（' + why + '）' : '') + ' · Tab 输 workbuddy 可再交回');
  hud();
}

/**
 * 装全部宿主补丁。只做一次，重复调用直接返回。
 * @param {object} rootObj 全局对象（浏览器是 window）
 */
function install(rootObj) {
  if (ui.patched) return true;
  if (!rootObj.SB || typeof rootObj.step !== 'function') {
    loadError = '宿主不可用：没找到 window.SB / window.step';
    return false;
  }
  SB = rootObj.SB;
  if (!loadModel()) return false;
  obs.buf416 = new Float32Array(OBS_DIM);

  /* ① 决策时序核心：包 step，在队列消费之前跑决策。
     挂在 step() **内部**而不是 rAF 循环里，一次解决四个问题：
       · 一帧最多 8 步（guard<8）→ 内部就调 8 次，天然逐步对齐
       · tickNow() 因秘笈变速 → AI 频次自动跟随，无需重算
       · 秘笈面板暂停 → loop() 根本不进 while，AI 自然冻结
       · 1 tick 延迟 → 本拍 setDir 的方向本拍就被 shift 消费
     ⚠ 异常绝不能冒泡进游戏主循环，否则一个彩蛋能把游戏搞崩。 */
  var origStep = rootObj.step;
  rootObj.step = function () {
    if (ui.mode === 'ai') {
      try { beforeStep(); }
      catch (e) {
        ui.errors++;
        if (ui.errors === 1) {
          loadError = 'AI 推理出错：' + (e && e.message);
          try { rootObj.console && rootObj.console.error('[snake_ai] ' + loadError); } catch (e2) {}
          try { rootObj.console && rootObj.console.error(e && e.stack); } catch (e3) {}
        }
        if (ui.errors >= 3) { ui.mode = 'off'; flash('📈 AI 出错已自动关闭'); }
      }
    }
    return origStep.apply(this, arguments);
  };
  ui.patchedFns.push('step');
  /* headless 驱动走的是 __game.step，同步换掉，保证验收路径一致 */
  if (rootObj.__game) rootObj.__game.step = rootObj.step;

  /* ② 秘笈注册：往 CHEATS 挂一条。matchCheat 用 for...in 遍历，自动收录，
        匹配器本身一行都不用改。
        ⚠ 顶层 const 不挂到 window（classic script 语义），所以这些符号一律
          要从 window.SB 上取 —— 这也是本方案只包 function、不碰 const 的原因。 */
  var M = rootObj.SB;
  if (M.CHEATS) {
    M.CHEATS.workbuddy = {
      key: 'workbuddy',
      alias: ['ai', 'bot', '托管', '自动', '代打'],
      name: 'WorkBuddy 托管',
      icon: '📈',
      desc: '强化学习蛇接管方向盘（仅「同源相噬」可用 · 按任意方向键随时收回 · 本局不计入存档）'
    };
  }

  /* ③ 秘笈执行：包 applyCheat 加 workbuddy 分支；'off' 顺带关 AI。 */
  if (typeof M.applyCheat === 'function') {
    var origApply = M.applyCheat;
    M.applyCheat = function (c) {
      if (c && c.key === 'workbuddy') return toggleAI();
      var r = origApply.apply(this, arguments);
      if (c && c.key === 'off') ui.mode = 'off';
      return r;
    };
    ui.patchedFns.push('applyCheat');
    /* 这里只在 SB 上挂一份带 workbuddy 分支的 applyCheat，**供调试与 headless 用**。
       ⚠⚠ 故意**不**写 rootObj.applyCheat = … ：
          游戏 submitCheat 调的是自由变量 applyCheat（解析到脚本顶层的全局绑定），
          所以真写成 rootObj.applyCheat 它会生效 —— 但那样一旦 ⑥ 的 Enter 拦截
          或 #chGo 的覆盖失效，一次输入就会同时触发两条路径，toggleAI 被调用两次
          → 一开一关、净效果还是关（正是本次要修的病症）。宁可只留一条通路。
       （顺带纠正早先这句注释的误判：applyCheat 是**顶层 function 声明**、本身可重绑，
         不是「顶层 const 不可重绑」；它改不动的真正原因是这里改的是 **SB 对象的属性**，
         游戏内部根本不读那个属性。） */
    if (rootObj.__game) rootObj.__game.applyCheat = M.applyCheat;
  }

  /* ④ 秘笈列表：**不再替换 renderCheatList**（早先这里写的是 M.renderCheatList = …）。
     ⚠ 那是给 SB 对象换属性，而 renderCheatList 是**块级**函数声明 ——
       游戏内部调用走的是块级绑定，换了也看不见。workbuddy 那一行的「[已开启]」
       改由 syncCheatList 直接操作 DOM 补上（见 watchDom）。
     （顺带记一笔：游戏原版的 on 判定是硬编码的 (ghost||rush)，连 half 都漏了 ——
       那是游戏本体的瑕疵，不属于本彩蛋的职责，不动它。） */

  /* ⑥ 真实玩家路径：拦 #cheatIn 的 Enter（capture），抢在游戏那个 bubble
        监听器之前。详见 retryEnter() 处的注释 —— 为什么不能包 submitCheat。*/
  if (retryEnter()) ui.submitPatched = true;
  ui.patchedFns.push('cheatIn:Enter');

  /* ⑤ 按键夺回：capture 阶段同步翻转状态，抢在游戏自己的 bubble 处理器之前。
        游戏处理器随后照常跑，setDir 立即生效 —— 所以这里既不碰 S.queue，
        也不 preventDefault。玩家感觉就是「方向盘瞬间回到手里」，零延迟。
        ⚠ 必须先判 cheatPanelOpen()：面板开着时玩家在打字，
          输 workbuddy 里的 w/d 会被误判成夺回。 */
  if (rootObj.addEventListener) {
    rootObj.addEventListener('keydown', function (e) {
      if (ui.mode !== 'ai') return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (cheatPanelOpen()) return;
      var key = e.key;
      if (HUMAN_KEYS[key] || key === 'ArrowUp' || key === 'ArrowDown' ||
          key === 'ArrowLeft' || key === 'ArrowRight') {
        handToHuman('按键');
      }
    }, true);
    rootObj.addEventListener('touchend', function () {
      if (ui.mode !== 'ai') return;
      if (cheatPanelOpen()) return;
      handToHuman('滑动');
    }, true);
    ui.patchedFns.push('keydown', 'touchend');
  }

  /* ⑥ HUD 状态芯片 + 秘笈列表标记：走 MutationObserver，**不包函数**。
        为什么包不动（updateHUD / renderCheatList 是块级函数声明），
        见 syncHudChip 上方的长注释。 */
  if (watchDom()) ui.patchedFns.push('MutationObserver #pwrs,#chList');

  ui.patched = true;
  return true;
}

/* ---------------------------------------------------------------- 段7 导出 */

var api = {
  OBS_DIM: OBS_DIM,
  ACTIONS: ACTIONS,
  DIRV: DIRV,
  version: '1.1.0',
  /* 注入与控制 */
  attach: attach,
  install: install,
  beforeStep: beforeStep,
  onNewState: onNewState,
  toggleAI: toggleAI,
  handToHuman: handToHuman,
  /* 推理 */
  loadModel: loadModel,
  forward: forward,
  decodeB64: decodeB64,
  /* 观测（node 侧测试与调试都要用，所以一并暴露） */
  observe: observe,
  observe380: observe380,
  aux36: aux36,
  legalActions: legalActions,
  decide: decide,
  /* 诊断 */
  loaded: function () { return !!NET; },
  tag: function () { return NET ? NET.tag : null; },
  loadError: function () { return loadError; },
  dims: function () { return { d380: obs.dim380, d36: obs.dim36, d416: obs.dim416 }; },
  mode: function () { return ui.mode; },
  status: function () {
    return {
      mode: ui.mode, patched: ui.patched, patchedFns: ui.patchedFns.slice(),
      errors: ui.errors, tag: NET ? NET.tag : null,
      loadError: loadError, dims: api.dims()
    };
  },
  /* 浏览器里自动装载（权重文件必须先于本文件加载）。
     检测不到宿主就安静待命 —— 游戏照常可玩，不弹错。 */
  autoInstall: function () {
    if (root.SB && typeof root.step === 'function') return install(root);
    return false;
  }
};

root.SNAKE_AI = api;
if (typeof module !== 'undefined' && module.exports) module.exports = api;

/* 浏览器自动装载。node 下没有 window，这段天然不触发。
     标签放在 </body> 之前，DOM 已就绪，retryEnter 通常一次就绑上；
     DOMContentLoaded / load 再各兜一次（幂等）。 */
if (typeof window !== 'undefined') {
  try { api.autoInstall(); } catch (e) { /* 彩蛋不该有能力让游戏崩 */ }
  var _retry = function () { try { if (ui.patched) retryEnter(); } catch (e) {} };
  if (root.document && root.document.addEventListener) {
    root.document.addEventListener('DOMContentLoaded', _retry);
  }
  root.addEventListener('load', _retry);
}

})(typeof globalThis !== 'undefined' ? globalThis : this);