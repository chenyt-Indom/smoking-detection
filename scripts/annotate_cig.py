"""香烟数据集复标工具（旋转框 + 可排除样本 + 黑白图筛查）。

为什么要有这个东西
------------------
v32 香烟模型 mAP50 已经 0.875（能检出），但 mAP50-95 只有 0.4405（框不准）。
诊断结论不是"练得不够"，而是**标注框自相矛盾**：
  · train/box_loss 1.496 ≈ val/box_loss 1.514（训练和验证一样差）
  · 两代模型都早停（v32 计划 60 轮 33 轮停，v31 计划 100 轮 75 轮停）
  · 标注框在 960 输入下中位 84×75 像素 —— 快成正方形了，而香烟是细长的
所以唯一出路是把框重标标准；香烟常是斜的，轴对齐框天生盖不住，因此支持旋转。

三条独立的处置动作（互不干扰）
------------------------------
1. 改框      → labels_fixed/（轴对齐，现有 detect 模型可用）
                labels_obb/（四点旋转，将来换 -obb 模型用）
2. 排除      → excluded.json（这张图不加入训练清单，例如黑白/翻拍/水印太重的废图）
                ⚠️ 只影响"清单"，**不删原图**，随时可恢复
3. 黑白筛查  → bw_flags.json（自动算色度，供批量排除）

原始 labels/ 与 images/ 一个字节都不会动。

用法
----
    python scripts/annotate_cig.py            # http://127.0.0.1:5077
    python scripts/annotate_cig.py --port 5088
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(r"D:\training_data\smoke\v32_smoking")
APP_HTML = ROOT / "web" / "annotate_cig.html"

FIXED = DATA / "labels_fixed"
OBB = DATA / "labels_obb"
REVIEWS = DATA / "reviews.jsonl"
EXCLUDED = DATA / "excluded.json"
BWFLAGS = DATA / "bw_flags.json"
PROGRESS = DATA / "progress.json"      # 上次标到哪张：关机重启后自动回到原地
ANGLES = DATA / "angles.json"          # 每张图每框的旋转角（弧度）
# ⚠️ 角度必须单独存，不能从归一化四点反推：
#    四点存的是 x/iw、y/ih，非正方形图上 atan2(dy,dx) 会被宽高比压歪
#    （实测 1280x720 的图，真值 15° 会读成 25.47°）

_LOCK = threading.Lock()
_BW = {"running": False, "done": 0, "total": 0, "split": ""}
_ANG = {"mtime": None, "data": {}}
_WH: dict[str, tuple[int, int] | None] = {}
_LIST_CACHE: dict = {"sig": None, "data": {}}


# ---------------------------------------------------------------- 小工具
def img_dir(split: str) -> Path:
    return DATA / "images" / split


def lbl_dir(split: str) -> Path:
    return DATA / "labels" / split


def _load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                     # noqa: BLE001
        return default


def _save_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def _write_text_atomic(p: Path, text: str) -> None:
    """先写 .tmp 再 os.replace：关机/断电写到一半也不会留下半截标注文件。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def load_excluded() -> dict:
    d = _load_json(EXCLUDED, {})
    return {"train": set(d.get("train") or []), "val": set(d.get("val") or [])}


def save_excluded(d: dict) -> None:
    _save_json(EXCLUDED, {k: sorted(v) for k, v in d.items()})


def load_bw() -> dict:
    return _load_json(BWFLAGS, {})


# ---------------------------------------------------------------- 标注读写
def _angles_from_obb(split: str, stem: str) -> list[float]:
    """从四点 OBB 反推角度 —— 仅用于兼容没有 angles.json 的旧数据。

    ⚠️ 必须**逐行**读：数据集里存在多框图（train 42 张 ≥3 框、最多一张 14 个框），
    只取第一行会让界面看起来"只有一个框"，一保存就把其余框覆盖掉。
    ⚠️ 四点坐标是归一化的（x/iw、y/ih），直接 atan2 会被宽高比压歪 ——
       必须乘回真实宽高再算，否则 1280x720 上真值 15° 会算成 25.47°。
    """
    f = OBB / split / f"{stem}.txt"
    if not f.is_file():
        return []
    wh = _img_wh(split, f"{stem}.jpg")
    sx, sy = wh if wh else (1.0, 1.0)
    out = []
    for line in f.read_text(encoding="utf-8").strip().splitlines():
        p = line.split()
        if len(p) < 9:
            continue
        try:
            v = [float(x) for x in p[1:9]]
        except ValueError:
            continue
        out.append(math.atan2((v[3] - v[1]) * sy, (v[2] - v[0]) * sx))
    return out


def read_angles(split: str, stem: str) -> list[float]:
    """优先读 angles.json（精确）；没有记录才从四点反推（旧数据兼容）。"""
    d = load_angles().get(split) or {}
    key = f"{stem}.jpg"
    if key in d:
        try:
            return [float(x) for x in (d[key] or [])]
        except (TypeError, ValueError):
            pass
    return _angles_from_obb(split, stem)


def load_angles() -> dict:
    """{"train": {"000-0.jpg": [-0.2618]}, ...}，带 mtime 缓存。"""
    try:
        m = ANGLES.stat().st_mtime if ANGLES.is_file() else None
    except OSError:
        m = None
    if _ANG["mtime"] != m:
        _ANG["data"] = _load_json(ANGLES, {}) or {}
        _ANG["mtime"] = m
    return _ANG["data"]


def put_angles(split: str, name: str, angs) -> None:
    d = _load_json(ANGLES, {}) or {}
    d.setdefault(split, {})[Path(name).name] = [round(float(a), 6) for a in (angs or [])]
    _save_json(ANGLES, d)
    _ANG["mtime"] = None


def _img_wh(split: str, name: str):
    """图像真实宽高（只读文件头，很快）。用于把归一化四点还原到像素空间。"""
    key = f"{split}/{name}"
    if key in _WH:
        return _WH[key]
    wh = None
    try:
        from PIL import Image
        with Image.open(img_dir(split) / name) as im:
            wh = im.size
    except Exception:                     # noqa: BLE001
        wh = None
    _WH[key] = wh
    return wh


def _parse_boxes(text: str) -> list[list[float]]:
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            out.append([float(v) for v in p[1:5]])
        except ValueError:
            continue
    return out


def read_boxes(split: str, stem: str) -> tuple[list[list[float]], bool]:
    """返回 (所有框, 是否已复标)。空列表 + edited=True 表示"标记为无香烟"。"""
    fixed = FIXED / split / f"{stem}.txt"
    if fixed.is_file():
        return _parse_boxes(fixed.read_text(encoding="utf-8")), True
    f = lbl_dir(split) / f"{stem}.txt"
    if not f.is_file():
        return [], False
    return _parse_boxes(f.read_text(encoding="utf-8", errors="ignore")), False


def _dir_sig() -> str:
    """用来判断列表缓存能不能复用：任一相关目录/文件变了就重算。"""
    parts: list[str] = []
    for p in (img_dir("train"), img_dir("val"), lbl_dir("train"), lbl_dir("val"),
              FIXED / "train", FIXED / "val", OBB / "train", OBB / "val"):
        try:
            parts.append(f"{p.stat().st_mtime:.3f}")
        except OSError:
            parts.append("-")
    for p in (EXCLUDED, ANGLES, BWFLAGS):
        try:
            parts.append(f"{p.stat().st_mtime:.3f}")
        except OSError:
            parts.append("-")
    return "|".join(parts)


def list_items(split: str, only: str = "all") -> list[dict]:
    """带缓存。直算要读 7813 个 label 文件，实测 3.8 秒，页面一刷新就卡。"""
    sig = _dir_sig()
    c = _LIST_CACHE
    if c["sig"] != sig:          # 任一目录/文件变了，整份缓存作废
        c["sig"] = sig
        c["data"] = {}
    key = f"{split}|{only}"
    if key in c["data"]:
        return c["data"][key]
    out = _build_list(split, only)
    c["data"][key] = out
    return out


def _build_list(split: str, only: str = "all") -> list[dict]:
    ex = load_excluded()[split]
    bw = load_bw().get(split) or {}
    out = []
    for p in sorted(img_dir(split).glob("*.jpg")):
        boxes, edited = read_boxes(split, p.stem)
        angs = read_angles(split, p.stem) if edited else []
        item = {
            "name": p.name, "stem": p.stem,
            "boxes": boxes,
            "angs": angs + [0.0] * max(0, len(boxes) - len(angs)),
            "edited": edited,
            "excluded": p.name in ex,
            "bw": bw.get(p.name),
        }
        if only == "edited" and not edited:
            continue
        if only == "excluded" and not item["excluded"]:
            continue
        if only == "bw" and item["bw"] is None:
            continue
        if only == "unreviewed" and (edited or item["excluded"]):
            continue
        out.append(item)
    return out


def stats() -> dict:
    ex = load_excluded()
    res = {}
    for split in ("train", "val"):
        total = len(list(img_dir(split).glob("*.jpg")))
        done = len(list((FIXED / split).glob("*.txt"))) if (FIXED / split).is_dir() else 0
        res[split] = {"total": total, "reviewed": done, "excluded": len(ex[split])}
    return res


# ---------------------------------------------------------------- 保存
def save(payload: dict) -> dict:
    split = str(payload.get("split") or "train")
    if split not in ("train", "val"):
        return {"ok": False, "err": "bad split"}
    stem = Path(str(payload.get("name") or "")).stem
    if not stem:
        return {"ok": False, "err": "bad name"}
    action = str(payload.get("action") or "ok")
    # 多框：优先用 boxes（列表），兼容旧的单框 aabb
    boxes = payload.get("boxes")
    obbs = payload.get("obbs")
    if boxes is None:
        aabb = payload.get("aabb")
        boxes = [aabb] if aabb else []
        obbs = [payload.get("obb")] if payload.get("obb") else []
    boxes = [b for b in (boxes or []) if b and len(b) == 4]
    obbs = list(obbs or [])
    angs_in = payload.get("angs")      # 前端直传的像素空间真实角度（最准）
    fx = FIXED / split / f"{stem}.txt"
    ob = OBB / split / f"{stem}.txt"
    with _LOCK:
        fx.parent.mkdir(parents=True, exist_ok=True)
        ob.parent.mkdir(parents=True, exist_ok=True)
        if action == "empty" or not boxes:
            _write_text_atomic(fx, "")
            _write_text_atomic(ob, "")
        else:
            _write_text_atomic(fx, "".join(
                "0 " + " ".join(f"{float(v):.6f}" for v in b) + "\n" for b in boxes))
            # 客户端没给旋转四点（脚本/旧版调用）时，别拿轴对齐四点去覆盖已存的旋转标注
            has_obb = any(o and len(o) == 8 for o in obbs)
            keep_old = False
            if not has_obb and ob.is_file():
                try:
                    old = [ln for ln in ob.read_text(encoding="utf-8").splitlines() if ln.strip()]
                except OSError:
                    old = []
                keep_old = len(old) == len(boxes)
            if not keep_old:
                ob_lines = []
                for i, b in enumerate(boxes):
                    if i < len(obbs) and obbs[i] and len(obbs[i]) == 8:
                        vals = [float(v) for v in obbs[i]]
                    else:
                        cx, cy, w, h = (float(v) for v in b)
                        vals = [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2,
                                cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]
                    ob_lines.append("0 " + " ".join(f"{v:.6f}" for v in vals) + "\n")
                _write_text_atomic(ob, "".join(ob_lines))
        # 角度单独落盘：不靠四点反推（归一化四点反推会被宽高比压歪）
        if boxes:
            if isinstance(angs_in, list) and len(angs_in) == len(boxes):
                put_angles(split, f"{stem}.jpg", angs_in)
            else:
                put_angles(split, f"{stem}.jpg", _angles_from_obb(split, stem))
        else:
            put_angles(split, f"{stem}.jpg", [])
        try:
            with open(REVIEWS, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "split": split,
                                    "name": f"{stem}.jpg", "action": action,
                                    "n_boxes": len(boxes), "boxes": boxes},
                                   ensure_ascii=False) + "\n")
        except OSError:
            pass
        # 记住进度：下次打开自动回到这张（列表可能被筛选/排除改变，所以只存名字）
        try:
            _save_json(PROGRESS, {"split": split, "name": f"{stem}.jpg",
                                  "action": action, "ts": round(time.time(), 1)})
        except OSError:
            pass
    return {"ok": True, "stats": stats()}


def migrate_angles() -> int:
    """给已标图补 angles.json —— 用真实宽高把归一化四点还原到像素空间再反推。

    没被二次保存过的图，这样算出来的角度就是当初标的角度（误差 <0.01°）。
    """
    d = _load_json(ANGLES, {}) or {}
    n = 0
    for split in ("train", "val"):
        od = OBB / split
        if not od.is_dir():
            continue
        for f in sorted(od.glob("*.txt")):
            d.setdefault(split, {})[f"{f.stem}.jpg"] = [
                round(a, 6) for a in _angles_from_obb(split, f.stem)]
            n += 1
    _save_json(ANGLES, d)
    _ANG["mtime"] = None
    return n


def set_excluded(split: str, names: list[str], excluded: bool) -> dict:
    with _LOCK:
        d = load_excluded()
        for n in names:
            n = Path(str(n)).name
            if excluded:
                d[split].add(n)
            else:
                d[split].discard(n)
        save_excluded(d)
    return {"ok": True, "stats": stats()}


# ---------------------------------------------------------------- 黑白筛查
def _chroma(path: Path) -> float | None:
    """色度均值（0~255）。全灰的图接近 <6；彩色图通常 >15。"""
    try:
        import cv2
        import numpy as np
        buf = np.fromfile(str(path), dtype=np.uint8)     # 中文路径安全读法
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            return None
        im = cv2.resize(im, (48, 48), interpolation=cv2.INTER_AREA).astype("int16")
        d = im.max(axis=2) - im.min(axis=2)              # 每像素通道极差
        return float(d.mean())
    except Exception:                     # noqa: BLE001
        return None


def bw_scan(split: str) -> None:
    files = sorted(img_dir(split).glob("*.jpg"))
    _BW.update(running=True, done=0, total=len(files), split=split)
    res: dict[str, float] = {}
    for i, p in enumerate(files):
        c = _chroma(p)
        if c is not None:
            res[p.name] = round(c, 2)
        _BW["done"] = i + 1
    with _LOCK:
        allb = load_bw()
        allb[split] = res
        _save_json(BWFLAGS, allb)
    _BW.update(running=False)


# ---------------------------------------------------------------- 导出训练清单
def export_list(split: str) -> dict:
    ex = load_excluded()[split]
    names = [p.name for p in sorted(img_dir(split).glob("*.jpg")) if p.name not in ex]
    out = DATA / f"{split}_use.txt"
    out.write_text("\n".join((img_dir(split) / n).as_posix() for n in names) + "\n",
                   encoding="utf-8")
    return {"ok": True, "file": str(out), "count": len(names), "excluded": len(ex)}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj):
        self._send(200, "application/json; charset=utf-8",
                   json.dumps(obj, ensure_ascii=False).encode())

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            if not APP_HTML.is_file():
                self._send(500, "text/plain; charset=utf-8",
                           b"missing web/annotate_cig.html")
                return
            self._send(200, "text/html; charset=utf-8", APP_HTML.read_bytes())
            return
        if u.path == "/api/stats":
            self._json(stats())
            return
        if u.path == "/api/progress":
            self._json(_load_json(PROGRESS, {}))
            return
        if u.path == "/api/list":
            self._json(list_items((q.get("split") or ["train"])[0],
                                  (q.get("only") or ["all"])[0]))
            return
        if u.path == "/api/bw_status":
            self._json(dict(_BW))
            return
        if u.path == "/img":
            split = (q.get("split") or ["train"])[0]
            name = (q.get("name") or [""])[0]
            if "/" in name or "\\" in name or ".." in name:
                self._send(400, "text/plain", b"bad name")
                return
            p = img_dir(split) / name
            if not p.is_file():
                self._send(404, "text/plain", b"not found")
                return
            self._send(200, mimetypes.guess_type(p.name)[0] or "image/jpeg",
                       p.read_bytes())
            return
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as exc:                       # noqa: BLE001
            self._json({"ok": False, "err": f"bad body: {exc}"})
            return

        if u.path == "/api/save":
            self._json(save(body))
        elif u.path == "/api/exclude":
            split = str(body.get("split") or "train")
            names = body.get("names") or ([body.get("name")] if body.get("name") else [])
            self._json(set_excluded(split, names, bool(body.get("excluded", True))))
        elif u.path == "/api/bw_scan":
            split = str(body.get("split") or "train")
            if _BW["running"]:
                self._json({"ok": False, "err": "已在扫描中"})
            else:
                threading.Thread(target=bw_scan, args=(split,), daemon=True).start()
                self._json({"ok": True})
        elif u.path == "/api/export":
            self._json(export_list(str(body.get("split") or "train")))
        else:
            self._send(404, "text/plain", b"not found")


def _warm_list_cache() -> None:
    """后台预热列表缓存：不预热的话，页面第一次打开要空等 3.9 秒（读 7813 个 label）。

    只预热 train/all —— 标数据集基本都在 train，val 等真用到再算。
    """
    try:
        t0 = time.time()
        list_items("train", "all")
        print(f"[✓] 列表缓存已预热（{time.time() - t0:.1f}s）")
    except Exception as e:                # noqa: BLE001
        print(f"[!] 预热失败：{e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5077)
    ap.add_argument("--open", action="store_true",
                    help="启动后自动用默认浏览器打开页面（双击 bat 时用）")
    ap.add_argument("--migrate-angles", action="store_true",
                    help="给已标图补 angles.json（修宽高比导致的读角度偏差）")
    a = ap.parse_args()
    if not DATA.is_dir():
        print(f"[x] 数据集不存在：{DATA}")
        return 2
    if a.migrate_angles:
        n = migrate_angles()
        print(f"[✓] 已迁移 {n} 张图的角度 → {ANGLES}")
        return 0
    st = stats()
    print("=" * 72)
    print("香烟数据集复标工具（旋转框 / 排除样本 / 黑白筛查）")
    print(f"  数据集 {DATA}")
    for k, v in st.items():
        print(f"  {k}: {v['total']} 张，已复标 {v['reviewed']}，已排除 {v['excluded']}")
    print(f"  结果目录：{FIXED.name}/ 与 {OBB.name}/（原始 labels/ 不动）")
    print()
    print("  ⚠️ 必须用浏览器打开下面这个地址（预览面板里的静态副本没有接口）：")
    print(f"     http://{a.host}:{a.port}/")
    print("  Ctrl+C 停止")
    print("=" * 72)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    threading.Thread(target=_warm_list_cache, daemon=True).start()
    if a.open:
        threading.Timer(0.8, lambda: webbrowser.open(
            f"http://{a.host}:{a.port}/")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
