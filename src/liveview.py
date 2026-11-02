"""实时画面：把每帧的检测结果画成带框的图，供浏览器以 MJPEG 观看。

为什么框画在后端而不是前端叠加
------------------------------
前端叠加需要同时同步两条流（图像 + 框坐标）。网络一抖动，框就会和图错位几帧，
看起来像"框飘了"。直接在后端把框画死，前端一个 <img src="/stream"> 就能看，
弱网最差也只是掉帧，不会框图不匹配。

详细的框信息（类别、置信度、轨迹 ID）另外走 /api/state 给前端列表用，
所以"能不能看清框"和"能不能查细节"这两件事互不干扰。

为什么限流
----------
检测本身 16~19 fps，但画框 + JPEG 编码是纯 CPU 开销。如果不限流，
CPU 会被视频编码吃掉，反而拖慢检测。默认压到 12 fps，肉眼完全够。

中文标签说明
------------
cv2.putText 不支持中文，硬写会变成一串问号。这里类别名一律用英文大写
（FIRE / SMOKE），颜色才是主要的区分手段：火红、烟橙、其他灰。
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

# 类别 → 框颜色（BGR）。0=fire 1=smoke，与 dataset.yaml 一致
# 颜色只表达"是什么"，不再表达"有多确定"。
#
# 之前用红/黄区分置信度档位，问题是：同一种目标在不同置信下会变色，
# 看起来像换了个东西；而且黄红都在抢注意力，反而分不出类别。
# 现在按用户要求改成固定四色：
#   灰 = 未达阈值（只是看到了，不作主张）
#   红 = 火（达阈值）
#   黄 = 烟（达阈值）
#   蓝 = 香烟（达阈值）
COLORS = {
    0: (48, 48, 235),      # 红：火
    1: (0, 215, 255),      # 黄：烟
}
# 香烟单独一色：蓝。它是独立类别（自己的模型、自己的门限），
# 视觉上也要能一眼和"烟情"分开 —— 很多场景里烟和香烟会同时出现。
CIG_COLOR = (235, 140, 40)     # 蓝（BGR）
CAND_COLOR = (150, 150, 150)   # 灰：未达阈值
CAND_COLORS = {0: CAND_COLOR, 1: CAND_COLOR}
DEFAULT_COLOR = (190, 190, 190)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class LiveView:
    """最新帧的单一存放点：pipeline 写，HTTP 线程读。"""

    def __init__(self, *, max_width: int = 1280, jpeg_quality: int = 80,
                 fps_limit: float = 12.0, max_bytes: int = 90000,
                 fallback_quality: int = 50):
        self.max_width = max_width
        self.jpeg_quality = int(jpeg_quality)
        self.min_interval = 1.0 / max(fps_limit, 1.0)
        # 体积兜底：静止画面一帧约 26KB，但画面剧烈变化（开关灯、大幅移动）
        # 时压缩率骤降，实测能到 143KB —— 编码本身只多花 0.4ms，可传输和
        # 浏览器解码会明显变慢，前端串行拉取时表现为"画面一变就停一下"。
        # 超过上限就降一档质量重编，体积能砍一半，代价才 0.3ms。
        self.max_bytes = int(max_bytes)
        self.fallback_quality = int(fallback_quality)
        # 三档着色（都由配置驱动）：
        #   < alarm_conf          灰：未达阈值，只表示"系统看到这块区域了"
        #   alarm_conf ~ hot_conf 黄：刚过线，疑似，等 LLM 复核
        #   >= hot_conf           红/橙：高置信，基本可以定性
        # 两档不够用的原因：人脸被强光照到能拿到 fire 0.49，刚过 0.45 就变红，
        # 和真火 0.82 一个颜色，反而分不出轻重。
        # 告警阈值（pipeline 用来决定要不要送审/报警，由 run_multi 注入）
        self.alarm_confs: dict[int, float] = {0: 0.45, 1: 0.50}
        # **彩色框阈值**：比告警阈值更高。
        #
        # 用户的要求是"门限可以压低，但不要一直有彩色框认错"。
        # 这两件事本来是两个问题：
        #   · 检测门限低  -> 多出框（灰框），不漏
        #   · 彩色框门槛高 -> 只有相当确定才上色，不刺眼
        # 拆开之后可以放心把检测放宽，而画面依然干净。
        self.color_confs: dict[int, float] = {0: 0.45, 1: 0.55}
        self.hot_confs: dict[int, float] = {0: 0.70, 1: 0.75}

        self._cond = threading.Condition(threading.Lock())
        self._jpeg: bytes | None = None
        self._seq = 0
        self._state: dict = {"seq": 0, "dets": [], "counts": {}, "fps": 0.0}
        # 跨帧框平滑缓存：[(box, cls_name, conf), ...]
        # 用 IOU 把本帧的框和上一帧匹配上，然后做指数滑动平均。
        # 为什么在这层做：上游已经用 Kalman 平滑了位置，但轨迹重建
        # （目标短暂丢失后重新捕获）时新轨迹的框会跳到新位置 —— 这一层
        # 兜住那种跨轨迹的跳变，让眼睛看到的是连续的。
        self._prev_boxes: list = []

        self._last_pub = 0.0
        self._n_pub = 0
        self._t_first = time.time()
        self.llm = None                      # 由外部注入 LlmArbiter，供 /api/llm 用
        self.last_frame = None               # 最近一帧原图（存训练样本用）
        self.last_dets = []                  # 最近一帧的检测（做预标注用）

    # ------------------------------------------------------------ 写入
    def publish(self, image: np.ndarray, dets, *, frame_idx: int | None = None,
                cam_id: str = "", fps: float = 0.0, extra: dict | None = None) -> bool:
        """由 pipeline 每帧调用。返回本次是否真的编码了一帧（被限流时为 False）。"""
        # 存一份**原始帧 + 本帧检测**，供"存样本"用。
        # 放在限流判断**之前**：限流是给推流用的，存样本不受它影响。
        # 存的是没画框的 image，训练素材必须是干净画面。
        self.last_frame = image
        self.last_dets = list(dets or [])
        now = time.time()
        if now - self._last_pub < self.min_interval:
            return False
        self._last_pub = now

        try:
            drawn = self._draw(image, dets, frame_idx, cam_id)
            ok, buf = cv2.imencode(".jpg", drawn,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                if not getattr(self, "_err_logged", False):
                    self._err_logged = True
                    print("[liveview] jpeg 编码失败", flush=True)
                return False
            if len(buf) > self.max_bytes:
                ok2, buf2 = cv2.imencode(
                    ".jpg", drawn,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.fallback_quality])
                if ok2 and len(buf2) < len(buf):
                    buf = buf2
            jpeg = buf.tobytes()
        except Exception as exc:               # noqa: BLE001  画框失败绝不能拖垮检测
            if not getattr(self, "_err_logged", False):
                self._err_logged = True
                print(f"[liveview] publish 异常: {type(exc).__name__}: {exc}", flush=True)
            return False

        counts: dict[str, int] = {}
        det_list = []
        for d in dets:
            name = getattr(d, "cls_name", "obj")
            counts[name] = counts.get(name, 0) + 1
            det_list.append({
                "cls_id": int(getattr(d, "cls_id", -1)),
                "name": name,
                "conf": round(float(getattr(d, "conf", 0.0)), 4),
                "xyxy": [round(float(v), 1) for v in getattr(d, "xyxy", (0, 0, 0, 0))],
            })

        self._n_pub += 1
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._state = {
                "seq": self._seq,
                "ts": round(now, 3),
                "frame_idx": frame_idx,
                "cam": cam_id,
                "size": [int(image.shape[1]), int(image.shape[0])],
                "dets": det_list,
                "counts": counts,
                "fps": round(fps, 2),
                "pub_fps": round(self._n_pub / max(1e-6, now - self._t_first), 2),
                **(extra or {}),
            }
            self._cond.notify_all()
        return True

    # ------------------------------------------------------------ 画框
    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        return inter / union if union > 0 else 0.0

    def _smooth(self, dets, alpha: float = 0.45):
        """跨帧框平滑：同类别 + IOU 匹配 → 指数滑动平均。

        alpha 是基准权重（0.45）。实际使用时按每帧位移自适应：
          静止  -> 约 0.45，重在历史，压抖动
          移动快 -> 最高 0.9，重在观测，不拖影
        框的宽高也各自独立平滑，避免尺寸忽大忽小。
        """
        prev = self._prev_boxes
        used = set()
        out = []
        for d in dets:
            box = getattr(d, "xyxy", None)
            if box is None:
                out.append((None, getattr(d, "cls_name", ""), getattr(d, "conf", 0.0)))
                continue
            name = str(getattr(d, "cls_name", ""))
            best, best_iou = -1, 0.25
            for i, (pb, pn, _pc) in enumerate(prev):
                if i in used or pn != name or pb is None:
                    continue
                v = self._iou(pb, box)
                if v > best_iou:
                    best, best_iou = i, v
            if best >= 0:
                pb = prev[best][0]
                # 自适应 alpha：按"这一帧移动了多少"动态调整历史权重。
                #   目标基本不动 -> 重在历史（更稳，压下抖动）
                #   目标快速移动 -> 重在观测（跟手，避免拖影和滞后）
                # 移动量用两框中心的位移 / 框对角线长度衡量，做归一化。
                _mx = (float(box[0] + box[2]) - float(pb[0] + pb[2])) / 2.0
                _my = (float(box[1] + box[3]) - float(pb[1] + pb[3])) / 2.0
                _diag = max(1.0, ((float(pb[2] - pb[0])) ** 2
                                  + (float(pb[3] - pb[1])) ** 2) ** 0.5)
                _shift = (_mx * _mx + _my * _my) ** 0.5 / _diag

                # ── 基准 alpha 按**目标特性**分，不是照搬头部追踪的 0.45 ──
                # 头部追踪那套是为"大目标、平缓移动"调的。现在这三类完全不一样：
                #
                #   火苗  20~40px，**剧烈抖动 + 闪烁**，帧间位置能跳十几像素
                #         -> 基准要压得更低（0.28），重在历史，否则框跟着火苗乱跳
                #   香烟  12~40px，手持、移动慢但检测分数低、偶尔漏帧
                #         -> 0.30，压抖为主，慢移动时不会拖影
                #   烟    通常 60~200px，缓慢扩散
                #         -> 0.45（原来的基准），形态变化本来就是它的特征
                #
                # 另外**小目标整体再压一档**：小目标上同样的像素抖动
                # 占框比例大得多，同样的 alpha 视觉上就是"更抖"。
                _w = abs(float(box[2]) - float(box[0]))
                _h = abs(float(box[3]) - float(box[1]))
                _base = {"fire": 0.28, "cigarette": 0.30, "smoke": 0.45}.get(name, 0.40)
                if min(_w, _h) < 48:          # 小目标：再重一点历史
                    _base *= 0.78
                elif min(_w, _h) > 160:       # 大目标：可以更信任观测
                    _base = min(0.5, _base * 1.15)

                # 位移增益 1.4 也按目标大小调：小目标归一化位移天然更大，
                # 用同样的增益会让"稍微动一下"就跳到 0.9，等于没平滑。
                _gain = 0.9 if min(_w, _h) < 48 else 1.4
                _a = min(0.9, max(0.20, _base + _shift * _gain))
                box = tuple(_a * float(c) + (1.0 - _a) * float(p)
                            for c, p in zip(box, pb))
                used.add(best)
            out.append((box, name, getattr(d, "conf", 0.0)))
        self._prev_boxes = out
        return out

    def _draw(self, image: np.ndarray, dets, frame_idx, cam_id) -> np.ndarray:
        # 先做跨帧平滑，再画。把平滑后的框包成和 Detection 同构的对象，
        # 这样下面所有绘制/统计代码不用改结构。
        class _S:
            __slots__ = ("xyxy", "cls_id", "cls_name", "conf", "area")
            def __init__(self, box, name, conf, cid):
                self.xyxy = box if box is not None else (0, 0, 0, 0)
                self.cls_name = name
                self.conf = conf
                self.cls_id = cid
                self.area = max(0.0, self.xyxy[2] - self.xyxy[0]) * max(0.0, self.xyxy[3] - self.xyxy[1])

        # 恢复**自适应 EMA 平滑**。
        #
        # 之前我判断"Kalman 已经平滑过，这里再平滑是重复"，把它删了 ——
        # 这个判断是错的，删完之后框开始跳，用户明确反馈"以前头部追踪
        # 可以精准丝滑不偏离，现在越来越糟"。
        #
        # 两者根本不重复，负责的事不同：
        #   Kalman  预测下一帧在哪 + 用检测校正（处理**移动**）
        #   EMA     跨帧压抖动、尺寸不忽大忽小（处理**抖动**）
        # 而且这里的 EMA 是**自适应**的：alpha = clamp(0.45 + 位移比*1.4, 0.25, 0.9)
        #   静止   -> 0.25~0.45，重在历史，压抖动
        #   快速移动 -> 最高 0.9，重在观测，不拖影
        # 所以它不会像固定 EMA 那样引入滞后。
        _raw = list(dets or [])
        _sm = self._smooth(_raw)
        dets = [_S(b, n, c, getattr(_raw[i], "cls_id", -1) if i < len(_raw) else -1)
                for i, (b, n, c) in enumerate(_sm)]
        img = image.copy()
        h, w = img.shape[:2]
        placed: list[tuple[int, int, int, int]] = []   # 已占用的标签区域，防重叠
        # 灰框显示规则（三段收紧，专治"满屏框"）：
        #   ① 最多画 2 个 —— 之前 8 个，屏幕上十几个标签糊成一片
        #   ② 面积必须 >= 画面 0.6% —— 小噪点（脸上的、墙上的几十像素块）
        #      对值班员没有信息量，只会干扰
        #   ③ 按置信度从高到低取前 2 个
        # 灰框的价值只有"这里可能有问题"这一句话，说两遍就够了。
        _cand_limit = 2
        _cand_drawn = 0
        # 面积门槛必须按类别分开：
        #   火/烟 是成片的东西，小于画面 0.6% 的基本是噪点
        #   香烟 本身就是细长小目标，一支烟在 960x540 里通常只有几百像素
        #        （约 0.06%）。之前统一用 0.6%，等于把香烟全挡掉了 ——
        #        用户举着烟却"检测 0 个目标"就是这么来的。
        _cand_min_area = h * w * 0.006      # 火/烟
        _cand_min_area_cig = h * w * 0.00015  # 香烟：约 78 像素起

        def collides(box) -> bool:
            x1, y1, x2, y2 = box
            return any(not (x2 <= a or x1 >= c or y2 <= b or y1 >= d)
                       for a, b, c, d in placed)

        for d in dets:
            x1, y1, x2, y2 = (int(round(v)) for v in getattr(d, "xyxy", (0, 0, 0, 0)))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            cid = int(getattr(d, "cls_id", -1))
            conf = float(getattr(d, "conf", 0.0))
            # 画彩色框用 color_thr（更严）；alarm_thr 只用于告警逻辑，不参与配色。
            # 香烟单独取阈值：它的 cls_id 也是 0，和 fire 撞了，
            # 所以按**类别名**判断，不能按 id。
            _nm = str(getattr(d, "cls_name", "")).lower()
            if "cig" in _nm:
                alarm_thr = self.color_confs.get(2, 0.55)
            else:
                alarm_thr = self.color_confs.get(cid, 0.50)
            # 颜色只由「类别 + 是否达阈值」决定，不再分置信度档位
            # 灰框限量：未达阈值的框最多画 N 个（按置信度从高到低）。
            # 阈值降到 0.08 后画面里能冒出几十个灰框，满屏标签反而什么都看不清 ——
            # 它们的信息量本来就只有"这里有东西"这一点，留前几个足够提示了。
            if conf < alarm_thr:
                _bw = max(1.0, x2 - x1)
                _bh = max(1.0, y2 - y1)
                _is_cig = "cig" in str(getattr(d, "cls_name", "")).lower()
                _need = _cand_min_area_cig if _is_cig else _cand_min_area
                if _cand_drawn >= _cand_limit or _bw * _bh < _need:
                    continue
            if conf >= alarm_thr:
                if "cig" in str(getattr(d, "cls_name", "")).lower():
                    color = CIG_COLOR
                else:
                    color = COLORS.get(cid, DEFAULT_COLOR)
                thick, suffix = 2, ""
            else:
                color, thick, suffix = CAND_COLOR, 1, " ?"
                _cand_drawn += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

            name = str(getattr(d, "cls_name", "obj")).upper()
            # 带上轨迹 ID，格式 FIRE #12 0.34
            # 用户要这个是为了肉眼确认"断了之后回来的是不是同一个目标" ——
            # 只有带着 ID 才能判断一次抖动到底换了几个轨迹。
            _tid = int(getattr(d, "tid", -1))
            _id_txt = f" #{_tid}" if _tid >= 0 else ""
            # 光敏引擎的框 conf=0，显示 "0.00" 没意义，只留颜色名
            label = (f"{name}{_id_txt}" if conf <= 0.0
                     else f"{name}{_id_txt} {conf:.2f}{suffix}")
            (tw, th), base = cv2.getTextSize(label, FONT, 0.5, 1)
            bh = th + base + 4

            # 默认贴在框的上沿；与已画的标签撞了就往下挪，最多挪 6 次
            ly = y1 - bh - 2
            for _ in range(6):
                if ly >= 0 and not collides((x1, ly, x1 + tw + 6, ly + bh)):
                    break
                ly += bh
            if ly < 0:
                ly = y1 + 2
            ly = max(0, min(h - bh - 1, ly))
            placed.append((x1, ly, x1 + tw + 6, ly + bh))

            cv2.rectangle(img, (x1, ly), (x1 + tw + 6, ly + bh), color, -1)
            # 告警框用白字压在彩色底上；候选框用深字压在灰底上，进一步弱化
            # 暗色候选底上用浅灰字，彩色告警底上用白字
            txt = (215, 215, 215) if conf < alarm_thr else (255, 255, 255)
            cv2.putText(img, label, (x1 + 3, ly + th + 1), FONT, 0.5,
                        txt, 1, cv2.LINE_AA)

        # 左上角信息条：帧号 / 检测数 / 时间
        bar = f"f{frame_idx if frame_idx is not None else '-'}  det {len(dets)}  " \
              f"{time.strftime('%H:%M:%S')}"
        if cam_id:
            bar = f"{cam_id}  " + bar
        cv2.putText(img, bar, (8, 22), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, bar, (8, 22), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if w > self.max_width:                  # 控制带宽与解码开销
            scale = self.max_width / w
            img = cv2.resize(img, (self.max_width, int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return img

    # ------------------------------------------------------------ 读出
    def state(self) -> dict:
        with self._cond:
            return dict(self._state)

    def frame_bytes(self) -> bytes | None:
        """最新一帧的 JPEG 字节，供 /api/frame.jpg 逐帧拉取。

        为什么还要留这条路径：MJPEG（multipart/x-mixed-replace）在部分浏览器、
        内嵌 WebView 和预览面板里根本不会渲染，表现为"黑屏但接口有数据"。
        逐帧拉取是普通 img 请求，任何环境都能显示，代价只是多几个 HTTP 请求。
        """
        with self._cond:
            return self._jpeg

    def mjpeg_iter(self):
        """持续产出 multipart 分片。调用方负责写响应头。"""
        last = -1
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._seq != last, timeout=2.0)
                jpeg, seq = self._jpeg, self._seq
            if jpeg is None or seq == last:
                continue                        # 还没第一帧，或只是等超时
            last = seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                   + jpeg + b"\r\n")

    # ------------------------------------------------------------ LLM 状态
    def llm_state(self) -> dict:
        """给 /api/llm 用。没注入 arbiter 时返回明确的未配置状态。"""
        if self.llm is None:
            return {"configured": False, "reason": "主程序未启用 LLM 裁判"}
        try:
            return self.llm.snapshot()
        except Exception as e:                  # noqa: BLE001
            return {"configured": False, "reason": f"读取失败: {e}"}
