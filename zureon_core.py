# ===============================================
# ZUREON Core Seven - Polished Prototype
# Author: Yuki & Marumaru (ZUREON Lab)
# Version: 2025.10.06
# ===============================================

import json, os, random, math
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

import streamlit as st

# -----------------------------------------------
# 0. ユーティリティ & 永続ログ
# -----------------------------------------------

LOGDIR = Path.home() / ".zureon"
LOGDIR.mkdir(parents=True, exist_ok=True)

def current_time_iso() -> str:
    return datetime.utcnow().isoformat()

def save_jsonl(filename: str, data: Dict[str, Any]):
    path = LOGDIR / filename
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# -----------------------------------------------
# 1. Fractal Seed Core（揺らぎ）– DRD基盤
# -----------------------------------------------

class FractalSeedCore:
    def __init__(self, seed=None):
        self.state = 0.0
        if seed is not None:
            random.seed(seed)

    def generate_deviation(self) -> float:
        # 後で周波数ベースに拡張可能な簡易揺らぎ
        fluct = random.uniform(-1.0, 1.0)
        self.state += fluct * 0.1
        return self.state

# -----------------------------------------------
# 2. Adaptive Law（法則進化核）– 負帰還
# -----------------------------------------------

class AdaptiveLaw:
    def __init__(self, adjust_rate=0.05):
        self.adjust_rate = adjust_rate

    def evolve(self, deviation: float) -> float:
        return -deviation * self.adjust_rate

# -----------------------------------------------
# 3. PAT Core（ズレ↔感情）
# -----------------------------------------------

class PATCore:
    def __init__(self, th=0.5):
        self.th = th

    def map_to_emotion(self, deviation: float) -> str:
        if deviation > self.th:
            return "興奮"
        elif deviation < -self.th:
            return "退屈"
        else:
            return "安定"

# -----------------------------------------------
# 4. Dream Core（未来予測：EMA）
# -----------------------------------------------

class DreamCore:
    def __init__(self, alpha=0.3):
        self.ema = None
        self.alpha = alpha

    def predict_future(self, history: List[float]) -> float:
        if not history:
            return 0.0
        x = history[-1]
        self.ema = x if self.ema is None else self.alpha * x + (1 - self.alpha) * self.ema
        return self.ema

# -----------------------------------------------
# 5. Illusion Core（幻想補正：感情別非線形）
# -----------------------------------------------

class IllusionCore:
    def correct_perception(self, deviation: float, emotion: str) -> float:
        if emotion == "興奮":
            return math.tanh(deviation * 1.2)      # 強刺激を圧縮
        if emotion == "退屈":
            return deviation * 1.08 + 0.02         # 微増幅＋僅かなバイアス
        return deviation * 0.98                    # 安定：微減衰

# -----------------------------------------------
# 6. Fluidnéa Core（流動変換：減衰）
# -----------------------------------------------

class FluidneaCore:
    def __init__(self, base_decay=0.98):
        self.base_decay = base_decay

    def flow_adjust(self, value: float, decay: float) -> float:
        return value * decay

# -----------------------------------------------
# 7. ZONT Core（存在全体ログ）
# -----------------------------------------------

class ZontCore:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def record(self, payload: Dict[str, Any]):
        entry = dict(time=current_time_iso(), **payload)
        self.logs.append(entry)
        save_jsonl("zont_log.jsonl", entry)

# -----------------------------------------------
# 8. ZUREON Core Manager（統合）
# -----------------------------------------------

class ZureonCore:
    def __init__(self, seed=None):
        self.seed = FractalSeedCore(seed=seed)
        self.law = AdaptiveLaw(adjust_rate=0.05)
        self.pat = PATCore(th=0.5)
        self.dream = DreamCore(alpha=0.3)
        self.illusion = IllusionCore()
        self.fluid = FluidneaCore(base_decay=0.98)
        self.zont = ZontCore()
        self.history: List[float] = []
        self.base_adjust = self.law.adjust_rate
        self.base_decay = self.fluid.base_decay
        self.emotions = {"興奮":0, "安定":0, "退屈":0}

    def step(self):
        # 1) 揺らぎ
        deviation = self.seed.generate_deviation()
        # 2) 予測（EMA）
        future_pred = self.dream.predict_future(self.history)
        # 3) 感情
        emotion = self.pat.map_to_emotion(deviation)
        self.emotions[emotion] += 1
        # 4) 幻想補正（非線形）
        corrected = self.illusion.correct_perception(deviation, emotion)

        # 5) 予測×感情で減衰を可変（先読みでブレーキ/アクセル）
        decay = self.base_decay
        if emotion == "興奮":
            decay = self.base_decay - 0.02 * max(0.0, future_pred)
        elif emotion == "退屈":
            decay = self.base_decay + 0.02 * max(0.0, -future_pred)
        flowed = self.fluid.flow_adjust(corrected, decay)

        # 6) 予測で負帰還ゲインも可変（揺れが続くと締め付け強め）
        self.law.adjust_rate = self.base_adjust * (1.0 + 0.5 * abs(future_pred))
        correction = self.law.evolve(flowed)
        new_state = flowed + correction

        # 7) 記録
        self.history.append(new_state)
        self.zont.record({
            "deviation_raw": deviation,
            "emotion": emotion,
            "corrected": corrected,
            "decay": decay,
            "gain": self.law.adjust_rate,
            "future": future_pred,
            "new_state": new_state,
            "history_len": len(self.history)
        })

        return {
            "raw": deviation,
            "emotion": emotion,
            "corrected": corrected,
            "decay": decay,
            "gain": self.law.adjust_rate,
            "flowed": flowed,
            "future": future_pred,
            "new_state": new_state
        }

# -----------------------------------------------
# 9. Streamlit UI（状態保持＋可視化）
# -----------------------------------------------

def get_core():
    if "core" not in st.session_state:
        st.session_state.core = ZureonCore()
    return st.session_state.core

def main_ui():
    st.set_page_config(page_title="ZUREON Core Seven", page_icon="🌀", layout="centered")
    st.title("🌀 ZUREON Core Seven — Prototype")
    st.caption("Deviation → Predict(EMA) → Emotion → Illusion(nonlinear) → Flow(decay) → Law(gain) → Log")

    core = get_core()

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("▶ 1 Step"):
        res = core.step()
        st.json(res, expanded=False)

    if c2.button("⏩ 10 Steps"):
        last = None
        for _ in range(10):
            last = core.step()
        st.success("10 steps done")
        if last: st.json(last, expanded=False)

    if c3.button("📥 Save Snapshot"):
        payload = {
            "time": current_time_iso(),
            "len": len(core.history),
            "last": core.history[-1] if core.history else None,
            "emotions": core.emotions
        }
        save_jsonl("snapshot.jsonl", payload)
        st.toast("Snapshot saved to ~/.zureon/snapshot.jsonl")

    if c4.button("♻ Reset Core"):
        st.session_state.core = ZureonCore()
        st.success("Core reset")

    with c5.popover("⚙ Options"):
        seed_in = st.text_input("Seed（空で現状維持）", "")
        if st.button("Apply Seed"):
            seed_val = None if seed_in.strip()=="" else int(seed_in)
            st.session_state.core = ZureonCore(seed=seed_val)
            st.success(f"Seed applied: {seed_in or 'None'}")

    st.subheader("📈 State History")
    if core.history:
        st.line_chart(core.history)
    else:
        st.info("まだ履歴がありません。『▶ 1 Step』から。")

    st.subheader("🧠 Emotions (session)")
    st.write(core.emotions)

    with st.expander("🧾 ZONT Log (latest 50)"):
        st.write(core.zont.logs[-50:])

    st.caption(f"Logs: {LOGDIR}/zont_log.jsonl / Snapshots: {LOGDIR}/snapshot.jsonl")

# -----------------------------------------------
# 10. CLIモード
# -----------------------------------------------

def cli_loop():
    core = ZureonCore()
    print("=== ZUREON Core Seven CLI ===")
    print("Enterで進化、'seed <int>'でシード設定、'q'で終了")
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        if cmd.startswith("seed"):
            parts = cmd.split()
            if len(parts)==2 and parts[1].isdigit():
                core = ZureonCore(seed=int(parts[1]))
                print(f"[seed set to {parts[1]}]")
                continue
        res = core.step()
        print(f"[{res['emotion']}]\traw={res['raw']:.3f}\tnew={res['new_state']:.3f}\tfut={res['future']:.3f}\tdecay={res['decay']:.3f}\tgain={res['gain']:.3f}")

# -----------------------------------------------
# Entry
# -----------------------------------------------

if __name__ == "__main__":
    # UI起動:  streamlit run this_file.py
    # CLI起動: python this_file.py
    if any("streamlit" in arg for arg in os.sys.argv):
        main_ui()
    else:
        cli_loop()
