# ZURE-Chat Seed: ultra-minimal CLI prototype
# Character-level online bigram learner with per-turn 'phi' (CE) and meaning snapshots.
# Run: python zure_chat_seed.py   (type 'exit' to stop)

import json, time, sys, os, math, random
from collections import defaultdict, Counter
from datetime import datetime

SAVE_DIR = os.path.dirname(__file__)
session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = os.path.join(SAVE_DIR, f"session-{session_id}.jsonl")
meaning_path = os.path.join(SAVE_DIR, f"meaning-{session_id}.jsonl")

unigram = Counter()
bigram = defaultdict(Counter)
cooccur = defaultdict(Counter)
alpha = 1.0      # add-one smoothing
window = 2       # co-occur window
ma_beta = 0.8    # moving average for CE
ma_ce = None
turn = 0
unique_tokens = set()
unique_bigrams = set()

def tokenize(text):
    # character-level tokens (no predefined meaning)
    return list(text)

def bigram_prob(prev, curr):
    numer = bigram[prev][curr] + alpha
    denom = sum(bigram[prev].values()) + alpha * max(1, len(unigram))
    return numer / denom

def avg_cross_entropy(tokens):
    if len(tokens) <= 1:
        return 0.0
    ce = 0.0
    n = 0
    for i in range(1, len(tokens)):
        p = bigram_prob(tokens[i-1], tokens[i])
        p = max(p, 1e-12)
        ce -= math.log(p)
        n += 1
    return ce / max(n, 1)

def update_counts(tokens):
    for i, tok in enumerate(tokens):
        unigram[tok] += 1
        unique_tokens.add(tok)
        if i > 0:
            prev = tokens[i-1]
            bigram[prev][tok] += 1
            unique_bigrams.add((prev, tok))
        # co-occur within window
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j == i: 
                continue
            cooccur[tok][tokens[j]] += 1

def generate_reply(user_tokens, max_len=40, temperature=0.9):
    # toy reply seeded by last token; samples via bigram model
    if user_tokens:
        start = user_tokens[-1]
    else:
        start = (list(unigram.keys()) or ["あ"])[0]
    reply = [start]
    for _ in range(max_len-1):
        prev = reply[-1]
        candidates = list(unigram.keys()) or ["。","あ","い"]
        # make a distribution
        ps = []
        s = 0.0
        for c in candidates:
            p = bigram_prob(prev, c)
            p = max(p, 1e-12) ** (1.0/max(temperature,1e-6))
            ps.append(p); s += p
        if s <= 0:
            nxt = random.choice(candidates)
        else:
            ps = [p/s for p in ps]
            nxt = random.choices(candidates, ps, k=1)[0]
        reply.append(nxt)
        if nxt in ["。","！","!","?","？","\n"]:
            break
    return "".join(reply)

def top_cooccur_for(token, k=5):
    if token not in cooccur or not cooccur[token]:
        return []
    return [t for t,_ in cooccur[token].most_common(k)]

def write_meaning_snapshot():
    with open(meaning_path, "a", encoding="utf-8") as f:
        for tok, cnt in unigram.items():
            if cnt < 3:
                continue
            neighbors = top_cooccur_for(tok, k=5)
            rec = {
                "token": tok,
                "count": cnt,
                "neighbors": neighbors,
                "updated_at": datetime.now().isoformat()
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    global ma_ce, turn
    print("ZURE-Chat Seed (超ミニ) — type 'exit' to quit")
    print("学びの指標: phi(CE) / 移動平均CE / 語彙サイズ / ビグラム数")
    with open(log_path, "a", encoding="utf-8") as lf:
        while True:
            try:
                user = input("> ")
            except EOFError:
                break
            if user.strip().lower() == "exit":
                print("bye.")
                break
            tokens = tokenize(user)
            # compute phi before updating
            phi = avg_cross_entropy(tokens)
            ma_ce = phi if ma_ce is None else ma_beta*ma_ce + (1-ma_beta)*phi
            reply = generate_reply(tokens, max_len=20, temperature=0.9)
            rec = {
                "t_iso": datetime.now().isoformat(),
                "turn_id": turn,
                "user_text": user,
                "phi": round(phi, 4),
                "ma_ce": round(ma_ce, 4),
                "reply": reply
            }
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lf.flush()
            update_counts(tokens)
            turn += 1
            print(f"[phi={phi:.3f} / maCE={ma_ce:.3f} / vocab={len(unique_tokens)} / bigrams={len(unique_bigrams)}]")
            print(reply)
            if turn % 5 == 0:
                write_meaning_snapshot()
                print("(meaning snapshot updated)")
    print(f"ログ: {log_path}")
    print(f"意味スナップショット: {meaning_path}")

if __name__ == "__main__":
    main()
