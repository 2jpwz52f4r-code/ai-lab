"""AI Research Lab (無料API版・議論版)
流れ: ①参加AIが意見を短く → ②全員が全員の意見を読んで応答 → ③論点の整理(1回)
      → ④深掘り(最大2回) → ⑤最終回答を1つ
途中経過は画面の「途中経過を開く」で見られる。最終回答だけ読めばよい形。
"""
import os, secrets, sqlite3, threading, time
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()
DB = os.getenv("LAB_DB", "lab.db")
MAX_CALLS = int(os.getenv("MAX_CALLS", "16"))  # Render側に MAX_CALLS があればそちらが優先
TIMEOUT = 60
lock = threading.Lock()

# 得意分野の初期値（評価ボタンの結果で少しずつ変わる）
PRIOR = {
    "gemini":     {"hypothesis": .60, "critique": .60, "synthesize": .80},
    "groq":       {"hypothesis": .50, "critique": .70, "synthesize": .50},
    "mistral":    {"hypothesis": .65, "critique": .60, "synthesize": .60},
    "openrouter": {"hypothesis": .50, "critique": .50, "synthesize": .50},
}
BASE = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class Adapter:
    """1社ぶんの接続。鍵とモデル名は環境変数から。"""

    def __init__(self, name):
        self.name, self.base = name, BASE[name]
        self.key = os.getenv(f"{name.upper()}_API_KEY")
        self.model = os.getenv(f"{name.upper()}_MODEL")
        self.state = "設定済み（未テスト）" if self.live() else "未設定（モック）"

    def live(self):
        return bool(self.key and self.model) and os.getenv("MOCK") != "1"

    def ask(self, prompt, system=""):
        """returns (text, is_mock)"""
        if not self.live():
            return f"[モック:{self.name}] 「{prompt[:30]}…」への仮の回答です。", True
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        r = httpx.post(self.base + "/chat/completions", timeout=TIMEOUT,
                       headers={"Authorization": "Bearer " + self.key},
                       json={"model": self.model, "messages": msgs})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], False


A = {n: Adapter(n) for n in BASE}
ALL = list(BASE.keys())
# 実際に鍵とモデルが入っているAIだけを議論に参加させる（未設定のモックは入れない）
ROSTER = [n for n in ALL if A[n].live()] or ALL

# ---- DB ----
PG = os.getenv("DATABASE_URL")  # あればクラウドDB(Postgres)、無ければ手元のSQLite


def q(sql, args=(), ins=False):
    """SQLを1つ実行。ins=True なら新しいidを、それ以外は行のlist[dict]を返す。"""
    if PG:
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(PG, row_factory=dict_row) as c:
            cur = c.execute(sql.replace("?", "%s") + (" RETURNING id" if ins else ""), args)
            if ins:
                return cur.fetchone()["id"]
            return cur.fetchall() if cur.description else []
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    try:
        with c:
            cur = c.execute(sql, args)
            if ins:
                return cur.lastrowid
            return [dict(r) for r in cur.fetchall()] if cur.description else []
    finally:
        c.close()


_ID = "SERIAL PRIMARY KEY" if PG else "INTEGER PRIMARY KEY"
for _t in (
    f"CREATE TABLE IF NOT EXISTS runs(id {_ID}, topic TEXT, ts DOUBLE PRECISION, calls INTEGER)",
    f"CREATE TABLE IF NOT EXISTS msgs(id {_ID}, run_id INTEGER, step TEXT, ai TEXT, task TEXT,"
    " content TEXT, mock INTEGER, ok INTEGER, ts DOUBLE PRECISION)",
    f"CREATE TABLE IF NOT EXISTS ratings(id {_ID}, ai TEXT, task TEXT, good INTEGER, ts DOUBLE PRECISION)",
):
    q(_t)


def score(ai, task):
    """初期値5回ぶんの重みで平滑化。少ない評価では断定しない。"""
    r = q("SELECT COALESCE(SUM(good),0) AS g, COUNT(*) AS n FROM ratings WHERE ai=? AND task=?", (ai, task))[0]
    p = PRIOR[ai][task]
    return (p * 5 + r["g"]) / (5 + r["n"])


def pick(task, n=1, exclude=()):
    return sorted((a for a in A if a not in exclude), key=lambda a: -score(a, task))[:n]


class Run:
    def __init__(self, topic):
        self.topic, self.calls = topic, 0
        self.id = q("INSERT INTO runs(topic,ts,calls) VALUES(?,?,0)", (topic, time.time()), ins=True)

    def left(self):
        return MAX_CALLS - self.calls

    def call(self, step, ai, task, prompt, fallback=True):
        """1回の呼び出し。fallback=True なら失敗時に別のAIへ回す。全部記録する。"""
        order = [ai] + ([a for a in pick(task, len(A)) if a != ai] if fallback else [])
        for name in order:
            with lock:
                if self.calls >= MAX_CALLS:
                    return None, None, None
                self.calls += 1
            try:
                text, mock = A[name].ask(prompt)
                ok = 1
            except Exception as e:  # 429やタイムアウトなど
                code = getattr(getattr(e, "response", None), "status_code", "")
                text, mock, ok = f"エラー: {type(e).__name__} {code}", False, 0
            mid = q("INSERT INTO msgs(run_id,step,ai,task,content,mock,ok,ts) VALUES(?,?,?,?,?,?,?,?)",
                    (self.id, step, name, task, text, int(mock), ok, time.time()), ins=True)
            q("UPDATE runs SET calls=? WHERE id=?", (self.calls, self.id))
            if ok:
                return name, text, mid
        return None, None, None


# ---- プロンプト（短く・役割をはっきり）----
OPINION_FMT = ("次の形式だけで答えてください（日本語・前置き不要）\n"
               "結論: （1〜2文）\n根拠: （1行）")

RESPOND_FMT = ("次の形式で答えてください（日本語）\n"
               "同意する点: （1〜2行）\n"
               "反対する点: （必ず1つ以上。1〜2行）\n"
               "足りない点: （1行）\n"
               "※自分の意見も見直してよい。賛成だけで終わらないこと。")

ORGANIZE_FMT = ("次の形式で整理してください（日本語）\n"
                "合意している点: （箇条書き2個まで）\n"
                "割れている論点: （箇条書き2個まで。どの立場が何を主張しているかを書く）\n"
                "まだ分からない点: （1行）\n"
                "※事実・推論・仮説を区別すること。")

DEEPEN_FMT = ("上の論点整理で割れている論点の中から最も重要な1つを選び、"
              "その反対側の立場から最も強い根拠を3行以内で述べ、"
              "どこが妥当かを1行で評価してください（日本語）。")

COUNTER_FMT = ("前の深掘りに対して、反論か補強を3行以内で書いてください（日本語）。"
               "同じ結論の繰り返しは不要。")

FINAL_FMT = ("上の議論を踏まえて、最終回答を次の形式で1つにまとめてください（日本語）\n"
             "結論: （3文以内）\n根拠: （箇条書き3個まで）\n残る疑問: （1〜2個）\n"
             "※議論を経た仮説であり、検証済みの結論ではないことを明記すること。")


def opinions_text(ops):
    return "\n\n".join(f"【{n}の意見】\n{t}" for n, t in ops)


# 公開URLだと誰でも無料APIの枠を使えてしまうので、クラウドではパスワード必須（ユーザー名は何でもOK）
PW = os.getenv("LAB_PASSWORD")
if PG and not PW:
    raise RuntimeError("クラウド運用では LAB_PASSWORD を必ず設定してください")
_basic = HTTPBasic(auto_error=False)


def auth(c: HTTPBasicCredentials = Depends(_basic)):
    if PW and (c is None or not secrets.compare_digest(c.password.encode(), PW.encode())):
        raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})


app = FastAPI(dependencies=[Depends(auth)])


class RunIn(BaseModel):
    topic: str


@app.post("/api/run")
def run(body: RunIn):
    r = Run(body.topic.strip()[:500])
    topic = r.topic

    # ① 参加AIが、それぞれ短く意見を出す（互いの意見は見せない）
    ops = []
    for ai in ROSTER:
        name, text, _ = r.call("意見", ai, "hypothesis",
                               f"テーマ: {topic}\n{OPINION_FMT}", fallback=False)
        if name:
            ops.append((name, text))
    if len(ops) < 2:
        return get_run(r.id)  # 意見が2つも揃わんときは議論にならん

    # ② 全員が、全員の意見を読んで応答する（自分の意見も含む）
    resp = []
    for ai, _ in ops:
        name, text, _ = r.call("応答", ai, "critique",
                               f"テーマ: {topic}\n\n{opinions_text(ops)}\n\n"
                               f"あなたは【{ai}】です。\n{RESPOND_FMT}", fallback=False)
        if name:
            resp.append((name, text))

    # 残り回数を見ながら進める（最後の最終回答の分は必ず残す）
    if r.left() < 2:
        return get_run(r.id)

    # ③ 論点を整理（1回）
    discussion = opinions_text(ops) + "\n\n" + "\n\n".join(f"【{n}の応答】\n{t}" for n, t in resp)
    _, organized, _ = r.call("整理", pick("synthesize", 1)[0], "synthesize",
                             f"テーマ: {topic}\n\n{discussion}\n\n{ORGANIZE_FMT}")
    organized = organized or discussion  # 整理が失敗したら生の議論を次へ渡す

    # ④ 深掘り（最大2回。最終回答の分を残す）
    deep = []
    if r.left() >= 2:
        n1, t1, _ = r.call("深掘り", pick("critique", 1)[0], "critique",
                           f"テーマ: {topic}\n\n論点整理:\n{organized}\n\n{DEEPEN_FMT}")
        if n1:
            deep.append((n1, t1))
    if deep and r.left() >= 2:
        n2, t2, _ = r.call("深掘り", pick("critique", 1, exclude=[deep[0][0]])[0], "critique",
                           f"テーマ: {topic}\n\n前の深掘り:\n{deep[0][1]}\n\n{COUNTER_FMT}")
        if n2:
            deep.append((n2, t2))

    # ⑤ 最終回答を1つ
    deep_text = "\n\n".join(f"【深掘り by {n}】\n{t}" for n, t in deep) or "(深掘りなし)"
    if r.left() >= 1:
        r.call("最終回答", pick("synthesize", 1)[0], "synthesize",
               f"テーマ: {topic}\n\n議論の整理:\n{organized}\n\n深掘り:\n{deep_text}\n\n{FINAL_FMT}")
    return get_run(r.id)


@app.get("/api/run/{rid}")
def get_run(rid: int):
    run_ = q("SELECT * FROM runs WHERE id=?", (rid,))[0]
    msgs = q("SELECT * FROM msgs WHERE run_id=? ORDER BY id", (rid,))
    return {"run": run_, "msgs": msgs, "max_calls": MAX_CALLS}


@app.get("/api/runs")
def runs():
    return q("SELECT * FROM runs ORDER BY id DESC LIMIT 30")


class RateIn(BaseModel):
    msg_id: int
    good: bool


@app.post("/api/rate")
def rate(b: RateIn):
    m = q("SELECT ai,task FROM msgs WHERE id=?", (b.msg_id,))
    if m:
        q("INSERT INTO ratings(ai,task,good,ts) VALUES(?,?,?,?)", (m[0]["ai"], m[0]["task"], int(b.good), time.time()))
    return {"ok": bool(m)}


@app.post("/api/test/{name}")
def test(name: str):
    if name not in A:
        return {"ok": False}
    a = A[name]
    if not a.live():
        a.state = "未設定（モック）"
        return {"state": a.state}
    try:
        a.ask("OKとだけ返して")
        a.state = "接続OK"
    except Exception as e:
        a.state = f"失敗: {type(e).__name__}"
    return {"state": a.state}


@app.get("/api/status")
def status():
    return [{"name": n, "model": a.model if a.live() else None, "state": a.state,
             "scores": {t: round(score(n, t), 2) for t in PRIOR[n] if t in ("hypothesis", "critique", "synthesize")}}
            for n, a in A.items()]


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
