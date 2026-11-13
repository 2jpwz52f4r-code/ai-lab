"""AI Research Lab (無料API版・第1段階)
独立仮説 -> 反論 -> 要約 の流れを、得意分野スコアで動的に割り振って実行する。
鍵が無いAIは自動でモック動作になり、画面にも「モック」と表示される。
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
MAX_CALLS = int(os.getenv("MAX_CALLS", "12"))  # 1回の研究での呼び出し上限
TIMEOUT = 45
lock = threading.Lock()

# ---- 得意分野の初期値（クロさんの仮説。固定ではなく、👍👎の評価で上書きされる）----
# gemini: 長文を扱えて要約・整理向き / groq(Llama): 速いので反論・ツッコミ向き
# mistral: 別系統のモデルなので独立した視点向き / openrouter: 予備・入れ替え枠
PRIOR = {
    "gemini":     {"hypothesis": .60, "critique": .60, "synthesize": .80},
    "groq":       {"hypothesis": .50, "critique": .70, "synthesize": .50},
    "mistral":    {"hypothesis": .65, "critique": .60, "synthesize": .60},
    "openrouter": {"hypothesis": .50, "critique": .50, "synthesize": .50},
}
BASE = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",  # 要・公式で確認
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class Adapter:
    """1社ぶんの接続。鍵とモデル名は .env から。モデル名は勝手に決めない。"""

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
            return f"[モック:{self.name}] 「{prompt[:40]}…」への仮の回答です。", True
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        r = httpx.post(self.base + "/chat/completions", timeout=TIMEOUT,
                       headers={"Authorization": "Bearer " + self.key},
                       json={"model": self.model, "messages": msgs})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], False


A = {n: Adapter(n) for n in BASE}

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

    def call(self, step, ai, task, prompt, system=""):
        """上限チェック・失敗時は次のAIへ。全部記録する。"""
        for name in [ai] + [a for a in pick(task, len(A)) if a != ai]:
            with lock:
                if self.calls >= MAX_CALLS:
                    return None, None, None
                self.calls += 1
            try:
                text, mock = A[name].ask(prompt, system)
                ok = 1
            except Exception as e:  # 429など。別のAIに回す
                text, mock, ok = f"エラー: {type(e).__name__} {getattr(e, 'response', None) and e.response.status_code or ''}", False, 0
            mid = q("INSERT INTO msgs(run_id,step,ai,task,content,mock,ok,ts) VALUES(?,?,?,?,?,?,?,?)",
                    (self.id, step, name, task, text, int(mock), ok, time.time()), ins=True)
            q("UPDATE runs SET calls=? WHERE id=?", (self.calls, self.id))
            if ok:
                return name, text, mid
        return None, None, None


FORMAT = "次の項目で簡潔に: 仮説 / 根拠 / 前提 / 予測 / 反証できる条件 / 不確実な点。事実・推論・仮説を区別して書く。"

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
    # STEP2: 互いの回答を見せずに仮説（独立）
    hyps = []
    for ai in pick("hypothesis", 3):
        out = r.call("hypothesis", ai, "hypothesis", f"テーマ: {r.topic}\n{FORMAT}")
        if out[0]:
            hyps.append(out)
    # STEP4: 反論（自分以外で反論が得意なAI。なるべく担当を散らす）
    used, crits = set(), []
    for name, text, _ in hyps:
        cands = pick("critique", len(A), exclude=[name])
        critic = next((a for a in cands if a not in used), cands[0])
        used.add(critic)
        out = r.call("critique", critic, "critique",
                     f"次の仮説の論理の穴、見落とした前提、反例、別の説明、検証方法を指摘して。賛成だけはしない。\n\n{text}")
        if out[0]:
            crits.append((name, out))
    # 要約: 得意なAIが整理。「発見」とは呼ばせない
    body_txt = "\n\n".join(f"【仮説 by {n}】\n{t}" for n, t, _ in hyps) + "\n\n" + \
               "\n\n".join(f"【{h}への反論 by {o[0]}】\n{o[1]}" for h, o in crits)
    synth = pick("synthesize", 1)[0]
    r.call("synthesize", synth, "synthesize",
           "以下を整理して。1)合意点 2)対立点 3)未解決 4)次に調べること。"
           "事実/推論/仮説を分け、これは検証済みの結果ではなく未確認の仮説として扱うこと。\n\n" + body_txt)
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
