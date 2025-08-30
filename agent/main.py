## 사용법
# # 1) 의존성
# pip install openai tiktoken numpy
# 
# # 2) 키
# export OPENAI_API_KEY=sk-...
# 
# # 3) 실행 예
# python agent/main.py "ITK로 nii.gz 읽고 median 필터 후 저장하는 코드"
# # (옵션) 풀 색인 일부도 같이 주입
# python agent/main.py --use-index "RTK FDK 파이프라인 스켈레톤 만들어줘"

#!/usr/bin/env python3
import os, sys, json, re, math, argparse, hashlib, time
from pathlib import Path
from typing import List, Tuple, Dict

try:
    import tiktoken  # pip install tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    tiktoken = None
    _HAS_TIKTOKEN = False
import numpy as np
try:
    from openai import OpenAI  # pip install openai
    _HAS_OPENAI = True
except Exception:
    OpenAI = None
    _HAS_OPENAI = False

# --------- CONFIG ---------
ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    ROOT/"docs/AgentStyle.md",
    ROOT/"docs/ITK_API.md",
    ROOT/"docs/VTK_API.md",
    ROOT/"docs/RTK_API.md",
]
INDEX_DIR = ROOT/"docs_api_index"            # (선택) 풀 색인
CACHE_DIR = ROOT/"agent/.rag_cache"          # 임베딩 캐시
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optional web RAG (config-overridable)
WEB_CFG = {
    "enabled": False,
    "provider": "bing",
    "allow_domains": [],
    "max_results": 5,
    "timeout_s": 8,
}

# --- Config-driven doc selection (AGENTS.md, persona_path, extra_docs) ---
# Preserve backward compatibility: compute DOC_PATHS dynamically.
DEFAULT_DOCS = [
    Path("docs/AgentStyle.md"),
    Path("docs/ITK_API.md"),
    Path("docs/VTK_API.md"),
    Path("docs/RTK_API.md"),
]

def _load_config() -> dict:
    cfg_path = ROOT/"configs/agent.config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _to_abs(p: Path | str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT/p)

def _compute_doc_paths() -> List[Path]:
    cfg = _load_config()

    # 1) Auto-include root AGENTS.md with highest priority
    ordered: List[Path] = []
    agents_md = ROOT/"AGENTS.md"
    if agents_md.exists():
        ordered.append(agents_md)

    # 2) Include persona_path from config if present
    persona = cfg.get("persona_path")
    if isinstance(persona, str) and persona.strip():
        persona_path = _to_abs(persona)
        if persona_path.exists():
            ordered.append(persona_path)

    # 3) Include default docs
    for rel in DEFAULT_DOCS:
        p = _to_abs(rel)
        if p.exists():
            ordered.append(p)

    # 4) Include extra_docs from config if any
    extras = cfg.get("extra_docs") or []
    if isinstance(extras, list):
        for e in extras:
            try:
                ep = _to_abs(e)
                if ep.exists():
                    ordered.append(ep)
            except Exception:
                continue

    # De-duplicate while preserving order (earlier has higher priority)
    seen = set()
    deduped: List[Path] = []
    for p in ordered:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    return deduped

# Override DOC_PATHS with computed result
try:
    DOC_PATHS = _compute_doc_paths()
except Exception:
    # Fallback to existing DOC_PATHS if computation fails
    DOC_PATHS = DOC_PATHS

# Load web config if present
try:
    _cfg = _load_config()
    if isinstance(_cfg, dict) and isinstance(_cfg.get("web_search"), dict):
        wc = _cfg["web_search"]
        WEB_CFG.update({
            "enabled": bool(wc.get("enabled", WEB_CFG["enabled"])),
            "provider": wc.get("provider", WEB_CFG["provider"]),
            "allow_domains": wc.get("allow_domains", WEB_CFG["allow_domains"]),
            "max_results": int(wc.get("max_results", WEB_CFG["max_results"])),
            "timeout_s": int(wc.get("timeout_s", WEB_CFG["timeout_s"]))
        })
except Exception:
    pass

EMBED_MODEL = "text-embedding-3-large"       # 고성능 임베딩 (다국어↑)  :contentReference[oaicite:1]{index=1}
GEN_MODEL   = "gpt-5.1"                      # 예시: 최신 챗/리즌 모델(원하는 모델로 교체)  :contentReference[oaicite:2]{index=2}

# Responses API 권장(통합형). Chat Completions을 써도 됨.  :contentReference[oaicite:3]{index=3}

# Override models from config if present
try:
    _cfg_models = _load_config()
    if isinstance(_cfg_models, dict):
        EMBED_MODEL = _cfg_models.get("embed_model", EMBED_MODEL)
        GEN_MODEL = _cfg_models.get("model", GEN_MODEL)
except Exception:
    pass

# --------- UTILS: chunking / token count ----------
def tokenize_len(text, model="gpt-4o"):  # 토큰 근사치
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

# Redefine with fallback when tiktoken is unavailable
def tokenize_len(text, model="gpt-4o"):  # fallback-safe
    if globals().get('_HAS_TIKTOKEN'):
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    return max(1, len(text.split()))

def md_to_chunks(text: str, max_tokens=600, overlap_tokens=80) -> List[str]:
    # 헤더 기준 우선 분할 → 길면 슬라이딩 윈도우로 세분화
    parts = re.split(r"(?m)^#{1,6}\s", text)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if tokenize_len(p) <= max_tokens:
            chunks.append(p)
        else:
            toks = re.split(r"(\s+)", p)  # 단어 경계 기준
            buf, t = [], 0
            for w in toks:
                buf.append(w); t = tokenize_len("".join(buf))
                if t > max_tokens:
                    chunk = "".join(buf[:-1]).strip()
                    if chunk: chunks.append(chunk)
                    # overlap
                    back = []
                    while buf and tokenize_len("".join(back)) < overlap_tokens:
                        back.insert(0, buf.pop())
                    buf = back
            if buf:
                chunks.append("".join(buf).strip())
    return chunks

def file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]

# --------- OPENAI CLIENT ----------
def get_client() -> OpenAI:
    # 환경변수 OPENAI_API_KEY 필요
    if not _HAS_OPENAI:
        raise RuntimeError("OpenAI SDK is not installed")
    return OpenAI()

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    # 배치 임베딩
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)  # :contentReference[oaicite:4]{index=4}
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vecs)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# --------- RAG INDEX (BUILD/LOAD) ----------
def build_or_load_index() -> Dict:
    """
    구조:
    {
      "docs": [{"path": "...", "fp": "abcd", "chunks": [str,...]}],
      "embeds": np.ndarray (N, D) as list[list[float]]
      "meta": [{"doc_i": int, "chunk_i": int, "title": "ITK_API.md#...", "source": "docs/ITK_API.md"}, ...]
    }
    """
    client = get_client()
    cache_key = "core_" + hashlib.sha256("".join(str(p) for p in DOC_PATHS).encode()).hexdigest()[:16]
    cache_file = CACHE_DIR/(cache_key+".json")

    # 캐시 유효성: 파일 핑거프린트 비교
    need_rebuild = True
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            fps_old = [d["fp"] for d in data["docs"]]
            fps_now = [file_fingerprint(p) for p in DOC_PATHS]
            if fps_old == fps_now:
                need_rebuild = False
                return {
                    **data,
                    "embeds": np.array(data["embeds"], dtype=np.float32),
                }
        except Exception:
            pass

    # 재빌드
    docs_info = []
    all_chunks, meta = [], []
    for p in DOC_PATHS:
        txt = p.read_text(encoding="utf-8")
        chunks = md_to_chunks(txt)
        base = p.name
        for i, c in enumerate(chunks):
            # 메타 제목(대충 첫 헤더/첫 줄)
            first_line = c.splitlines()[0][:80]
            meta.append({"doc": base, "chunk_idx": i, "title": first_line, "source": str(p)})
        docs_info.append({"path": str(p), "fp": file_fingerprint(p), "chunks": len(chunks)})
        all_chunks.extend(chunks)

    embeds = embed_texts(client, all_chunks)

    payload = {
        "docs": docs_info,
        "embeds": embeds.tolist(),
        "chunks": all_chunks,
        "meta": meta,
        "built_at": time.time(),
        "model": EMBED_MODEL,
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    payload["embeds"] = embeds
    return payload

# (선택) 풀 색인도 같은 방식으로 보강 검색
def optional_index_search(query: str, topk=2) -> List[str]:
    if not INDEX_DIR.exists(): return []
    # 매우 단순한 키워드 점수
    qws = set(re.findall(r"\w+", query.lower()))
    scored = []
    for p in INDEX_DIR.glob("*.md"):
        text = p.read_text(encoding="utf-8")
        score = sum(text.lower().count(w) for w in qws)
        if score > 0:
            scored.append((score, p, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [scored[i][2][:8000] for i in range(min(topk, len(scored)))]

def optional_web_search(query: str, provider: str, allow_domains: List[str], max_results: int, timeout_s: int) -> List[str]:
    try:
        from .rag_web import web_retrieve
    except Exception:
        return []
    return web_retrieve(query, provider=provider, allow_domains=allow_domains, max_results=max_results, timeout_s=timeout_s)

# --------- RETRIEVE ----------
def retrieve_chunks(index: Dict, query: str, topk=6) -> List[Tuple[str, Dict, float]]:
    client = get_client()
    qv = embed_texts(client, [query])[0]
    sims = [cosine_sim(qv, v) for v in index["embeds"]]
    # 상위 topk
    top_ix = np.argsort(sims)[::-1][:topk]
    hits = []
    for j in top_ix:
        meta = index["meta"][j]
        hits.append((index["chunks"][j], meta, sims[j]))
    return hits

# --------- GENERATE ----------
SYSTEM_PROMPT = """You are a domain-specific Python coding agent for ITK/VTK/RTK.
Follow AgentStyle (style/guardrails). Prefer functions/classes from the Quick Refs.
Return a runnable Minimal Working Example first; add short Korean explanation.
If info is missing, assume safe defaults and mark TODOs at the top of the code."""

def call_llm(user_query: str, retrieved_blobs: List[str]) -> str:
    client = get_client()

    context_block = "\n\n".join(
        f"### Context {i+1}\n{blob}" for i, blob in enumerate(retrieved_blobs)
    )

    # Responses API 예시 (Chat Completions를 쓰려면 /chat 엔드포인트 사용)  :contentReference[oaicite:5]{index=5}
    resp = client.responses.create(
        model=GEN_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"[PROJECT CONTEXT]\n{context_block}\n\n[QUERY]\n{user_query}"}
        ],
        temperature=0.2,
    )
    # 본문 텍스트 추출
    return resp.output_text

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="RAG-powered ITK/VTK/RTK agent")
    ap.add_argument("query", nargs="*", help="질의(자연어). 비면 예시 실행")
    ap.add_argument("--k", type=int, default=6, help="retrieval top-k")
    ap.add_argument("--use-index", action="store_true", help="docs_api_index/* 일부도 함께 주입")
    ap.add_argument("--web", action="store_true", help="도메인 허용목록 기반 웹 RAG 활성화")
    ap.add_argument("--web-provider", default=None, help="웹 검색 제공자(bing)")
    ap.add_argument("--max-web-results", type=int, default=None, help="웹 검색 결과 상한")
    ap.add_argument("--allow-domain", action="append", default=None, help="허용 도메인 추가(중복 호출 가능)")
    args = ap.parse_args()

    q = " ".join(args.query) or "VTK로 STL 읽고 노멀 재계산 후 저장하는 코드를 만들어줘."
    idx = build_or_load_index()
    hits = retrieve_chunks(idx, q, topk=args.k)

    retrieved = [h[0] for h in hits]
    if args.use_index:
        retrieved += optional_index_search(q, topk=2)
    # optional web
    if args.web or WEB_CFG.get("enabled"):
        prov = args.web_provider or WEB_CFG.get("provider", "bing")
        maxr = args.max_web_results or WEB_CFG.get("max_results", 5)
        allow = list(WEB_CFG.get("allow_domains") or [])
        if args.allow_domain:
            allow.extend(args.allow_domain)
        # de-dup while preserving order
        seen = set(); merged = []
        for d in allow:
            if d not in seen:
                seen.add(d); merged.append(d)
        retrieved += optional_web_search(q, provider=prov, allow_domains=merged, max_results=maxr, timeout_s=WEB_CFG.get("timeout_s", 8))

    answer = call_llm(q, retrieved)
    print(answer)

if __name__ == "__main__":
    # OPENAI_API_KEY 환경변수 필요
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY first.", file=sys.stderr)
        sys.exit(1)
    main()
