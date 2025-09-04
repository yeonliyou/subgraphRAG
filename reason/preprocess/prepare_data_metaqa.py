# preprocess/prepare_data_metaqa.py
import os
import re
import json
import torch
from typing import Any, Dict, List, Tuple, Optional

def _norm_str(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    return s

def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        # 여러 정답 구분자 허용
        for d in ["|", "####", "|||", ";", ",", "\t"]:
            if d in x:
                return [a.strip() for a in x.split(d) if a.strip()]
        return [x.strip()] if x.strip() else []
    if isinstance(x, (list, tuple, set)):
        return [str(a).strip() for a in x if str(a).strip()]
    return [str(x).strip()] if str(x).strip() else []

def _get_topk_from_prompt_mode(prompt_mode: str, default_k: int = 100) -> int:
    # "scored_100" -> 100
    m = re.search(r"scored[_\-]?(\d+)", str(prompt_mode).lower())
    return int(m.group(1)) if m else default_k

def _pick_triplets(obj: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    retrieval_result.pth 내에서 (h,r,t)[,score] 후보 키를 폭넓게 지원.
    """
    cand_keys = [
        "scored_triplets", "retrieved_triplets", "triplets",
        "good_triplets_rog", "graph", "subgraph"
    ]
    triples = None
    for k in cand_keys:
        if k in obj and obj[k]:
            triples = obj[k]
            break
    if triples is None:
        return []

    out = []
    for t in triples:
        # 가능한 형태들: (h,r,t), [h,r,t], (h,r,t,score), dict{"h":..,"r":..,"t":..}
        if isinstance(t, (list, tuple)):
            if len(t) >= 3:
                h, r, e = t[0], t[1], t[2]
                out.append((_norm_str(h), _norm_str(r), _norm_str(e)))
        elif isinstance(t, dict):
            h = t.get("h") or t.get("head") or t.get(0)
            r = t.get("r") or t.get("rel")  or t.get(1)
            e = t.get("t") or t.get("tail") or t.get(2)
            if h is not None and r is not None and e is not None:
                out.append((_norm_str(h), _norm_str(r), _norm_str(e)))
    return out

def _guess_hops_from_name(name: Optional[str], fallback: int = 3) -> int:
    m = re.search(r"(?:metaqa[\-_]?)?([123])hop", (name or "").lower())
    return int(m.group(1)) if m else fallback

def _extract_id(obj: Dict[str, Any], idx: int) -> Any:
    for k in ["id", "qid", "question_id", "uid", "sample_id"]:
        if k in obj:
            return obj[k]
    return idx  # 마지막 수단

def _extract_question(obj: Dict[str, Any]) -> Optional[str]:
    for k in ["question", "q", "query", "text", "input"]:
        if k in obj and obj[k]:
            return str(obj[k])
    return None

def _extract_answers(obj: Dict[str, Any]) -> List[str]:
    for k in ["answers", "answer", "ground_truth", "golds", "labels", "targets"]:
        if k in obj and obj[k]:
            return _as_list(obj[k])
    return []

def _ensure_triplet_list(x: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    # 빈 문자열 제거
    return [(h, r, t) for (h, r, t) in x if h and r and t]

def get_data_metaqa(score_dict_path: str,
                    split: str = "test",
                    prompt_mode: str = "scored_100",
                    dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    MetaQA 전용 로더.
    - 필수: score_dict_path(retrieval_result.pth). 여기에 id/question/answers/(triplets)가 있어야 함.
    - 출력: 기존 get_data()와 동일한 키를 가짐 (graph/good_triplets_rog/scored_triplets 등)
    """
    if not score_dict_path or not os.path.exists(score_dict_path):
        raise FileNotFoundError(
            f"[MetaQA] retrieval 결과(.pth)를 찾지 못했습니다: {score_dict_path}\n"
            f"'-p /path/to/retrieval_result.pth' 인자를 지정하세요."
        )

    obj = torch.load(score_dict_path, weights_only=False)
    # 형태 통일
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            raw_list = obj["data"]
        else:
            # id->sample dict
            raw_list = []
            for k, v in obj.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv.setdefault("id", k)
                    raw_list.append(vv)
    elif isinstance(obj, list):
        raw_list = obj
    else:
        raise ValueError(f"[MetaQA] 지원하지 않는 pth 포맷: type={type(obj)}")

    topk = _get_topk_from_prompt_mode(prompt_mode, default_k=100)
    hop = _guess_hops_from_name(dataset_name, fallback=3)

    data: List[Dict[str, Any]] = []
    no_trip = 0
    no_q = 0
    no_ans = 0

    for i, s in enumerate(raw_list):
        sid = _extract_id(s, i)
        q = _extract_question(s)
        a = _extract_answers(s)
        trips = _ensure_triplet_list(_pick_triplets(s))

        if q is None:
            no_q += 1
            continue
        if not a:
            no_ans += 1
        if not trips:
            no_trip += 1

        item = {
            "id": sid,
            "dataset": dataset_name or "metaqa",
            "split": split,
            "question": q,
            "ground_truth": a,
            # 프롬프트 유틸이 참조하는 키들
            "graph": trips,
            "good_paths_rog": [],
            "good_triplets_rog": trips[:topk],
            "scored_triplets": trips[:topk],
            # hop 정보 (참고용)
            "max_path_length": hop,
        }
        data.append(item)

    print(f"{len(data)} questions loaded from retrieval pth: {os.path.basename(score_dict_path)}")
    print("Adding scored triplets...")
    print(f"Triplets not found for {no_trip} questions; Questions w/o text: {no_q}; w/o answers: {no_ans}")
    return data
