import os
import re
import json
import torch
import wandb
import argparse
from tqdm import tqdm
from pathlib import Path

from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import get_prompts_for_data
from llm_utils import llm_init, llm_inf_all

from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original

from preprocess.prepare_data_metaqa import get_data_metaqa  # MetaQA 전용 로더


# ==== MetaQA align helpers ====

def _norm_id(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r'^(train_|dev_|valid_|val_|test_)', '', s)   # 접두 제거
    s = re.sub(r'^0+([0-9]+)$', r'\1', s)                    # 0패딩 제거
    return s

def _to_scalar(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.item() if x.ndim == 0 else x.detach().cpu().tolist()
    except Exception:
        pass
    return x

def _triples_from(sample: dict):
    # 리트리벌 내부의 다양한 필드명 지원
    for k in ["scored_triplets", "scored_triples", "triplets", "triples", "retrieved_triples", "topk_triples"]:
        if k in sample and sample[k] is not None:
            return sample[k]
    return []

def _answers_from(sample: dict):
    # 가능한 정답 키를 폭넓게 수용
    for k in ["ground_truth", "answers", "answer", "a_entity", "a_entities", "answer_entities"]:
        if k in sample and sample[k] is not None:
            v = sample[k]
            if isinstance(v, (list, tuple, set)):
                return [str(x) for x in v]
            return [str(v)]
    if "a_entity_in_graph" in sample and sample["a_entity_in_graph"]:
        v = sample["a_entity_in_graph"]
        return [str(x) for x in v] if isinstance(v, (list, tuple, set)) else [str(v)]
    return []

def _normalize_triples(tris):
    norm = []
    for t in tris:
        if len(t) >= 3:
            h, r, t2 = map(_to_scalar, t[:3])
            s = _to_scalar(t[3]) if len(t) >= 4 else None
            norm.append((str(h), str(r), str(t2), float(s) if isinstance(s, (int, float)) else None))
    return norm

def attach_metaqa_info(data, score_dict_path):
    """
    retrieval pth의 triples/answers를 data의 각 샘플에 부착
    - triples: 표준 키 'scored_triplets'로 통일 (list[(h,r,t,score)])
    - answers: 표준 키 'ground_truth'로 통일 (list[str])
    """
    R = torch.load(score_dict_path, map_location="cpu")

    # 리트리벌 키 정규화 인덱스
    index = {}
    for k in R.keys():
        nk = _norm_id(k)
        index[nk] = k
        index[f"test_{nk}"] = k
        only_num = re.sub(r'\D', '', nk)
        if only_num:
            index[only_num] = k

    attached, miss, no_ans = 0, 0, 0
    for ex in data:
        did = _norm_id(ex.get("id", ""))
        hit = None
        for cand in [did, f"test_{did}", re.sub(r'\D', '', did)]:
            if cand and cand in index:
                hit = R[index[cand]]
                break

        if hit is None:
            ex["scored_triplets"] = []
            ex["ground_truth"] = []
            miss += 1
            continue

        # triples 부착 (표준 키로 통일)
        tris = _triples_from(hit)
        ex["scored_triplets"] = _normalize_triples(tris)

        # answers 부착 (ground_truth로 통일)
        gt = _answers_from(hit)
        ex["ground_truth"] = gt
        if not gt:
            no_ans += 1

        attached += 1

    print(f"[align] attached triples to {attached} / {len(data)} samples; misses={miss}; no_ground_truth={no_ans}")
    return data

# ==== /MetaQA align helpers ====


def get_defined_prompts(prompt_mode, model_name, llm_mode):
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt


def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data) + "\n")


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        with open(file_path, "r") as f:
            ckpt = [json.loads(line) for line in f]
        try:
            print(f"Last processed item: {ckpt[-1]['id']}")
        except IndexError:
            pass
        print("*" * 50)
        return ckpt
    return []


def eval_all(pred_file_path, run, subset, split=None, eval_hops=-1):
    print("=" * 50)
    print("=" * 50)
    print(f"Evaluating on subset: {subset}")
    print("Results:")
    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = \
        eval_results_corrected(str(pred_file_path), cal_f1=True, subset=subset, split=split, eval_hops=eval_hops)
    postfix = "_sub" if subset else ""
    run.log({f"results{postfix}/hit@1": hit1,
             f"results{postfix}/macro_f1": f1,
             f"results{postfix}/macro_precision": prec,
             f"results{postfix}/macro_recall": recall,
             f"results{postfix}/exact_match": em,
             f"results{postfix}/totally_wrong": tw,
             f"results{postfix}/micro_f1": mi_f1,
             f"results{postfix}/micro_precision": mi_prec,
             f"results{postfix}/micro_recall": mi_recall,
             f"results{postfix}/total_cnt": total_cnt,
             f"results{postfix}/no_ans_cnt": no_ans_cnt,
             f"results{postfix}/no_ans_ratio": no_ans_ratio,
             f"results{postfix}/hal_score": hal_score})
    if stats is not None:
        for k, v in stats.items():
            run.log({f"stats{postfix}/{k}": v})
    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    run.log({f"results{postfix}/hit": hit})
    print("=" * 50)
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="cwq", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str)
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0.16, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    tensor_parallel_size = args.tensor_parallel_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres

    # MetaQA면 평가기에 retrieval pth 경로 전달
    if "metaqa" in dataset_name.lower() and args.score_dict_path:
        os.environ["METAQA_SCORED_TRIPLES"] = str(args.score_dict_path)

    # MetaQA hop 자동
    hop = -1
    dn = dataset_name.lower()
    if "metaqa-3hop" in dn: hop = 3
    elif "metaqa-2hop" in dn: hop = 2
    elif "metaqa-1hop" in dn: hop = 1

    pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    run_name = f"{model_name}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}"
    run = wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=args)

    # 점수/리트리벌 파일 경로
    if args.score_dict_path is None:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
        else:
            score_dict_path = None
    else:
        score_dict_path = args.score_dict_path

    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    raw_pred_file_path = raw_pred_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-predictions-resume.jsonl"

    llm = llm_init(model_name, tensor_parallel_size, max_seq_len_to_capture, max_tokens, seed, temperature, frequency_penalty)

    # 데이터 로드
    if dataset_name.lower().startswith("metaqa"):
        if not score_dict_path:
            raise FileNotFoundError("[MetaQA] '-p /path/to/retrieval_result.pth' 를 지정하세요.")
        data = get_data_metaqa(score_dict_path=score_dict_path, split=split, prompt_mode=prompt_mode, dataset_name=dataset_name)
    else:
        data = get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode)

    # MetaQA: triples/ground_truth 부착
    if dataset_name.lower().startswith("metaqa"):
        data = attach_metaqa_info(data, score_dict_path)

    # 프롬프트 준비
    sys_prompt, cot_prompt = get_defined_prompts(prompt_mode, model_name, llm_mode)
    print("Generating prompts...")
    data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)

    # (옵션) 앞 2개만 프리뷰
    for i, ex in enumerate(data[:2]):
        print(f"\n===== Sample {i} =====")
        print("Q:", ex.get("question"))
        print("GT:", ex.get("ground_truth"))
        print("Triples count:", len(ex.get("scored_triplets", [])))
        if ex.get("scored_triplets"):
            print("One triple:", ex["scored_triplets"][0])
        preview = ex.get("all_query") or ex.get("user_query") or ex.get("sys_query")
        print("Prompt preview:\n", preview[:600] if isinstance(preview, str) else "<not string>")
        if i == 1:
            break

    print("Starting inference...")
    start_idx = len(load_checkpoint(raw_pred_file_path))
    with open(raw_pred_file_path, "a") as pred_file:
        for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
            res = llm_inf_all(llm, each_qa, llm_mode, model_name)
            # 안전 삭제
            for k in ("graph", "good_paths_rog", "good_triplets_rog", "scored_triplets"):
                if k in each_qa:
                    del each_qa[k]
            each_qa["prediction"] = res[0]
            save_checkpoint(pred_file, each_qa)

    final_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix)
    os.rename(raw_pred_file_path, final_pred_file_path)

    # 평가 (MetaQA면 hop 필터 적용)
    eval_all(final_pred_file_path, run, subset=True,  split=split, eval_hops=-1)
    eval_all(final_pred_file_path, run, subset=False, split=split, eval_hops=-1)


if __name__ == "__main__":
    main()
