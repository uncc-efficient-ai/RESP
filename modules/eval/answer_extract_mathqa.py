#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess MATHQA answers with the paper's local Qwen3 extractor.

Run with python -m modules.eval.answer_extract_mathqa --json_path results.json.
"""

import os
import sys
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import openai  # for exceptions
from openai import OpenAI

SYSTEM_INSTRUCTIONS = """
Return your answer in this EXACT format:
ANSWER: true/false
REASON: [your reasoning]

Task:
You are an expert evaluator for multiple-choice question answering tasks.
Given a question, a correct option with the specific value (target), and a model's solution, you must determine whether the solution is correct.
You should strictly judge correctness based on the meaning — if the solution matches the correct choice or expresses the same answer (even if written differently), mark it as true. Otherwise, mark it as false.
The model's solution may either mention the option letter (e.g., "B") or the actual value (e.g., "30 mph"). Both count as correct if they match in meaning.
Ignore case, punctuation, and minor phrasing differences when judging equivalence.
"""


# -------------------------------
# Helpers
# -------------------------------
def coerce_gsm8k_cot(x: Any) -> List[Dict[str, Any]]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


def coerce_resps(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return ["" if r is None else str(r) for r in x]
    return [str(x)]


def wait_for_http_ready(url: str, timeout_s: int = 120, interval_s: float = 1.0) -> bool:
    """Poll an HTTP endpoint until it returns a non-error response."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True
        except (URLError, HTTPError):
            pass
        time.sleep(interval_s)
    return False


from typing import List, Optional, TextIO


def stream_process_output(
        proc: subprocess.Popen,
        prefix: str = "[vLLM] ",
        tee_file: Optional[TextIO] = None,
):
    """
    将子进程的 stdout 实时打印到当前进程 stdout，并可选地同时写入文件。
    在独立 daemon 线程中运行，不会阻塞主线程。
    """

    def _run():
        try:
            # 逐行读，直到进程 stdout 关闭
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                msg = f"{prefix}{line.rstrip()}"
                print(msg, flush=True)
                if tee_file is not None:
                    tee_file.write(msg + "\n")
                    tee_file.flush()
        except Exception as e:
            print(f"{prefix}log stream error: {type(e).__name__}: {e}", flush=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t  # 如需在外部 join，可拿到线程句柄


def launch_vllm_openai_server(
        model: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        extra_args: List[str] = None,
        stream_logs: bool = True,  # 新增：是否实时打印 vLLM 日志
        log_file_path: Optional[str] = None,  # 新增：可选把日志也写到文件
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 42768,
) -> subprocess.Popen:
    """
    启动 vLLM 的 OpenAI 兼容服务: python -m vllm.entrypoints.openai.api_server ...
    返回 Popen 对象；若 stream_logs=True，会开启后台线程实时打印日志。
    """
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--dtype", dtype,
        "--tensor-parallel-size", str(tensor_parallel_size),
        # 可按需加入:
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        # "--max-num-batched-tokens", "32768",
        # "--enforce-eager"
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[vLLM] Launching: {' '.join(cmd)}", flush=True)

    # 注意：使用 PIPE 才能在当前进程里抓到并打印子进程 stdout
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,  # 文本模式
        bufsize=1,  # 行缓冲（配合 universal_newlines/text）
        universal_newlines=True,
    )

    # 实时打印日志（可选也写入文件）
    if stream_logs and proc.stdout is not None:
        tee_fp = open(log_file_path, "a", encoding="utf-8") if log_file_path else None
        stream_process_output(proc, prefix="[vLLM] ", tee_file=tee_fp)

    return proc


def graceful_terminate(proc: subprocess.Popen, timeout_s: int = 10):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            # 优先尝试 SIGINT，再尝试 SIGTERM
            if hasattr(signal, "SIGINT"):
                proc.send_signal(signal.SIGINT)
            time.sleep(1)
            if proc.poll() is None:
                if hasattr(signal, "SIGTERM"):
                    proc.terminate()
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def ask_true_false(
        client: OpenAI,
        model: str,
        target: str,
        resps: List[str],
        max_retries: int = 3
) -> Tuple[bool, str]:
    """
    使用 Chat Completions（vLLM 稳定支持）而非 Responses API。
    """
    payload = {
        "target": ("" if target is None else str(target)).strip(),
        "resps": [r.strip() for r in coerce_resps(resps)]
    }
    user_prompt = (
            "Return answer and reasoning in the specified format.\n"
            "DATA:\n" + json.dumps(payload, ensure_ascii=False)
    )

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS.strip()},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=150,
                stream=False,
            )
            content = (resp.choices[0].message.content or "").strip()

            answer = False
            reason = "No reason provided"
            for line in content.splitlines():
                s = line.strip()
                if s.lower().startswith("answer:"):
                    answer_text = s.split(":", 1)[1].strip().lower()
                    answer = (answer_text == "true")
                elif s.lower().startswith("reason:"):
                    reason = s.split(":", 1)[1].strip()

            return answer, reason

        except (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.APIStatusError) as e:
            if attempt == max_retries - 1:
                return False, f"API Error after retries: {type(e).__name__}: {e}"
            time.sleep(2 ** attempt)  # 指数退避
        except Exception as e:
            return False, f"Error: {type(e).__name__}: {e}"


def analyze_txt_file(file_path):
    """
    分析txt文件，统计总行数和true的占比
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = 0
        true_count = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue

            total_lines += 1

            # 解析每行数据，格式：数字,布尔值,数字,[...]
            try:
                # 找到第一个和第二个逗号的位置
                first_comma = line.find(',')
                second_comma = line.find(',', first_comma + 1)

                if first_comma == -1 or second_comma == -1:
                    print(f"警告：第{line_num}行格式不正确，跳过")
                    total_lines -= 1
                    continue

                # 提取布尔值部分
                bool_part = line[first_comma + 1:second_comma].strip()

                if bool_part.lower() == 'true':
                    true_count += 1
                elif bool_part.lower() == 'false':
                    pass  # false不需要特别处理
                else:
                    print(f"警告：第{line_num}行布尔值格式不正确: {bool_part}")

            except Exception as e:
                print(f"警告：第{line_num}行解析出错: {e}")
                total_lines -= 1
                continue

        # 计算统计结果
        if total_lines == 0:
            print("文件中没有有效的数据行")
            return

        true_percentage = (true_count / total_lines) * 100
        false_count = total_lines - true_count
        false_percentage = (false_count / total_lines) * 100

        # 输出结果
        print("=" * 50)
        print("文件分析结果")
        print("=" * 50)
        print(f"文件路径: {file_path}")
        print(f"总行数: {total_lines}")
        print(f"True 数量: {true_count}")
        print(f"False 数量: {false_count}")
        print(f"True 占比: {true_percentage:.2f}%")
        print(f"False 占比: {false_percentage:.2f}%")
        print("=" * 50)

        return {
            'total_lines': total_lines,
            'true_count': true_count,
            'false_count': false_count,
            'true_percentage': true_percentage,
            'false_percentage': false_percentage
        }

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"错误：读取文件时出现问题 {e}")


# -------------------------------
# Main
# -------------------------------
def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                    help="Model name passed to OpenAI client (should match vLLM model repo/name)")
    ap.add_argument("--out", default="boolean_results_local.txt", help="Output text file suffix")
    ap.add_argument("--workers", type=int, default=16, help="Number of concurrent worker threads")
    ap.add_argument("--max_retries", type=int, default=3, help="Max retries per API call")
    ap.add_argument("--json_path", default=None,
                    help="Path of the target json. If none will detect all json files in working folder")
    # vLLM 管理相关
    ap.add_argument("--no_launch_vllm", action="store_true",
                    help="Do not launch a local vLLM OpenAI server inside this script")
    ap.add_argument("--vllm_model", default=None,
                    help="HF repo or local path for vLLM (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)")
    ap.add_argument("--vllm_host", default=os.environ.get("VLLM_HOST", "127.0.0.1"))
    ap.add_argument("--vllm_port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    ap.add_argument("--vllm_dtype", default=os.environ.get("VLLM_DTYPE", "auto"))
    ap.add_argument("--vllm_tp", type=int, default=int(os.environ.get("VLLM_TP", "1")), help="tensor-parallel-size")
    ap.add_argument("--vllm_memory_usage", type=float, default=0.8, help="gpu-memory-utilization")
    ap.add_argument("--vllm_context_length", type=int, default=42768, help="max-model-len")
    ap.add_argument("--vllm_extra", nargs="*", default=None, help="Extra args for vLLM server")
    ap.add_argument("--ready_timeout", type=int, default=1000, help="Seconds to wait for vLLM HTTP ready")

    args = ap.parse_args(args)

    # 找到当前目录所有 JSON 文件
    if args.json_path is None:
        json_files = sorted(Path(".").glob("*.json"))
        if not json_files:
            print("当前目录没有找到任何 .json 文件")
            return
    else:
        if os.path.exists(args.json_path):
            json_files = [Path(args.json_path)]
        else:
            print(".json file not existed")
            return
    print(f"Json found: {json_files}")
    base_url_env = os.environ.get("OPENAI_BASE_URL")
    base_url = base_url_env or f"http://{args.vllm_host}:{args.vllm_port}/v1"

    vllm_proc = None
    try:
        if not args.no_launch_vllm:
            if not args.vllm_model:
                args.vllm_model = args.model
            vllm_proc = launch_vllm_openai_server(
                model=args.vllm_model,
                host=args.vllm_host,
                port=args.vllm_port,
                dtype=args.vllm_dtype,
                tensor_parallel_size=args.vllm_tp,
                max_model_len=args.vllm_context_length,
                gpu_memory_utilization=args.vllm_memory_usage,
                extra_args=args.vllm_extra,
            )
            ready = wait_for_http_ready(f"http://{args.vllm_host}:{args.vllm_port}/v1/models",
                                        timeout_s=args.ready_timeout)
            if not ready:
                print("[vLLM] Server did not become ready within timeout.")
                graceful_terminate(vllm_proc)
                sys.exit(2)
            else:
                print(f"[vLLM] Ready at {base_url}")

        client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "vllm"))

        # 依次处理所有 JSON 文件
        for input_path in json_files:
            print(f"\n==== Processing {input_path} ====")
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            samples = data.get("samples", [])
            jobs: List[Tuple[int, int, str, List[str]]] = []
            seen_doc_id = set()
            for si, sample in enumerate(samples):
                gsm_list = coerce_gsm8k_cot(samples[sample])  # ← 保持你原来的写法
                for oi, obj in enumerate(gsm_list):
                    doc_id = obj.get("doc_id", (si, oi))
                    if doc_id in seen_doc_id:
                        continue
                    seen_doc_id.add(doc_id)
                    target = obj.get("target", "")
                    resps = obj.get("resps", [])
                    jobs.append((si, oi, target, resps))

            total = len(jobs)
            if total == 0:
                print(f"{input_path.name}: No items to process.")
                continue

            # 多线程跑
            results_lines: List[str] = []
            results_lock = threading.Lock()
            print_lock = threading.Lock()

            def worker(si, oi, target, resps):
                verdict, reason = ask_true_false(
                    client=client, model=args.model,
                    target=target, resps=resps,
                    max_retries=args.max_retries
                )
                line = f"{si}-{oi},{str(verdict).lower()},{json.dumps(str(target), ensure_ascii=False)}," \
                       f"{json.dumps(coerce_resps(resps), ensure_ascii=False)}," \
                       f"{json.dumps(reason, ensure_ascii=False)}"
                with print_lock:
                    print(f"{input_path.name} {si}-{oi}: {verdict} - {reason}")
                return si, oi, line

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(worker, si, oi, target, resps) for si, oi, target, resps in jobs]
                for fut in as_completed(futs):
                    try:
                        _, _, line = fut.result()
                    except Exception as e:
                        line = f"0-0,false,\"\",[],\"WorkerError:{type(e).__name__}:{e}\""
                    with results_lock:
                        results_lines.append(line)

            results_lines.sort(key=lambda s: (int(s.split(",")[0].split("-")[0]),
                                              int(s.split(",")[0].split("-")[1])))
            output_path = str(input_path.parent / input_path.stem) + "_" + args.out
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(results_lines) + "\n")

            print(f"Done. total={total}. Saved -> {output_path}")

            stats = analyze_txt_file(output_path)
            if stats:
                stats_json_path = str(input_path.parent / input_path.stem) + "_stats.json"
                with open(stats_json_path, "w", encoding="utf-8") as sf:
                    json.dump(stats, sf, ensure_ascii=False, indent=2)
                print(f"Saved stats JSON -> {stats_json_path}")
    finally:
        if vllm_proc is not None:
            graceful_terminate(vllm_proc)
    return stats

if __name__ == "__main__":
    main()
