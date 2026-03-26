# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def _apply_eval_llm_fallback(res, solution_str, ground_truth, extra_info):
    """Apply LLM fallback for eval routes. Always active unless DISABLE_LLM_FALLBACK=true."""
    import os
    # Extract score from result
    if isinstance(res, dict):
        score = float(res.get("score", res.get("acc", 0)))
    elif isinstance(res, (int, float, bool)):
        score = float(res)
    else:
        score = float(res[0])

    if score > 0:
        if isinstance(res, dict):
            res["llm_corrected"] = 0.0
        else:
            res = {"score": score, "acc": score, "llm_corrected": 0.0}
        return res

    # Skip LLM fallback if disabled (pure rule-based mode)
    if os.environ.get("DISABLE_LLM_FALLBACK", "false").lower() == "true":
        if isinstance(res, dict):
            res["llm_corrected"] = 0.0
        else:
            res = {"score": score, "acc": score, "llm_corrected": 0.0}
        return res

    # Rule-based says wrong → try LLM fallback
    try:
        from . import llm_answer_grading as _lag
        llm_res = _lag.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
        llm_score = float(llm_res["score"]) if isinstance(llm_res, dict) else float(llm_res)
        if llm_score > 0:
            print(f"LLM JUDGE CORRECTION (eval): ORM=0 -> LLM=1 | gt={ground_truth}")
            return {"score": 1.0, "acc": 1.0, "llm_corrected": 1.0}
    except Exception as e:
        print(f"LLM JUDGE ERROR (eval): {e}")

    if isinstance(res, dict):
        res["llm_corrected"] = 0.0
    else:
        res = {"score": score, "acc": score, "llm_corrected": 0.0}
    return res


def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, daytona_api_key=None):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    reward_metric = extra_info.get("reward_metric", None) if extra_info else None

    # === Training routes ===
    # Training: numina_math (ORM + optional LLM fallback)
    # Control via env var: USE_LLM_VERIFIER (true/false)
    if data_source in ["numina_math"]:
        import os
        use_llm_verifier = os.environ.get("USE_LLM_VERIFIER", "true").lower() == "true"
        # print(f"USE_LLM_VERIFIER: {use_llm_verifier}")
        from . import prime_math as _pm
        # ORM: rule-based answer reward (Reasoning360 aligned)
        answer_res = _pm.compute_score(solution_str, ground_truth)
        answer_score = float(answer_res[0]) if not isinstance(answer_res, dict) else float(answer_res["score"])

        llm_corrected = False
        if use_llm_verifier and answer_score == 0:
            # ORM says wrong, use LLM as fallback to double-check
            from . import llm_answer_grading as _lag
            try:
                llm_res = _lag.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
                llm_score = float(llm_res["score"]) if isinstance(llm_res, dict) else float(llm_res)
                if llm_score > 0:
                    # LLM says correct, override ORM result
                    answer_score = 1.0
                    llm_corrected = True
                    print(f"LLM JUDGE CORRECTION: ORM=0 -> LLM=1 | gt={ground_truth}")
            except Exception as e:
                # LLM call failed, keep ORM result
                print(f"LLM JUDGE ERROR: {e}")

        res = {
            "score": answer_score,
            "acc": answer_score,
            "answer_reward": answer_score,
            "llm_corrected": 1.0 if llm_corrected else 0.0,
        }
    # LLM proof grading (with reference solution)
    elif data_source in ["numina_math_process", "proofbench", "aops"]:
        import os
        use_dual = os.environ.get("USE_DUAL_OBJECTIVE", "false").lower() == "true"
        use_mult = os.environ.get("USE_MULT_REWARD", "false").lower() == "true"
        use_prm_reward = os.environ.get("USE_PRM_REWARD", "false").lower() == "true"

        if use_prm_reward and data_source == "numina_math_process":
            # Pure PRM baseline: LLM proof grading on ALL responses
            prm_rubric = os.environ.get("PRM_RUBRIC", "default").lower()
            if prm_rubric == "imobench":
                from . import llm_proof_grading_imobench as _prm_module
            else:
                from . import llm_proof_grading as _prm_module
            res = _prm_module.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
        elif (use_dual or use_mult) and data_source == "numina_math_process":
            # Shared ORM + PRM computation for dual-objective and multiplicative ablation
            from . import prime_math as _pm
            answer_res = _pm.compute_score(solution_str, ground_truth)
            rule_score = float(answer_res[0]) if not isinstance(answer_res, dict) else float(answer_res["score"])

            llm_corrected = False
            use_llm_verifier = os.environ.get("USE_LLM_VERIFIER", "true").lower() == "true"
            if use_llm_verifier and rule_score == 0:
                from . import llm_answer_grading as _lag
                try:
                    llm_res = _lag.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
                    llm_score = float(llm_res["score"]) if isinstance(llm_res, dict) else float(llm_res)
                    if llm_score > 0:
                        rule_score = 1.0
                        llm_corrected = True
                        print(f"LLM JUDGE CORRECTION (dual): ORM=0 -> LLM=1 | gt={ground_truth}")
                except Exception as e:
                    print(f"LLM JUDGE ERROR (dual): {e}")

            # PRM score (only for correct answers — saves ~50% LLM calls)
            prm_score = 0.0
            prm_rubric = os.environ.get("PRM_RUBRIC", "default").lower()
            if rule_score == 1.0:
                try:
                    if prm_rubric == "imobench":
                        from . import llm_proof_grading_imobench as _prm_mod
                    else:
                        from . import llm_proof_grading as _prm_mod
                    prm_res = _prm_mod.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
                    prm_score = float(prm_res["score"]) if isinstance(prm_res, dict) else float(prm_res)
                except Exception as e:
                    print(f"LLM PROOF GRADING ERROR (dual): {e}")
                    prm_score = 1.0  # fallback: correct answer, assume full process score

            if use_mult:
                # Ablation: ORM × PRM as single reward, standard GRPO handles normalization
                res = {
                    "score": rule_score * prm_score,
                    "acc": rule_score == 1.0,
                    "llm_corrected": 1.0 if llm_corrected else 0.0,
                }
            else:
                # PA_GRPO: separate fields for dual-objective advantage
                res = {
                    "score": rule_score,       # binary 0/1 → token_level_rewards → A_out
                    "acc": rule_score == 1.0,
                    "reward_prm": prm_score,   # 0/0.5/1 → non_tensor_batch → A_proc
                    "llm_corrected": 1.0 if llm_corrected else 0.0,
                }
        elif data_source == "numina_math_process":
            # Baseline: ORM + optional LLM fallback (binary 0/1), no PRM
            # Same logic as numina_math route
            from . import prime_math as _pm
            answer_res = _pm.compute_score(solution_str, ground_truth)
            answer_score = float(answer_res[0]) if not isinstance(answer_res, dict) else float(answer_res["score"])

            llm_corrected = False
            use_llm_verifier = os.environ.get("USE_LLM_VERIFIER", "true").lower() == "true"
            if use_llm_verifier and answer_score == 0:
                from . import llm_answer_grading as _lag
                try:
                    llm_res = _lag.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
                    llm_score = float(llm_res["score"]) if isinstance(llm_res, dict) else float(llm_res)
                    if llm_score > 0:
                        answer_score = 1.0
                        llm_corrected = True
                        print(f"LLM JUDGE CORRECTION (baseline): ORM=0 -> LLM=1 | gt={ground_truth}")
                except Exception as e:
                    print(f"LLM JUDGE ERROR (baseline): {e}")

            res = {
                "score": answer_score,
                "acc": answer_score,
                "answer_reward": answer_score,
                "llm_corrected": 1.0 if llm_corrected else 0.0,
            }
        else:
            # Original: LLM proof grading only (proofbench, aops)
            from . import llm_proof_grading
            res = llm_proof_grading.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
    # === Eval routes (aligned with Reasoning360 rule-based) ===
    # math* prefix → naive_dapo with reward_metric sub-dispatch (Reasoning360)
    elif data_source.startswith("math"):
        if reward_metric == "prime_math":
            from . import prime_math
            res = prime_math.compute_score(solution_str, ground_truth)
            res = _apply_eval_llm_fallback(res, solution_str, ground_truth, extra_info)
        elif reward_metric == "math_llm_judge":
            from . import llm_answer_grading
            res = llm_answer_grading.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
        else:
            from . import naive_dapo
            res = naive_dapo.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
            res = _apply_eval_llm_fallback(res, solution_str, ground_truth, extra_info)
    # code generation
    elif data_source.startswith('codegen'):
        #yifan: add daytona check. this is remote
            from . import coder1
            res = coder1.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # simulation (code)
    elif data_source.startswith("simulation__codeio"):
        from . import codeio
        res = codeio.compute_score(solution_str, ground_truth)
    elif data_source.startswith("simulation__cruxeval"):
        from . import cruxeval
        res = cruxeval.compute_score(solution_str, ground_truth)
    # logic
    elif data_source.startswith("simulation__arcagi") or data_source.startswith("simulation__barc"):
        from . import arcagi
        res = arcagi.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__arcagi") or data_source.startswith("logic__barc"):
        # Added for guru dataset compatibility
        from . import arcagi
        res = arcagi.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__zebra_puzzle"):
        from . import zebra_puzzle
        res = zebra_puzzle.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__ordering_puzzle"):
        from . import puzzles_dataset
        res = puzzles_dataset.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__graph"):
        from . import graph_dataset
        res = graph_dataset.compute_score(solution_str, ground_truth)
    # table
    elif data_source.startswith("table"):
        # TODO: tmp placeholder using math_verify
        from . import tablereason
        res = tablereason.compute_score(solution_str, ground_truth)
    elif data_source.startswith('stem__gpqa'):
        from . import gpqa
        from . import supergpqa
        if "no_box" in data_source:
            res = gpqa.compute_score(solution_str, ground_truth)
        else:
            res = supergpqa.compute_score(solution_str, ground_truth)
    elif data_source.startswith('stem__supergpqa'):
        from . import supergpqa
        res = supergpqa.compute_score(solution_str, ground_truth)
    elif data_source.startswith('stem_web') or data_source.startswith('stem__web'):
        # Added stem__web for guru dataset compatibility
        from . import stem_llm_judge
        res = stem_llm_judge.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info)
        # Remove debug print
        # print(extra_info,res,solution_str,ground_truth)
    elif data_source in ["ood__ifeval"]:
        from . import ifeval
        res = ifeval.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source in ["ood__livebench"]:
        from . import livebench
        res = livebench.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source in ["ood__ifbench"]:
        from . import ifbench
        res = ifbench.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # NOTE: above is added by Reasoning360
    elif data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    # numina sub-sources → prime_math (Reasoning360 rule-based)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    # aime* → math_dapo (Reasoning360 原版)
    elif data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
        res = _apply_eval_llm_fallback(res, solution_str, ground_truth, extra_info)
    # amc*, olympiad, minerva, gpqa_diamond → naive_dapo (原版无对应，用 naive_dapo)
    elif data_source.startswith("amc") or data_source in ["olympiad", "minerva", "gpqa_diamond"]:
        from . import naive_dapo
        res = naive_dapo.compute_score(solution_str, ground_truth, extra_info=extra_info or {})
        res = _apply_eval_llm_fallback(res, solution_str, ground_truth, extra_info)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion
            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, solution_str, ground_truth, continuous=True)
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code
            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa", "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle"]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, daytona_api_key=None):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, daytona_api_key)


__all__ = ["default_compute_score"]