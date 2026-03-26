# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
import signal
from typing import Optional
import threading

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
import os
from .prime_math import math_normalize
from .prime_math.grader import math_equal
from math_verify import parse, verify

# Counter for sympy cache clearing (thread-safe)
_sympy_call_counter = 0
_sympy_call_counter_lock = threading.Lock()
_SYMPY_CACHE_CLEAR_INTERVAL = 100  # Clear cache every N calls


def _maybe_clear_sympy_cache():
    """Periodically clear sympy cache to prevent memory leaks.

    See: https://github.com/sympy/sympy/issues/26879
    """
    global _sympy_call_counter
    with _sympy_call_counter_lock:
        _sympy_call_counter += 1
        if _sympy_call_counter >= _SYMPY_CACHE_CLEAR_INTERVAL:
            _sympy_call_counter = 0
            try:
                sympy.core.cache.clear_cache()
            except Exception:
                pass


class timeout:

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.
    
    Args:
        final_answer: The answer string to normalize
        
    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
# 修复第148行的BAD_REGEXES
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def timeout(timeout_seconds: int = 8):
    """Thread-safe timeout decorator using concurrent.futures."""
    import concurrent.futures
    import threading

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check if we're in the main thread - if so, we can use signal-based timeout
            if threading.current_thread() is threading.main_thread():
                if os.name == "posix":
                    import signal

                    def handler(signum, frame):
                        raise TimeoutError("Operation timed out!")

                    old_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(timeout_seconds)

                    try:
                        return func(*args, **kwargs)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    return func(*args, **kwargs)
            else:
                # In worker thread - use ThreadPoolExecutor with timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError("Operation timed out!")

        return wrapper

    return decorator


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    # 修复第252行的正则表达式
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    # 修复第267行的正则表达式
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
            "degree",
            "cm",
            "centimeter",
            "meter",
            "mile",
            "second",
            "minute",
            "hour",
            "day",
            "week",
            "month",
            "year",
            "foot",
            "feet",
            "inch",
            "yard",
            "liter",
    ]:
        # 修复第301行的正则表达式
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    # 修复第302行的正则表达式
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


@timeout(timeout_seconds=2)
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (len(expr) > 2 and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS and
            all([ch not in expr[1:-1] for ch in TUPLE_CHARS])):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> tuple[bool, str]:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True, given_answer_normalized_mathd

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False, given_normalized

    if ground_truth_normalized == given_normalized:
        return True, given_normalized

    if len(given_normalized) == 0:
        return False, given_normalized

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (ground_truth_normalized[0] != given_normalized[0] or
                                        ground_truth_normalized[-1] != given_normalized[-1]):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct, given_normalized

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1:right_brace_idx].strip()


def match_answer(response):
    is_matched = False
    response = response.split("</think>")[-1]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed
    
    return is_matched, response

import math

def _parse_scientific_notation(s: str) -> float:
    """
    Parse various forms of scientific notation to a float.
    Handles: "4.5e33", "4.5E33", "4.5 * 10^33", "4.5 \\times 10^{33}", etc.
    Returns None if parsing fails.
    """
    if s is None:
        return None

    # Clean up the string
    s = s.strip().replace(" ", "").replace("$", "").replace("{", "").replace("}", "")
    s = s.replace("\\mathrm", "").replace("\\text", "")
    # Remove trailing units like "erg/s", "cm", etc.
    s = re.sub(r'[a-zA-Z/]+$', '', s)
    s = s.strip()

    if not s:
        return None

    # Standard scientific notation: 4.5e33, 4.5E33
    sci_match = re.match(r'^([+-]?\d+\.?\d*)[eE]([+-]?\d+)$', s)
    if sci_match:
        try:
            return float(s)
        except:
            pass

    # LaTeX style: 4.5\times10^33, 4.5*10^33, 4.5×10^33
    latex_patterns = [
        r'^([+-]?\d+\.?\d*)\\times10\^([+-]?\d+)$',
        r'^([+-]?\d+\.?\d*)\*10\^([+-]?\d+)$',
        r'^([+-]?\d+\.?\d*)×10\^([+-]?\d+)$',
        r'^([+-]?\d+\.?\d*)\\cdot10\^([+-]?\d+)$',
    ]

    for pattern in latex_patterns:
        match = re.match(pattern, s)
        if match:
            try:
                mantissa = float(match.group(1))
                exponent = int(match.group(2))
                return mantissa * (10 ** exponent)
            except:
                pass

    return None


def _check_scientific_notation_equivalence(answer: str, ground_truth: str, rel_tol: float = 1e-2) -> bool:
    """
    Check if two answers are equivalent when interpreted as scientific notation.
    Uses 1% relative tolerance to handle rounding differences.
    """
    ans_val = _parse_scientific_notation(answer)
    gt_val = _parse_scientific_notation(ground_truth)

    if ans_val is None or gt_val is None:
        return False

    # Handle zero case
    if gt_val == 0:
        return ans_val == 0

    # Relative tolerance comparison for large numbers
    return abs(ans_val - gt_val) / abs(gt_val) < rel_tol


@timeout(timeout_seconds=10)
def compute_score(solution_str: str,
                  ground_truth: str,
                  extra_info: dict) -> float:
    """Compute the reward score for a solution. This draws heavily from the LLM-as-judge and PRIME reward functions

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        extra_info: dict with additional info for the score computation

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Periodically clear sympy cache to prevent memory leaks
    _maybe_clear_sympy_cache()

    try:
        # First assert intended generation and gt type
        model_output = str(solution_str)
        ground_truth = str(ground_truth)

        # Extract answer from generated output
        is_matched, extracted_model_output = match_answer(model_output)
        
        # TWK NOTE: WE REMOVED THE RESPONSE TRUNCATION FROM math_dapo.compute_score

        # Verify the solution, first check simple comparisons.
        correct, pred = grade_answer(extracted_model_output, ground_truth)

        if not correct: 
            try:
                if "\\pi" in extracted_model_output or "\\pi" in ground_truth:
                    equivs = []
                    for pi in [math.pi, 3.14]:
                        equivs.append(math_equal(extracted_model_output, ground_truth, tiemout=True, pi=pi))
                        correct = any(equivs)
                else:
                    correct = math_equal(extracted_model_output, ground_truth, timeout=True)
            except:
                correct = False


        # Reward logic:
        # - Answer correct: reward = 1.0
        # - Format correct but answer wrong: reward = -0.5
        # - Format wrong and answer wrong: reward = -1.0
        
        # if correct:
        #     reward = 1.0
        # elif is_matched:  # Format is correct (has \\boxed{}) but answer is wrong
        #     reward = -0.5
        # else:  # Both format and answer are wrong
        #     reward = -1.0
        # If our parser says it's wrong, try Math-Verify as fallback
        if not correct:
            try:
                gold_parsed = parse(str(ground_truth))
                answer_parsed = parse(str(solution_str))
                correct = verify(gold_parsed, answer_parsed)
            except:
                pass

        # Scientific notation equivalence check (e.g., "4.5e33" vs "4.5 \times 10^{33}")
        if not correct:
            try:
                correct = _check_scientific_notation_equivalence(extracted_model_output, ground_truth)
            except:
                pass

        reward = 1.0 if correct else 0.
        acc = correct

        return {
            "score": reward,
            "acc": acc,
        }
    except TimeoutError:
        error_msg = (
            f"Timeout after 10 seconds while computing score!\n"
            f"Answer being validated: {solution_str[:500]}{'...' if len(solution_str) > 500 else ''}\n"
            f"Ground truth: {ground_truth}"
        )
        return {
            "score": 0.0,
            "acc": 0.0,
        }