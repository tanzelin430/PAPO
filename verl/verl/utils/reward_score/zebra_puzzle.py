import re
import ast
import json
import threading
import concurrent.futures


class TimeoutException(Exception):
    pass


def extract_solution(solution_str):
    """Extract solution from <answer>...</answer> tags."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
        try:
            solution = ast.literal_eval(final_answer)
            return solution
        except (SyntaxError, ValueError):
            try:
                solution = json.loads(final_answer)
                return solution
            except json.JSONDecodeError:
                return None
        except Exception as e:
            print(f"Error extracting solution: {e}")
            return None
    else:
        return None


def compute_accuracy(answer, ground_truth):
    """
    Compare grid level accuracy of the final answer with the ground truth.
    """
    if not isinstance(answer, dict):
        return 0

    try:
        # num_objects
        num_rows = len(ground_truth["rows"])
        # num_attributes
        num_cols = len(ground_truth["header"])

        # total_correct_cells
        correct_cells = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if answer["rows"][i][j] == ground_truth["rows"][i][j]:
                    correct_cells += 1
        # accuracy
        accuracy = correct_cells / (num_rows * num_cols)
        return accuracy
    except (KeyError, IndexError, TypeError):
        return 0


def _compute_score_inner(solution_str, ground_truth):
    """Inner function for score computation."""
    # Parse ground_truth if it's a JSON string
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    predicted_arrangement = extract_solution(solution_str)

    if predicted_arrangement is None:
        return 0.0

    return compute_accuracy(predicted_arrangement, ground_truth)


def compute_score(solution_str, ground_truth, extra_info=None, method='strict', timeout: float = 10.0):
    """
    Compute score for zebra puzzle with thread-safe timeout.
    """
    try:
        # Use ThreadPoolExecutor for thread-safe timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_compute_score_inner, solution_str, ground_truth)
            try:
                score = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print("Computation timed out in zebra_puzzle")
                score = 0.0
    except Exception as e:
        print(f"Error in compute_score in zebra_puzzle: {e}")
        score = 0.0

    return {"score": score, "acc": score}
