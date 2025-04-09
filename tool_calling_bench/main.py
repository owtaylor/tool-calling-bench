# tool_calling_bench/main.py

import click
import yaml
import json
import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import List  # Import List
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Import backend functions
from .llm_backends.ollama_backend import invoke_ollama
from .llm_backends.anthropic_backend import invoke_anthropic
from .distractors import get_distractor_messages, DISTRACTOR_SETS
from .secrets_manager import Secrets

# --- Logging Setup ---
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# Suppress logging from third-party libraries like 'requests'
logging.getLogger("urllib3").setLevel(logging.WARNING)  # Suppress urllib3 logs
logging.getLogger("requests").setLevel(logging.WARNING) # Suppress requests logs

# --- Constants ---
NO_DISTRACTOR_KEY = "none"  # Define a constant for the no-distractor case

# --- Backend Mapping ---
BACKEND_MAP = {
    "ollama": invoke_ollama,
    "anthropic": invoke_anthropic,
    # Add other backends here
}


# --- Helper Functions ---
def load_json(filepath: Path) -> list | dict:
    """Loads JSON data from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {filepath}")
        raise


def load_yaml(filepath: Path) -> dict:
    """Loads YAML data from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        raise
    except yaml.YAMLError:
        logger.error(f"Error: Could not parse YAML from {filepath}")
        raise


# --- Core Evaluation Logic ---
def run_evaluation(
    config: dict,
    tools: list[dict],
    questions: list[dict],
    num_runs: int,
    temperature: float,
    distractors_to_run: List[str],
    secrets: Secrets,
    verbose: bool = False,
) -> dict:
    """Runs the evaluation for a given configuration across specified distractors."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    backend_name = config.get("backend")
    model_name = config.get("model")
    system_prompt = config.get("systemPrompt", "")
    backend_config_params = {
        k: v for k, v in config.items() if k not in ["backend", "model", "systemPrompt"]
    }

    if not backend_name or not model_name:
        logger.error(f"Configuration missing 'backend' or 'model'. Skipping.")
        return {}

    invoke_func = BACKEND_MAP.get(backend_name)
    if not invoke_func:
        logger.error(f"Backend '{backend_name}' not implemented. Skipping.")
        return {}

    logger.info(f"\n--- Running Evaluation: [Backend: {backend_name}, Model: {model_name}] ---")

    # Results structure: results[question_id][distractor_key]
    results = defaultdict(
        lambda: defaultdict(
            lambda: {"success": 0, "total": 0, "errors": 0, "no_call": 0, "wrong_call": 0}
        )
    )

    for distractor_key in distractors_to_run:
        logger.info(f"Running with Distractor Context: '{distractor_key}' ({num_runs} times)...")

        base_messages = []
        if distractor_key != NO_DISTRACTOR_KEY:
            distractor_messages = get_distractor_messages(distractor_key)
            if distractor_messages:
                base_messages = list(distractor_messages)  # Ensure it's a mutable copy
            else:
                # This case should be caught during argument parsing, but handle defensively
                logger.error(
                    f"Distractor key '{distractor_key}' not found in DISTRACTOR_SETS. Skipping this context."
                )
                continue

        for q_data in questions:
            question_id = q_data["id"]
            question_text = q_data["question"]
            expected_tool = q_data.get("expected_tool")  # Use .get() to handle missing keys

            logger.info(
                f"Testing Question ID: {question_id} ('{question_text}') | Expected Tool: {expected_tool or 'None (No Tool Expected)'}"
            )

            for i in range(num_runs):
                # Construct messages for this run
                # Start with base (distractor) messages, add the actual question
                messages = base_messages + [{"role": "user", "content": question_text}]

                try:
                    called_tool, response_content = invoke_func(
                        model=model_name,
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        secrets=secrets,
                        **backend_config_params,
                    )

                    # Store results under the specific distractor key
                    results[question_id][distractor_key]["total"] += 1
                    if response_content and response_content.startswith("Error:"):
                        results[question_id][distractor_key]["errors"] += 1
                        logger.warning(
                            f"    Run {i+1}/{num_runs}: Backend Error - {response_content}"
                        )
                    elif expected_tool is None:
                        # If no tool is expected, ensure no tool was called
                        if called_tool:
                            results[question_id][distractor_key]["wrong_call"] += 1
                            logger.warning(
                                f"    Run {i+1}/{num_runs}: Wrong tool (Called {called_tool}, Expected None)"
                            )
                        else:
                            results[question_id][distractor_key]["success"] += 1
                            logger.debug(f"    Run {i+1}/{num_runs}: Success (No tool called)")
                    elif called_tool == expected_tool:
                        results[question_id][distractor_key]["success"] += 1
                        logger.debug(f"    Run {i+1}/{num_runs}: Success (Called {called_tool})")
                    elif called_tool:
                        results[question_id][distractor_key]["wrong_call"] += 1
                        logger.warning(
                            f"    Run {i+1}/{num_runs}: Wrong tool (Called {called_tool}, Expected {expected_tool})"
                        )
                    else:
                        results[question_id][distractor_key]["no_call"] += 1
                        logger.info(
                            f"    Run {i+1}/{num_runs}: No tool called. Response: {response_content[:100] if response_content else 'None'}..."
                        )

                except Exception as e:
                    results[question_id][distractor_key]["errors"] += 1
                    results[question_id][distractor_key]["total"] += 1  # Count the attempt
                    logger.error(
                        f"    Run {i+1}/{num_runs}: Invocation Error - {e}", exc_info=verbose
                    )

    return results


# --- Result Presentation ---
loaded_tools = []
loaded_questions = []


def display_results(all_results: dict, questions_data: list[dict], console: Console):
    """Formats and prints the evaluation results using Rich."""
    console.rule("[bold cyan]Evaluation Results Summary", style="cyan")

    # Initialize a dictionary to track summary statistics for each distractor
    distractor_summary = defaultdict(lambda: {"success": 0, "total": 0})

    for config_name, results_data in all_results.items():
        if not results_data:
            continue

        console.print(f"\n[bold underline]Configuration: {config_name}[/bold underline]")

        table = Table(
            title=f"Results for {config_name}", show_header=True, header_style="bold magenta"
        )
        table.add_column("Question ID", style="dim", width=20)
        table.add_column("Expected Tool", width=20)
        table.add_column("Distractor Context", width=20)
        table.add_column("Success Rate", justify="right")
        table.add_column("Success", justify="right")
        table.add_column("No Call", justify="right")
        table.add_column("Wrong Call", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Total Runs", justify="right")

        question_ids_in_results = list(results_data.keys())

        for i, q_id in enumerate(question_ids_in_results):
            runs_by_distractor = results_data[q_id]
            q_info = next((q for q in questions_data if q["id"] == q_id), {})
            expected_tool = q_info.get("expected_tool", "N/A")

            # Get the distractor keys for which results exist for this question
            distractor_keys_for_q = list(runs_by_distractor.keys())
            # Optional: Sort keys for consistent display order (e.g., 'none' first)
            distractor_keys_for_q.sort(key=lambda x: (x != NO_DISTRACTOR_KEY, x))

            for j, distractor_key in enumerate(distractor_keys_for_q):
                res = runs_by_distractor[distractor_key]
                is_first_row_for_question = j == 0
                # Determine if this is the last row for this specific question
                is_last_row_for_question = j == len(distractor_keys_for_q) - 1
                # Add section breaks between questions, not between distractors for the same question
                end_section = is_last_row_for_question and (i != len(question_ids_in_results) - 1)

                if res and res.get("total", 0) > 0:
                    rate = (res.get("success", 0) / res["total"]) * 100 if res["total"] > 0 else 0
                    table.add_row(
                        q_id if is_first_row_for_question else "",
                        expected_tool if is_first_row_for_question else "",
                        distractor_key,  # Display the name of the distractor context
                        f"{rate:.1f}%",
                        str(res.get("success", 0)),
                        str(res.get("no_call", 0)),
                        str(res.get("wrong_call", 0)),
                        str(res.get("errors", 0)),
                        str(res["total"]),
                        end_section=end_section,
                    )

                    # Update distractor summary statistics
                    distractor_summary[distractor_key]["success"] += res.get("success", 0)
                    distractor_summary[distractor_key]["total"] += res.get("total", 0)
                else:
                    # Add a placeholder row if results are missing/empty for this distractor
                    table.add_row(
                        q_id if is_first_row_for_question else "",
                        expected_tool if is_first_row_for_question else "",
                        distractor_key,
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        end_section=end_section,
                    )

        console.print(table)

    # Generate a summary table for distractor success rates
    summary_table = Table(
        title="Summary Success Rates by Distractor", show_header=True, header_style="bold green"
    )
    summary_table.add_column("Distractor Context", width=20)
    summary_table.add_column("Success Rate", justify="right")
    summary_table.add_column("Total Success", justify="right")
    summary_table.add_column("Total Runs", justify="right")

    # Add rows for each distractor
    for distractor_key, stats in distractor_summary.items():
        total = stats["total"]
        success = stats["success"]
        success_rate = (success / total) * 100 if total > 0 else 0
        summary_table.add_row(
            distractor_key,
            f"{success_rate:.1f}%",
            str(success),
            str(total),
        )

    # Add a row for the overall summary
    overall_success = sum(stats["success"] for stats in distractor_summary.values())
    overall_total = sum(stats["total"] for stats in distractor_summary.values())
    overall_success_rate = (overall_success / overall_total) * 100 if overall_total > 0 else 0
    summary_table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{overall_success_rate:.1f}%[/bold]",
        f"[bold]{overall_success}[/bold]",
        f"[bold]{overall_total}[/bold]",
    )

    console.print(summary_table)


# --- Click CLI Definition ---
@click.command()
# Changed --config-dir to --config-file (as per your provided file)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the model configuration YAML file.",
)
@click.option(
    "--tools-file",
    "-t",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default="data/tools.json",
    show_default=True,
    help="Path to the JSON file defining tools.",
)
@click.option(
    "--questions-file",
    "-q",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default="data/questions.json",
    show_default=True,
    help="Path to the JSON file containing test questions.",
)
@click.option(
    "--runs",
    "-n",
    type=int,
    default=5,
    show_default=True,
    help="Number of times to repeat each question per distractor context.",
)
@click.option(
    "--temperature",
    "--temp",
    type=float,
    default=0.7,
    show_default=True,
    help="Sampling temperature for the LLM.",
)
@click.option(
    "--distractors",
    "distractors_option",
    type=str,
    default=None,
    help=f'Comma-separated list of distractor sets to run. Use "{NO_DISTRACTOR_KEY}" for no distractor. '
    f'Available sets: {", ".join(DISTRACTOR_SETS.keys())}. '
    f'Default: Run "{NO_DISTRACTOR_KEY}" and all available sets.',
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable detailed debug logging.")
def main(
    config_file: Path,
    tools_file: Path,
    questions_file: Path,
    runs: int,
    temperature: float,
    distractors_option: str | None,
    verbose: bool,
):
    """
    Evaluates LLM tool calling across different configurations, prompts, and distractor contexts.
    """
    console = Console()
    console.rule("[bold green]Tool Calling Evaluator", style="green")

    secrets = Secrets()

    global loaded_tools, loaded_questions
    try:
        loaded_tools = load_json(tools_file)
        loaded_questions = load_json(questions_file)
        logger.info(f"Loaded {len(loaded_tools)} tools from {tools_file}")
        logger.info(f"Loaded {len(loaded_questions)} questions from {questions_file}")
    except Exception:
        logger.error("Failed to load tools or questions file.", exc_info=verbose)
        if verbose:
            console.print_exception(show_locals=verbose)
        return

    # --- Determine which distractors to run ---
    available_distractor_keys = list(DISTRACTOR_SETS.keys())
    valid_keys = [NO_DISTRACTOR_KEY] + available_distractor_keys
    distractors_to_run: List[str] = []

    if distractors_option is None:
        # Default: run 'none' and all defined distractors
        distractors_to_run = [NO_DISTRACTOR_KEY] + available_distractor_keys
        logger.info(
            f"No --distractors specified, running default set: {', '.join(distractors_to_run)}"
        )
    else:
        requested_distractors = [d.strip() for d in distractors_option.split(",") if d.strip()]
        invalid_keys = [key for key in requested_distractors if key not in valid_keys]
        if invalid_keys:
            logger.error(f"Invalid distractor key(s) specified: {', '.join(invalid_keys)}")
            logger.error(f"Valid keys are: {', '.join(valid_keys)}")
            return  # Exit if invalid keys are provided
        distractors_to_run = requested_distractors
        logger.info(f"Running with specified distractors: {', '.join(distractors_to_run)}")
    # ---

    all_results = {}

    try:
        config = load_yaml(config_file)
        config_name = config_file.stem
        results = run_evaluation(
            config=config,
            tools=loaded_tools,
            questions=loaded_questions,
            num_runs=runs,
            temperature=temperature,
            distractors_to_run=distractors_to_run,  # Pass the list of keys
            secrets=secrets,
            verbose=verbose,
        )
        all_results[config_name] = results
    except Exception as e:
        logger.error(f"Failed to process config file {config_file}: {e}", exc_info=verbose)
        if verbose:
            console.print_exception(show_locals=True)

    display_results(all_results, loaded_questions, console)


if __name__ == "__main__":
    # Note: If running as a script directly, imports might need adjustment
    # depending on your project structure (e.g., `from tool_calling_bench.llm_backends...`)
    # If using `python -m tool_calling_bench.main`, the relative imports should work.
    main()
