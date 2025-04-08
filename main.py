import click
import yaml
import json
import os
import logging
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Import backend functions
from llm_backends.ollama_backend import invoke_ollama
from llm_backends.anthropic_backend import invoke_anthropic
from distractors import get_distractor_messages, get_random_distractor, DISTRACTOR_SETS

# --- Logging Setup ---
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

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
        with open(filepath, 'r', encoding='utf-8') as f:
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
        with open(filepath, 'r', encoding='utf-8') as f:
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
    use_distractors: bool,
    specific_distractor: str | None = None,
    verbose: bool = False
) -> dict:
    """Runs the evaluation for a given configuration."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


    backend_name = config.get("backend")
    model_name = config.get("model")
    system_prompt = config.get("systemPrompt", "")
    backend_config_params = {k: v for k, v in config.items() if k not in ["backend", "model", "systemPrompt"]}


    if not backend_name or not model_name:
        logger.error(f"Configuration missing 'backend' or 'model'. Skipping.")
        return {}

    invoke_func = BACKEND_MAP.get(backend_name)
    if not invoke_func:
        logger.error(f"Backend '{backend_name}' not implemented. Skipping.")
        return {}

    logger.info(f"\n--- Running Evaluation: [Backend: {backend_name}, Model: {model_name}] ---")

    results = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0, "errors": 0, "no_call": 0, "wrong_call": 0}))

    for q_data in questions:
        question_id = q_data["id"]
        question_text = q_data["question"]
        expected_tool = q_data["expected_tool"]

        logger.info(f"Testing Question ID: {question_id} ('{question_text}') | Expected Tool: {expected_tool}")

        # --- Direct Run ---
        logger.info(f"  Running Direct ({num_runs} times)...")
        for i in range(num_runs):
            messages = [{"role": "user", "content": question_text}]
            called_tool, response_content = invoke_func(
                model=model_name,
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                temperature=temperature,
                **backend_config_params
            )

            results[question_id]["direct"]["total"] += 1
            if response_content and response_content.startswith("Error:"):
                 results[question_id]["direct"]["errors"] += 1
                 logger.warning(f"    Run {i+1}/{num_runs}: API Error - {response_content}")
            elif called_tool == expected_tool:
                results[question_id]["direct"]["success"] += 1
                logger.debug(f"    Run {i+1}/{num_runs}: Success (Called {called_tool})")
            elif called_tool:
                results[question_id]["direct"]["wrong_call"] += 1
                logger.warning(f"    Run {i+1}/{num_runs}: Wrong tool (Called {called_tool}, Expected {expected_tool})")
            else:
                results[question_id]["direct"]["no_call"] += 1
                logger.info(f"    Run {i+1}/{num_runs}: No tool called. Response: {response_content[:100]}...")


        # --- Distracted Run ---
        if use_distractors:
            if specific_distractor:
                distractor_name = specific_distractor
                distractor_messages = get_distractor_messages(distractor_name)
                if not distractor_messages:
                     logger.error(f"Specified distractor '{distractor_name}' not found. Skipping distracted run.")
                     continue # Skip to next question if specific distractor invalid
            else:
                 # Pick a random one for each question if not specified
                 distractor_name, distractor_messages = get_random_distractor()

            logger.info(f"  Running with Distractor '{distractor_name}' ({num_runs} times)...")

            for i in range(num_runs):
                # Ensure distractor messages are copied
                messages = list(distractor_messages) + [{"role": "user", "content": question_text}]

                called_tool, response_content = invoke_func(
                    model=model_name,
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    **backend_config_params
                )

                results[question_id]["distracted"]["total"] += 1
                if response_content and response_content.startswith("Error:"):
                    results[question_id]["distracted"]["errors"] += 1
                    logger.warning(f"    Run {i+1}/{num_runs}: API Error - {response_content}")
                elif called_tool == expected_tool:
                    results[question_id]["distracted"]["success"] += 1
                    logger.debug(f"    Run {i+1}/{num_runs}: Success (Called {called_tool})")
                elif called_tool:
                    results[question_id]["distracted"]["wrong_call"] += 1
                    logger.warning(f"    Run {i+1}/{num_runs}: Wrong tool (Called {called_tool}, Expected {expected_tool})")
                else:
                    results[question_id]["distracted"]["no_call"] += 1
                    logger.info(f"    Run {i+1}/{num_runs}: No tool called. Response: {response_content[:100]}...")

    return results

# --- Result Presentation ---
def display_results(all_results: dict, console: Console):
    """Formats and prints the evaluation results using Rich."""
    console.rule("[bold cyan]Evaluation Results Summary", style="cyan")

    for config_name, results_data in all_results.items():
        if not results_data: continue # Skip if config failed

        console.print(f"\n[bold underline]Configuration: {config_name}[/bold underline]")

        table = Table(title=f"Results for {config_name}", show_header=True, header_style="bold magenta")
        table.add_column("Question ID", style="dim", width=20)
        table.add_column("Expected Tool", width=20)
        table.add_column("Run Type", width=15)
        table.add_column("Success Rate", justify="right")
        table.add_column("Success", justify="right")
        table.add_column("No Call", justify="right")
        table.add_column("Wrong Call", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Total Runs", justify="right")

        for q_id, runs in results_data.items():
            q_info = next((q for q in loaded_questions if q["id"] == q_id), {}) # Find question info
            expected_tool = q_info.get("expected_tool", "N/A")

            # Direct Run Row
            direct_res = runs.get("direct", {})
            if direct_res and direct_res.get("total", 0) > 0:
                 rate_direct = (direct_res.get("success", 0) / direct_res["total"]) * 100 if direct_res["total"] > 0 else 0
                 table.add_row(
                    q_id,
                    expected_tool,
                    "Direct",
                    f"{rate_direct:.1f}%",
                    str(direct_res.get("success", 0)),
                    str(direct_res.get("no_call", 0)),
                    str(direct_res.get("wrong_call", 0)),
                    str(direct_res.get("errors", 0)),
                    str(direct_res["total"]),
                )
            else:
                 table.add_row(q_id, expected_tool,"Direct","-", "-", "-", "-", "-", "-")


            # Distracted Run Row (if applicable)
            distracted_res = runs.get("distracted", {})
            if distracted_res and distracted_res.get("total", 0) > 0:
                rate_distracted = (distracted_res.get("success", 0) / distracted_res["total"]) * 100 if distracted_res["total"] > 0 else 0
                table.add_row(
                    "", # Don't repeat question id/tool
                    "",
                    "Distracted",
                    f"{rate_distracted:.1f}%",
                    str(distracted_res.get("success", 0)),
                    str(distracted_res.get("no_call", 0)),
                    str(distracted_res.get("wrong_call", 0)),
                    str(distracted_res.get("errors", 0)),
                    str(distracted_res["total"]),
                     end_section=(q_id != list(results_data.keys())[-1]) # Add separator line between questions
                )
            elif "direct" in runs: # Only add if direct was run
                table.add_row("", "", "Distracted", "-", "-", "-", "-", "-", "-", end_section=(q_id != list(results_data.keys())[-1]))


        console.print(table)

# --- Global variables to hold loaded data ---
loaded_tools = []
loaded_questions = []

# --- Click CLI Definition ---
@click.command()
@click.option('--config-dir', '-c', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), default='configs', help='Directory containing model configuration YAML files.')
@click.option('--tools-file', '-t', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), default='data/tools.json', help='Path to the JSON file defining tools.')
@click.option('--questions-file', '-q', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), default='data/questions.json', help='Path to the JSON file containing test questions.')
@click.option('--runs', '-n', type=int, default=3, help='Number of times to repeat each question.')
@click.option('--temperature', '--temp', type=float, default=0.1, help='Sampling temperature for the LLM.')
@click.option('--distractors/--no-distractors', default=True, help='Run tests with distractor messages.')
@click.option('--distractor-set', type=click.Choice(list(DISTRACTOR_SETS.keys())), default=None, help='Specify a single distractor set to use (default: random).')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable detailed debug logging.')
def main(config_dir: Path, tools_file: Path, questions_file: Path, runs: int, temperature: float, distractors: bool, distractor_set: str | None, verbose: bool):
    """
    Evaluates LLM tool calling across different configurations and prompts.
    """
    console = Console()
    console.rule("[bold green]Tool Calling Evaluator", style="green")

    global loaded_tools, loaded_questions
    try:
        loaded_tools = load_json(tools_file)
        loaded_questions = load_json(questions_file)
        logger.info(f"Loaded {len(loaded_tools)} tools from {tools_file}")
        logger.info(f"Loaded {len(loaded_questions)} questions from {questions_file}")
    except Exception:
        console.print_exception()
        return # Exit if data loading fails

    config_files = list(config_dir.glob('*.yaml')) + list(config_dir.glob('*.yml'))
    if not config_files:
        logger.error(f"No YAML configuration files found in {config_dir}")
        return

    logger.info(f"Found {len(config_files)} configuration files in {config_dir}")

    all_results = {}

    for config_file in config_files:
        try:
            config = load_yaml(config_file)
            config_name = config_file.stem # Use filename (without extension) as identifier
            results = run_evaluation(
                config=config,
                tools=loaded_tools,
                questions=loaded_questions,
                num_runs=runs,
                temperature=temperature,
                use_distractors=distractors,
                specific_distractor=distractor_set,
                verbose=verbose
            )
            all_results[config_name] = results
        except Exception as e:
             logger.error(f"Failed to process config file {config_file}: {e}")
             console.print_exception(show_locals=verbose)


    display_results(all_results, console)


if __name__ == '__main__':
    main()

