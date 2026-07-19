import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightagent import LightAgent
from agent.requirement_parser import get_requirement_parser_agent
from agent.structure_generator import get_structure_generator_agent
from agent.simulator import get_simulator_agent
from agent.summary_reporter import get_summary_reporter_agent
from agent.manager import get_manager_agent
from tools.case_library import capture_terminal_output
from tools.session_logger import SessionLogger
from tools.tree_tracker import TreePlanTracker
from tools.prompt_logger import get_prompt_logger
from tools.workflow_manager import select_next_speaker
from tools.simulation_tools_wrapper import set_current_agent
from tools.tree_tracker import reset_tool_call_tracker, reset_property_table, get_property_table
from tools.saes_guidance import get_saes_guidance, reset_saes_guidance, is_multi_objective_scenario
# SAES: Simulation-aware evolutionary-search integration
from tools.saes_integration import (
    get_saes_integrator,
    reset_saes_integrator,
    is_saes_enabled,
)


ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_environment_variables(value: Any, location: str = "config") -> Any:
    """Resolve ${ENV_VAR} placeholders recursively without exposing values."""
    if isinstance(value, dict):
        return {
            key: resolve_environment_variables(item, f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_environment_variables(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    resolved = value
    for env_var in ENV_VAR_PATTERN.findall(value):
        env_value = os.environ.get(env_var)
        if not env_value:
            raise ValueError(
                f"Missing required environment variable '{env_var}' for {location}. "
                "Copy .env.example to .env and set the value."
            )
        resolved = resolved.replace(f"${{{env_var}}}", env_value)
    return resolved


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    load_dotenv(path.parent / ".env", override=False)
    with path.open('r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return resolve_environment_variables(config)


def validate_config_paths(config: Dict[str, Any]) -> None:
    prompt_paths = [config.get('ablation_manager_prompt', 'prompts/manager.yaml')]
    for agent_config in config.get('agents', {}).values():
        if isinstance(agent_config, dict) and agent_config.get('system_message_file'):
            prompt_paths.append(agent_config['system_message_file'])

    missing = [str(path) for path in prompt_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Required prompt files were not found: {', '.join(missing)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the AutoMS agent workflow with a configured request."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to a YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--query",
        help="User request. When omitted, read standard input or prompt interactively.",
    )
    return parser.parse_args(argv)


def read_query(query: str = None) -> str:
    try:
        if query is None:
            query = sys.stdin.read() if not sys.stdin.isatty() else input("Request: ")
    except EOFError as exc:
        raise ValueError("A request was not provided on standard input.") from exc

    query = query.strip()
    if not query:
        raise ValueError("A non-empty request is required. Pass --query or provide text on standard input.")
    return query


def configure_default_method():
    """Enable the complete AutoMS workflow rather than an ablation variant."""
    full_flags = {
        "CHATMS_ABLATION_SAES": "0",
        "CHATMS_ABLATION_SAES_GUIDANCE": "0",
        "CHATMS_ABLATION_DUAL_SOURCE": "0",
        "CHATMS_ABLATION_HISTORY_MUTATION": "0",
        "CHATMS_ABLATION_ADAPTIVE_WEIGHT": "0",
    }
    for key, value in full_flags.items():
        os.environ[key] = value


def _try_initialize_saes(response: str, saes_integrator):
    """Parse a RequirementParser JSON response and initialize SAES.

    Args:
        response: RequirementParser output.
        saes_integrator: SAES integrator instance.
    """
    try:
        # Extract JSON, which may be wrapped in a Markdown code block.
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fall back to locating a JSON object directly.
            json_match = re.search(r'\{[\s\S]*"status"[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                print("[SAES] Could not extract JSON from RequirementParser output.")
                return
        
        parsed_req = json.loads(json_str)
        
        base_material = parsed_req.get("recommended_base_material", {})
        material_params = parsed_req.get("material_parameters", {})
        specific_target = parsed_req.get("specific_target", {})
        
        # Only Type B requests require simulation.
        requires_simulation = parsed_req.get("requires_simulation", False)
        if not requires_simulation:
            print("[SAES] Type A request does not require simulation; SAES is inactive.")
            return
        
        # Build target properties for SAES initialization.
        target_properties = {}
        
        # Extract target values for each property.
        param_mapping = {
            "elastic_modulus_range": "elastic_modulus",
            "shear_modulus_range": "shear_modulus", 
            "poisson_ratio_range": "poisson_ratio",  # Include Poisson's ratio.
            "thermal_conductivity_range": "thermal_conductivity",
            "electrical_conductivity_range": "electrical_conductivity",
            "volume_fraction_range": "volume_fraction",
        }

        unsupported_protocol_targets = {
            "yield_strength_range": "proof/yield strength requires a separately specified post-processing protocol",
            "plastic_strain_range": "generic plastic strain is not equivalent to the solver's protocol-specific RVE EQPS",
        }
        for range_key, reason in unsupported_protocol_targets.items():
            if range_key in specific_target:
                print(f"[SAES] Skipping {range_key}: {reason}.")
        
        for range_key, prop_name in param_mapping.items():
            if range_key in specific_target:
                range_val = specific_target[range_key]
                if isinstance(range_val, list) and len(range_val) == 2:
                    # Use the midpoint as the target value.
                    target_properties[prop_name] = {
                        "min": range_val[0],
                        "max": range_val[1],
                        "target": (range_val[0] + range_val[1]) / 2
                    }
        
        if target_properties:
            # Initialize SAES.
            saes_integrator.initialize({
                "target_properties": target_properties,
                "material_parameters": material_params,
                "base_material": base_material
            })
            print(f"[SAES] Initialized from RequirementParser output with {len(target_properties)} targets.")
            
        else:
            print("[SAES] No valid target properties were extracted.")
            
    except json.JSONDecodeError as e:
        print(f"[SAES] JSON parsing failed: {e}")
    except Exception as e:
        print(f"[SAES] Initialization failed: {e}")

def main(argv=None):
    args = parse_args(argv)
    configure_default_method()
    config_path = Path(args.config).expanduser().resolve()
    os.chdir(PROJECT_ROOT)

    try:
        config = load_config(str(config_path))
        validate_config_paths(config)
        user_input = read_query(args.query)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    
    # Initialize Agents
    requirement_parser = get_requirement_parser_agent(config)
    structure_generator = get_structure_generator_agent(config)
    simulator = get_simulator_agent(config)
    summary_reporter = get_summary_reporter_agent(config)
    manager_agent = get_manager_agent(config)
    
    agents = {
        "RequirementParser": requirement_parser,
        "StructureGenerator": structure_generator,
        "Simulator": simulator,
        "SummaryReporter": summary_reporter,
        "Manager": manager_agent
    }
    
    session_logger = SessionLogger()
    
    # Initialize Tree Tracker
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracker = TreePlanTracker(session_id)
    tracker.add_node("requirement", "User Requirement", {"text": user_input})
    
    # Reset tool call tracker and property table for new session
    reset_tool_call_tracker()
    reset_property_table()
    reset_saes_guidance()  # Reset multi-objective optimization guidance.
    reset_saes_integrator()  # Reset the SAES optimizer.
    
    # Reset the simulated-file tracker to prevent duplicate simulations.
    from tools.simulation_tools_wrapper import reset_simulated_files_tracker
    reset_simulated_files_tracker()

    workflow_failed = False
    with capture_terminal_output() as buffer:
        print("\n===========================================================================")
        print("\n============================System Initialized.===========================")
        print("\n===========================================================================")
        print(f"User Requirement: {user_input}")
        print("\n[Method] AutoMS: SAES + SAES Guidance + Dual-Source + HDMO + AWA")
        print("\n[SAES] Simulation-aware evolutionary search optimizer is ready")
        print("   Optimization targets will be initialized automatically after requirement parsing is complete...")
        
        history = [{"role": "user", "content": user_input}]
        current_agent_name = "RequirementParser"
        current_instruction = "Analyze the user requirement." # Initial instruction
        
        # Multi-objective optimization guidance instance.
        saes_guidance = get_saes_guidance()
        
        # SAES optimizer integration.
        saes_integrator = get_saes_integrator()
        
        # Retrieve the tool-call tracker.
        from tools.tree_tracker import get_tool_call_tracker
        tool_tracker = get_tool_call_tracker()
        
        while True:
            print(f"\n---------------------------------------")
            print(f"\n--- {current_agent_name} is working ---")
            print(f"\n---------------------------------------")
            
            current_agent = agents[current_agent_name]
            
            # Set current agent name for tool call tracking
            set_current_agent(current_agent_name)
            
            # Mark a checkpoint to detect whether this agent calls new tools in this turn.
            tool_tracker.mark_check_point()
            
            # Prepare input for the agent. 
            # LightAgent.run takes a query. We pass the last relevant information or the whole context?
            # LightAgent has its own history management if we pass 'history' param.
            # But here we are switching agents. We should pass the conversation so far.
            
            # Construct the query/context for the agent
            # For the first turn, it's the user input.
            # For subsequent turns, it's the output of the previous agent.
            
            last_content = history[-1]['content']
            
            # Inject Context Summary from Tree Tracker
            context_summary = tracker.get_context_summary()
            
            # Use the Manager's instruction if available, otherwise fallback to last content
            task_instruction = current_instruction if current_instruction else last_content
            
            # Inject SAES guidance.
            saes_context = ""
            if is_saes_enabled():
                saes_context = saes_integrator.get_context_injection()
                # Print a concise SAES status summary.
                print(f"\n[SAES] Optimization status:")
                print(f"   Evolution generation: {saes_integrator.saes.population_manager.current_generation}")
                pop_stats = saes_integrator.saes.population_manager.get_population_stats()
                print(f"   Population: AI={pop_stats['ai_subpop_size']} + DB={pop_stats['db_subpop_size']}")
                print(f"   Pareto frontier: {len(saes_integrator.saes.pareto_front)} non-dominated solutions")
                if saes_context:
                    print(f"   OK: Context guidance injected into {current_agent_name}")
            else:
                print(f"\n[SAES] Status: inactive (waiting for target initialization)")
            
            augmented_query = f"{context_summary}\n{saes_context}\n\n[Manager Instruction]:\n{task_instruction}\n\n[Previous Output]:\n{last_content}"

            # Record the terminal-output position before execution.
            terminal_before = buffer.get("get_current", lambda: "")()
            terminal_before_len = len(terminal_before)

            # Run the agent
            # We pass the global history to the agent so it sees the context
            # Note: LightAgent.run expects history as list of dicts
            
            try:
                response = current_agent.run(augmented_query, history=history[:-1]) # Pass history excluding the current input which is passed as query
                # Wait, run(query, history=...) adds query to history.
                # So if we pass history[:-1], and query=last_content, it reconstructs the full history.
                
                # Capture terminal output produced during execution (incremental).
                terminal_after = buffer.get("get_current", lambda: "")()
                terminal_during_execution = terminal_after[terminal_before_len:]
                
                # Log full execution details, including the prompt visible to the agent.
                get_prompt_logger().log_full_agent_execution(
                    agent_name=current_agent_name,
                    agent_role=getattr(current_agent, 'role', 'Unknown'),
                    agent_instructions=getattr(current_agent, 'instructions', 'Unknown'),
                    context_summary=context_summary,
                    manager_instruction=task_instruction,
                    previous_output=last_content,
                    augmented_query=augmented_query,
                    history=history[:-1],
                    response=response,
                    terminal_output=terminal_during_execution
                )
                
            except Exception as e:
                print(f"Error running agent {current_agent_name}: {e}")
                tracker.add_node("error", f"Error in {current_agent_name}", {"error": str(e)})
                workflow_failed = True
                break
                
            print(f"\n{current_agent_name}: {response}")
            
            # SAES: Automatically initialize after RequirementParser completes.
            if current_agent_name == "RequirementParser":
                _try_initialize_saes(response, saes_integrator)
            
            # Update global history
            history.append({"role": "assistant", "content": f"[{current_agent_name}]: {response}"})
            
            # Add to tracker
            tracker.add_node("agent_response", current_agent_name, {"response": response})

            # Select next speaker
            try:
                next_agent_name, next_instruction = select_next_speaker(history, agents, manager_agent)
            except Exception as e:
                print(f"Error selecting the next agent: {e}")
                tracker.add_node("error", "Workflow manager", {"error": str(e)})
                workflow_failed = True
                break
            print(f"---------------------------------------")
            print(f"Next speaker: {next_agent_name}")
            print(f"Instruction: {next_instruction}")
            print(f"---------------------------------------")
            
            # Detect an iteration boundary: after Simulator output, a StructureGenerator next step starts a new iteration.
            # Hide all .obj files in workshop as .objk to prevent duplicate simulations.
            if current_agent_name == "Simulator" and next_agent_name == "StructureGenerator":
                from tools.simulation_tools_wrapper import hide_simulated_files_for_iteration
                print(f"\n[Iteration Boundary] New iteration detected: {current_agent_name} -> {next_agent_name}")
                hidden_count = hide_simulated_files_for_iteration()
                if hidden_count > 0:
                    print(f"   Hidden {hidden_count} simulated .obj files to prevent duplicate simulations")
                # Increment the design-iteration count.
                tracker.increment_design_iteration()
                print(f"   Design iteration: {tracker.design_iteration_count}/{tracker.max_design_iterations}")
                # SAES: Record the iteration and advance the generation.
                if is_saes_enabled():
                    saes_integrator.record_iteration()
                    print(f"   SAES evolution generation: {saes_integrator.saes.population_manager.current_generation}")
            
            # Increment the total iteration count.
            tracker.increment_iteration()
            
            # Check whether the iteration limit has been reached.
            iter_status = tracker.get_iteration_status()
            if iter_status['should_terminate']:
                print(f"\nWARNING: Iteration limit reached; terminating workflow.")
                print(f"   Agent Turns: {iter_status['current_iteration']}/{iter_status['max_iterations']}")
                print(f"   Design Iterations: {iter_status['design_iteration']}/{iter_status['max_design_iterations']}")
                # Force a switch to SummaryReporter.
                if next_agent_name != "SummaryReporter" and next_agent_name != "TERMINATE":
                    next_agent_name = "SummaryReporter"
                    next_instruction = "The iteration limit has been reached. Generate the final summary report immediately. Summarize the best results obtained so far, even if the objectives have not been fully met."
                    print(f"   Forced switch to SummaryReporter for report generation")
            
            if next_agent_name == "TERMINATE":
                print("Session terminated.")
                break
                
            if next_agent_name not in agents:
                print(f"Unknown agent selected: {next_agent_name}. Terminating.")
                workflow_failed = True
                break
                
            current_agent_name = next_agent_name
            current_instruction = next_instruction

    # Restore all hidden microstructure files after the conversation ends.
    from tools.simulation_tools_wrapper import restore_all_hidden_files_globally
    restore_all_hidden_files_globally()

    # Save the tree tracker
    tree_path = tracker.save_tree()
    print(f"\nTree Plan saved to: {tree_path}")
    
    # Export microstructure property table
    prop_table = get_property_table()
    if prop_table.properties:
        csv_path = f"data/case_library_tree/{session_id}_property_table.csv"
        json_path = f"data/case_library_tree/{session_id}_property_table.json"
        prop_table.export_to_csv(csv_path)
        prop_table.export_to_json(json_path)
        print(f"Property Table exported to: {csv_path}")
        print(f"Property Table (JSON) exported to: {json_path}")
        print("\n=== Microstructure Property Summary ===")
        print(prop_table.get_property_summary())

    session_logger.handle_session_end(user_input, buffer["text"])
    return 1 if workflow_failed else 0

if __name__ == "__main__":
    raise SystemExit(main())

