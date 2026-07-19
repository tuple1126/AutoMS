import yaml
from typing import Dict
from lightagent import LightAgent
from tools.simulation_tools_wrapper import (
    get_obj_files,
    run_heat_analysis_wrapper,
    run_stiffness_analysis_wrapper,
    batch_stiffness_analysis_wrapper,
    run_plasticity_simulation_wrapper,
    get_heat_analysis_files_wrapper,
    run_electrical_analysis_wrapper,
    get_electrical_analysis_files_wrapper,
)

def get_simulator_agent(config: Dict) -> LightAgent:
    llm_config = config['main_llm_config']
    prompt_path = config.get('agents', {}).get('simulator', {}).get('system_message_file', 'prompts/simulator.yaml')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
            system_message = prompt_config['system_message']
    except FileNotFoundError:
        system_message = "You are a simulation agent."

    agent = LightAgent(
        name="Simulator",
        instructions=system_message,
        role="Simulation Engineer",
        model=llm_config['model'],
        api_key=llm_config['api_key'],
        base_url=llm_config['api_base'],
        temperature=llm_config.get('temperature'),
        max_tokens=llm_config.get('max_tokens'),
        tool_choice=llm_config.get('function_call'),
        tools=[
            get_obj_files,
            run_heat_analysis_wrapper,
            run_stiffness_analysis_wrapper,
            batch_stiffness_analysis_wrapper,
            run_plasticity_simulation_wrapper,
            get_heat_analysis_files_wrapper,
            run_electrical_analysis_wrapper,
            get_electrical_analysis_files_wrapper,
        ]
    )
    return agent
