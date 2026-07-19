import yaml
from typing import Dict
from lightagent import LightAgent
from tools.structure_tools import acquire_microstructures

def get_structure_generator_agent(config: Dict) -> LightAgent:
    llm_config = config['main_llm_config']
    prompt_path = config.get('agents', {}).get('structure_generator', {}).get('system_message_file', 'prompts/structure_generator.yaml')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
            system_message = prompt_config['system_message']
    except FileNotFoundError:
        system_message = "You are a structure generator agent."

    agent = LightAgent(
        name="StructureGenerator",
        instructions=system_message,
        role="Microstructure Generation Expert",
        model=llm_config['model'],
        api_key=llm_config['api_key'],
        base_url=llm_config['api_base'],
        temperature=llm_config.get('temperature'),
        max_tokens=llm_config.get('max_tokens'),
        tool_choice=llm_config.get('function_call'),
        tools=[acquire_microstructures]
    )
    return agent
