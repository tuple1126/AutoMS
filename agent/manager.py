import yaml
from typing import Dict
from lightagent import LightAgent

def get_manager_agent(config: Dict) -> LightAgent:
    llm_config = config['main_llm_config']
    # Support an ablation-experiment-specific prompt path.
    prompt_path = config.get('ablation_manager_prompt', 'prompts/manager.yaml')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
            system_message = prompt_config['system_message']
    except FileNotFoundError:
        system_message = "You are a conversation manager. Select the next speaker."

    agent = LightAgent(
        name="Manager",
        instructions=system_message,
        role="Conversation Workflow Manager",
        model=llm_config['model'],
        api_key=llm_config['api_key'],
        base_url=llm_config['api_base'],
        temperature=llm_config.get('temperature'),
        max_tokens=llm_config.get('max_tokens'),
        tool_choice=llm_config.get('function_call'),
    )
    return agent
