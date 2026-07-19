import yaml
from typing import Dict
from lightagent import LightAgent

def get_summary_reporter_agent(config: Dict) -> LightAgent:
    llm_config = config['main_llm_config']
    prompt_path = config.get('agents', {}).get('summary_reporter', {}).get('system_message_file', 'prompts/summary_reporter.yaml')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
            system_message = prompt_config['system_message']
    except FileNotFoundError:
        system_message = "You are a summary reporter agent."

    agent = LightAgent(
        name="SummaryReporter",
        instructions=system_message,
        role="Summary Reporting Specialist",
        model=llm_config['model'],
        api_key=llm_config['api_key'],
        base_url=llm_config['api_base'],
        temperature=llm_config.get('temperature'),
        max_tokens=llm_config.get('max_tokens'),
        tool_choice=llm_config.get('function_call'),
        tools=[] # No specific tools for now, relies on context
    )
    return agent
