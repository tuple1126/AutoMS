import os
import yaml
import asyncio
import numpy as np
import threading
from typing import Dict, Any, Annotated, Callable
from dataclasses import dataclass
from openai import AsyncOpenAI, OpenAI

# Try to import nano_graphrag, handle if missing
try:
    from nano_graphrag import GraphRAG, QueryParam
    from nano_graphrag.base import BaseKVStorage
    from nano_graphrag._utils import compute_args_hash
    HAS_GRAPHRAG = True
except ImportError:
    HAS_GRAPHRAG = False
    print("Warning: nano_graphrag not found. GraphRAG capabilities will be disabled.")

# Try to apply nest_asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

from lightagent import LightAgent

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Decorator to add attributes to the function"""
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    return final_decro

class RequirementParserManager:
    def __init__(self, config: Dict):
        self.config = config
        self.graphrag = None
        self.enable_graphrag = bool(
            config.get('agents', {}).get('requirement_parser', {}).get('enable_graphrag', False)
        )
        
        if HAS_GRAPHRAG and self.enable_graphrag:
            self._setup_graphrag()
            self._initialize_graphrag()

    def _setup_graphrag(self):
        graphrag_config = self.config.get('graphrag', {})
        self.api_key = graphrag_config.get('llm', {}).get('api_key')
        self.api_base = graphrag_config.get('llm', {}).get('api_base')
        self.model = graphrag_config.get('llm', {}).get('model', 'kimi-k2-0711-preview')
        
        self.embedding_api_key = graphrag_config.get('embeddings', {}).get('api_key')
        self.embedding_api_base = graphrag_config.get('embeddings', {}).get('api_base')
        self.embedding_model = graphrag_config.get('embeddings', {}).get('model', 'text-embedding-ada-002')
        
        input_config = graphrag_config.get('input', {})
        base_dir = input_config.get('base_dir', 'data/rag_doc')
        # Adjust path relative to workspace root if needed, assuming running from root
        self.working_dir = os.path.join(os.path.dirname(base_dir), 'graphrag_cache')
        self.docs_path = base_dir

    def _create_embedding_func(self):
        embedding_api_key = self.embedding_api_key
        embedding_api_base = self.embedding_api_base.rstrip('/') if self.embedding_api_base else None
        embedding_model = self.embedding_model

        fixed_dim = (
            self.config
            .get('graphrag', {})
            .get('embeddings', {})
            .get('embedding_dim', 4096)
        )

        @wrap_embedding_func_with_attrs(embedding_dim=fixed_dim, max_token_size=8192)
        async def generic_embedding(
            texts: list[str],
            _api_key=embedding_api_key,
            _api_base=embedding_api_base,
            _model=embedding_model,
            _dim=fixed_dim
        ) -> np.ndarray:
            client = OpenAI(api_key=_api_key, base_url=_api_base)
            try:
                emb_resp = client.embeddings.create(input=texts, model=_model)
                vectors = [d.embedding for d in emb_resp.data]
                if any(len(v) != _dim for v in vectors):
                    adjusted = []
                    for v in vectors:
                        if len(v) > _dim:
                            adjusted.append(v[:_dim])
                        elif len(v) < _dim:
                            adjusted.append(v + [0.0] * (_dim - len(v)))
                        else:
                            adjusted.append(v)
                    vectors = adjusted
                return np.array(vectors)
            except Exception as e:
                raise RuntimeError(f"Embedding request failed: {e}") from e

        return generic_embedding

    @staticmethod
    def build_chat_model_func(api_key: str, api_base: str, model: str) -> Callable:
        async def _model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
            if history_messages is None:
                history_messages = []
            openai_async_client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
            if hashing_kv is not None:
                try:
                    args_hash = compute_args_hash(model, messages)
                    cache_hit = await hashing_kv.get_by_id(args_hash)
                    if cache_hit is not None:
                        return cache_hit["return"]
                except Exception:
                    pass

            try:
                response = await openai_async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', 6000),
                    temperature=kwargs.get('temperature', 0.0),
                    timeout=120,
                    **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature', 'hashing_kv']}
                )
                result = response.choices[0].message.content
                if hashing_kv is not None:
                    try:
                        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})
                    except Exception:
                        pass
                return result
            except Exception as e:
                return f"LLM Error: {e}"

        return _model_func

    def _initialize_graphrag(self):
        self.embedding_func = self._create_embedding_func()
        model_func = self.build_chat_model_func(self.api_key, self.api_base, self.model)
        
        for enable_cache in (True, False):
            try:
                self.graphrag = GraphRAG(
                    working_dir=self.working_dir,
                    enable_llm_cache=enable_cache,
                    best_model_func=model_func,
                    cheap_model_func=model_func,
                    embedding_func=self.embedding_func,
                )
                # self._check_and_insert_documents() # Skipping doc insertion for now to speed up, assuming already done or can be done manually
                print(f"GraphRAG initialized (enable_llm_cache={enable_cache})")
                return
            except Exception as e:
                print(f"GraphRAG init failed with cache={enable_cache}: {e}")
                self.graphrag = None
        print("GraphRAG initialization failed.")

    def retrieve_material_knowledge_graphrag(
        self,
        query: Annotated[str, "Materials-science query used to retrieve domain knowledge for requirement analysis"],
        mode: Annotated[str, "Retrieval mode: 'local' or 'global'"] = "local",
    ) -> str:
        """Retrieve materials-science domain knowledge with GraphRAG."""
        if not self.graphrag:
            return "GraphRAG is not initialized. Analyze the request using built-in domain knowledge."
        
        try:
            def _do_query():
                if mode == "global":
                    return self.graphrag.query(query, param=QueryParam(mode="global"))
                return self.graphrag.query(query, param=QueryParam(mode="local"))

            try:
                result = _do_query()
            except RuntimeError as re:
                if "event loop is already running" in str(re):
                    holder = {}
                    th = threading.Thread(target=lambda: holder.setdefault("r", _do_query()))
                    th.start(); th.join()
                    result = holder.get("r", "")
                else:
                    raise
            
            if result and result.strip():
                return f"GraphRAG retrieval result:\n{result}\n\nNote: This information was retrieved from the knowledge graph. Combine it with domain knowledge for a thorough analysis."
            else:
                return f"No information relevant to '{query}' was found. Analyze the request using built-in domain knowledge."
        except Exception as e:
            return f"An error occurred during GraphRAG retrieval: {str(e)}. Analyze the request using built-in domain knowledge."

def get_requirement_parser_agent(config: Dict) -> LightAgent:
    manager = RequirementParserManager(config)
    llm_config = config['main_llm_config']
    
    # Load system message
    prompt_path = config.get('agents', {}).get('requirement_parser', {}).get('system_message_file', 'prompts/requirement_parser.yaml')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
            system_message = prompt_config['system_message']
    except FileNotFoundError:
        system_message = "You are a requirement parser agent."

    # Define the tool function with metadata
    def retrieve_material_knowledge_graphrag(
        query: str,
        mode: str = "local",
    ) -> str:
        """
        Retrieve materials-science domain knowledge with GraphRAG.
        
        Args:
            query: Materials-science query to retrieve.
            mode: Retrieval mode: 'local' or 'global'.
        """
        return manager.retrieve_material_knowledge_graphrag(query, mode)
    
    # Add tool info for LightAgent
    retrieve_material_knowledge_graphrag.tool_info = {
        "tool_name": "retrieve_material_knowledge_graphrag",
        "tool_title": "GraphRAG Retrieval",
        "tool_description": "Retrieve materials-science domain knowledge with GraphRAG to support requirement analysis.",
        "tool_params": [
            {"name": "query", "description": "Materials-science query to retrieve.", "type": "string", "required": True},
            {"name": "mode", "description": "Retrieval mode: 'local' or 'global'.", "type": "string", "required": False, "default": "local"}
        ]
    }

    agent = LightAgent(
        name="RequirementParser",
        instructions=system_message,
        role="Requirement Parsing Expert",
        model=llm_config['model'],
        api_key=llm_config['api_key'],
        base_url=llm_config['api_base'],
        temperature=llm_config.get('temperature'),
        max_tokens=llm_config.get('max_tokens'),
        tool_choice=llm_config.get('function_call'),
        tools=[retrieve_material_knowledge_graphrag]
    )
    return agent
