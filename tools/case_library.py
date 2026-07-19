"""
Case library management utilities.

Manage a microstructure-design case library, including case creation,
storage, retrieval, and statistical analysis.
"""

import ast
import asyncio
import sys
import io
import contextlib
import json
import logging
import os
import yaml
import re
from typing import Dict, List, Any, Optional
from datetime import datetime


# Capture terminal output and mirror stdout/stderr to both the console and an
# in-memory buffer.
class _TeeIO(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                # Do not let one failed stream write break all output.
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        for s in self._streams:
            try:
                if s.isatty():
                    return True
            except Exception:
                continue
        return False


@contextlib.contextmanager
def capture_terminal_output():
    """
    Capture the current process's stdout/stderr and mirror it to the console.
    After the context exits, buf["text"] contains the complete output.
    buf["get_current"]() also returns accumulated output while the context runs.
    """
    orig_out, orig_err = sys.stdout, sys.stderr
    buf_out, buf_err = io.StringIO(), io.StringIO()
    # Replace standard output/error streams with tees.
    sys_stdout_tee = _TeeIO(orig_out, buf_out)
    sys_stderr_tee = _TeeIO(orig_err, buf_err)
    sys.stdout = sys_stdout_tee
    sys.stderr = sys_stderr_tee

    # Mirror streams used by logging.StreamHandler instances as well.
    logger_handlers = []
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        try:
            if isinstance(h, logging.StreamHandler):
                original_stream = h.stream
                # Choose the corresponding tee to preserve the original behavior.
                if original_stream is sys.stdout:
                    h.stream = sys_stdout_tee
                elif original_stream is sys.stderr:
                    h.stream = sys_stderr_tee
                else:
                    # Mirror any other stream to buf_out so it is captured too.
                    h.stream = _TeeIO(original_stream, buf_out)
                logger_handlers.append((h, original_stream))
        except Exception:
            continue
    try:
        # Expose the output accumulated so far.
        def get_current_output():
            try:
                return buf_out.getvalue() + buf_err.getvalue()
            except Exception:
                return ""
        
        buffer = {
            "text": "",
            "get_current": get_current_output,  # Available while the context runs.
            "_buf_out": buf_out,  # Direct reference for advanced callers.
            "_buf_err": buf_err
        }
        yield buffer
    finally:
        # Combine output and restore original streams.
        try:
            buffer["text"] = buf_out.getvalue() + buf_err.getvalue()
        except Exception:
            buffer["text"] = ""
        # Restore original logger-handler streams.
        for h, s in logger_handlers:
            try:
                h.stream = s
            except Exception:
                pass
        sys.stdout = orig_out
        sys.stderr = orig_err


class CaseLibraryManager:
    """Manage the unified case library."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the case-library manager."""
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir = base_dir
        self.cases_dir = os.path.join(base_dir, "data", "experience_db")
        self.unified_cases_path = os.path.join(self.cases_dir, "unified_cases.json")
        
        # Ensure the directory exists.
        os.makedirs(self.cases_dir, exist_ok=True)
        
        # Configure logging.
        self.logger = logging.getLogger(__name__)
    
    def parse_terminal_output(self, terminal_output: str) -> Dict[str, Any]:
        """Parse terminal output and extract relevant information."""
        parsed_data = {
            "agent_interactions": [],
            "tool_calls": [],
            "simulation_results": [],
            "performance_metrics": {},
            "flow_analysis": {
                "total_agents": 0,
                "agent_sequence": [],
                "total_tool_calls": 0,
                "total_rounds": 0,
                "agent_reply_count": {}
            }
        }
        
        # Precise regular-expression patterns.
        agent_pattern = r'\x1b\[33m([^)]+)\x1b\[0m \(to chat_manager\):'
        next_speaker_pattern = r'Next speaker: ([^\n\x1b]+)'
        tool_call_pattern = r'\*{5} Suggested tool call \(([^)]+)\): ([^*]+) \*{5}'
        function_execution_pattern = r'>>>>>>>> EXECUTING FUNCTION ([^.]+)\.\.\..*?>>>>>>>> EXECUTED FUNCTION.*?Input arguments: ({[^}]*}.*?}).*?Output:\s*(.*?)(?=\x1b|\*{5}|>>>>>>>>>|$)'
        
        # Parse agent reply counts and sequence.
        agent_matches = re.findall(agent_pattern, terminal_output)
        next_speaker_matches = re.findall(next_speaker_pattern, terminal_output)
        
        # Count agent replies.
        agent_reply_count = {}
        agent_sequence = []
        
        for match in agent_matches:
            agent_name = match.strip()
            if agent_name not in agent_reply_count:
                agent_reply_count[agent_name] = 0
                agent_sequence.append(agent_name)
            agent_reply_count[agent_name] += 1
        
        # Count conversation rounds from "Next speaker" messages.
        total_rounds = len(next_speaker_matches)
        
        parsed_data["flow_analysis"]["agent_sequence"] = agent_sequence
        parsed_data["flow_analysis"]["total_agents"] = len(agent_sequence)
        parsed_data["flow_analysis"]["agent_reply_count"] = agent_reply_count
        parsed_data["flow_analysis"]["total_rounds"] = total_rounds
        
        # Parse tool-call details.
        tool_matches = re.findall(tool_call_pattern, terminal_output, re.DOTALL)
        for tool_match in tool_matches:
            tool_call_info = {
                "call_id": tool_match[0].strip(),
                "tool_name": tool_match[1].strip(),
                "timestamp": datetime.now().isoformat(),
                "status": "suggested"
            }
            parsed_data["tool_calls"].append(tool_call_info)
        
        # Parse function-execution details.
        func_matches = re.findall(function_execution_pattern, terminal_output, re.DOTALL | re.MULTILINE)
        for func_match in func_matches:
            function_name = func_match[0].strip()
            try:
                arguments_raw = func_match[1].strip()
                # Parse arguments safely.
                arguments = {}
                try:
                    arguments = ast.literal_eval(arguments_raw)
                except (ValueError, SyntaxError):
                    try:
                        arguments = json.loads(arguments_raw)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"raw": arguments_raw}
            except:
                arguments = {}
            
            output_content = func_match[2].strip() if len(func_match) > 2 else ""
            
            tool_execution = {
                "function_name": function_name,
                "arguments": arguments,
                "output": output_content[:500] + "..." if len(output_content) > 500 else output_content,
                "status": "executed"
            }
            parsed_data["tool_calls"].append(tool_execution)
        
        parsed_data["flow_analysis"]["total_tool_calls"] = len(parsed_data["tool_calls"])
        
        # Parse heat-conduction simulation results.
        heat_conductivity_pattern = r'Effective thermal conductivity:\s*Kxx = ([0-9.]+)\s*Kyy = ([0-9.]+)\s*Kzz = ([0-9.]+)'
        heat_matches = re.findall(heat_conductivity_pattern, terminal_output)
        for heat_match in heat_matches:
            heat_data = {
                "kxx": float(heat_match[0]),
                "kyy": float(heat_match[1]),
                "kzz": float(heat_match[2]),
                "avg_conductivity": (float(heat_match[0]) + float(heat_match[1]) + float(heat_match[2])) / 3
            }
            parsed_data["simulation_results"].append({
                "type": "heat_conductivity",
                "data": heat_data
            })
        
        # Parse specific-energy-absorption data.
        energy_pattern = r'Specific energy absorption:\s*([0-9.]+)'
        energy_matches = re.findall(energy_pattern, terminal_output)
        if energy_matches:
            parsed_data["performance_metrics"]["specific_energy_absorption"] = [float(e) for e in energy_matches]
        
        # Parse microstructure names and performance data.
        lattice_performance_pattern = r'(lattice_\d+).*?Specific energy absorption:\s*([0-9.]+)'
        lattice_matches = re.findall(lattice_performance_pattern, terminal_output)
        
        analyzed_structures = []
        if lattice_matches:
            for lattice_match in lattice_matches:
                analyzed_structures.append({
                    "name": lattice_match[0],
                    "specific_energy_absorption": float(lattice_match[1])
                })
        
        parsed_data["performance_metrics"]["analyzed_structures"] = analyzed_structures
        
        # Parse numerical parameters.
        resolution_matches = re.findall(r'Resolution:\s*(\d+)x(\d+)x(\d+)', terminal_output)
        if resolution_matches:
            parsed_data["performance_metrics"]["resolution"] = f"{resolution_matches[0][0]}x{resolution_matches[0][1]}x{resolution_matches[0][2]}"
        
        # Parse voxel data.
        voxel_matches = re.findall(r'Solid voxel count:\s*([0-9,]+)', terminal_output)
        if voxel_matches:
            parsed_data["performance_metrics"]["solid_voxels"] = int(voxel_matches[0].replace(',', ''))
        
        return parsed_data
    
    def categorize_requirement(self, requirement: str) -> str:
        """Categorize a requirement."""
        requirement_lower = requirement.lower()
        if any(word in requirement_lower for word in ["query", "find", "database", "search"]):
            return "database_query"
        elif any(word in requirement_lower for word in ["design", "new", "create", "generate"]):
            return "design_generation"
        elif any(word in requirement_lower for word in ["optimize", "improve", "enhance"]):
            return "optimization"
        else:
            return "general"
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from system output."""
        keywords = []
        
        # Keyword patterns based on actual system output.
        keyword_patterns = [
            # Requirement-type keywords.
            r'(query|find|filter|select|search|retrieve)',
            r'(design|generate|create|construct|model)',
            r'(optimize|improve|enhance|boost)',
            r'(simulate|simulation|compute|analysis|evaluate)',
            
            # Technical-metric keywords.
            r'(specific energy absorption|energy absorption|absorption_energy)',
            r'(heat conduction|thermal conductivity|conductivity|heat)',
            r'(microstructure|structure|lattice|porous)',
            r'(volume fraction|porosity|vof)',
            
            # Material-property keywords.
            r'(youngs modulus|shear modulus|poisson ratio|E|G|nu)',
            r'(anisotropic|isotropic|anisotropy)',
            r'(voxel|resolution|mesh)',
            
            # Application-domain keywords.
            r'(helmet|padding|protective|protection)',
            r'(lightweight|weight|mass)',
            r'(crash|collision|impact)',
            
            # Numeric values and units.
            r'(\d+\.?\d*\s*(GPa|W/\(m\*K\)|J/kg|%))',
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                # Handle grouped matches.
                keywords.extend([match[0] for match in matches])
            else:
                keywords.extend(matches)
        
        # Deduplicate while preserving order.
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        return unique_keywords[:20]  # Limit the result to 20 keywords.
    
    def assess_complexity(self, requirement: str) -> str:
        """Assess requirement complexity."""
        requirement_lower = requirement.lower()
        complexity_indicators = 0
        
        if any(word in requirement_lower for word in ["simultaneously", "multiple", "combined"]):
            complexity_indicators += 1
        if any(word in requirement_lower for word in ["optimize", "maximize", "minimize", "best"]):
            complexity_indicators += 1
        if any(word in requirement_lower for word in ["simulate", "analysis", "compute"]):
            complexity_indicators += 1
        if len(requirement.split()) > 15:
            complexity_indicators += 1
            
        if complexity_indicators >= 3:
            return "high"
        elif complexity_indicators >= 2:
            return "medium"
        else:
            return "low"
    
    def determine_workflow_type(self, requirement: str) -> str:
        """Determine the workflow type."""
        if "query" in requirement.lower() or "find" in requirement.lower() or "database" in requirement.lower():
            return "query_workflow"
        elif "design" in requirement.lower():
            return "design_workflow"
        else:
            return "mixed_workflow"
    
    def analyze_workflow_pattern(self, flow_analysis: Dict) -> str:
        """Analyze the workflow pattern."""
        agent_sequence = flow_analysis.get("agent_sequence", [])
        
        if "RequirementParser" in agent_sequence and "CodeGenerator" in agent_sequence and "CodeExecutor" in agent_sequence:
            if "OptimizationPlanner" in agent_sequence:
                return "query_with_simulation"
            else:
                return "database_query_only"
        elif "StructureGenerator" in agent_sequence:
            return "generative_design"
        else:
            return "custom_workflow"
    
    def extract_unique_tools(self, tool_calls: List[Dict]) -> List[str]:
        """Extract unique tool names."""
        tools = set()
        for tool in tool_calls:
            if "tool_name" in tool:
                tools.add(tool["tool_name"])
            elif "function_name" in tool:
                tools.add(tool["function_name"])
        return list(tools)
    
    def extract_key_findings(self, terminal_output: str) -> List[str]:
        """Extract key findings."""
        findings = []
        
        # Patterns used to identify key findings.
        finding_patterns = [
            r'Key findings:\s*([^\n]+)',
            r'Important conclusion:\s*([^\n]+)',
            r'Main features:\s*([^\n]+)',
            r'Performance characteristics:\s*([^\n]+)',
        ]
        
        for pattern in finding_patterns:
            matches = re.findall(pattern, terminal_output, re.IGNORECASE)
            findings.extend(matches)
        
        # Add numeric findings.
        if "heat conduction" in terminal_output.lower():
            findings.append("Completed heat-conduction simulation analysis")
        if "specific energy absorption" in terminal_output.lower():
            findings.append("Successfully selected microstructures with high specific energy absorption")
        if "voxel" in terminal_output.lower():
            findings.append("Used a high-resolution voxelized mesh")
            
        return findings[:10]  # Return at most 10 key findings.
    
    def extract_quantitative_results(self, parsed_output: Dict) -> Dict[str, Any]:
        """Extract quantitative results."""
        quantitative = {}
        
        perf_metrics = parsed_output.get("performance_metrics", {})
        
        if "specific_energy_absorption" in perf_metrics:
            quantitative["max_specific_energy"] = max(perf_metrics["specific_energy_absorption"])
            quantitative["min_specific_energy"] = min(perf_metrics["specific_energy_absorption"])
            
        if "solid_voxels" in perf_metrics:
            quantitative["solid_voxels"] = perf_metrics["solid_voxels"]
            
        if "resolution" in perf_metrics:
            quantitative["simulation_resolution"] = perf_metrics["resolution"]
            
        # Extract thermal conductivity from simulation results.
        sim_results = parsed_output.get("simulation_results", [])
        heat_results = [r for r in sim_results if r.get("type") == "heat_conductivity"]
        if heat_results:
            avg_conductivities = [r["data"].get("avg_conductivity", 0) for r in heat_results]
            if avg_conductivities:
                quantitative["avg_thermal_conductivity"] = sum(avg_conductivities) / len(avg_conductivities)
        
        return quantitative
    
    def extract_structure_names(self, performance_metrics: Dict) -> List[str]:
        """Extract structure names."""
        structures = []
        
        if "analyzed_structures" in performance_metrics:
            if isinstance(performance_metrics["analyzed_structures"], list):
                for item in performance_metrics["analyzed_structures"]:
                    if isinstance(item, dict) and "name" in item:
                        structures.append(item["name"])
                    elif isinstance(item, str):
                        structures.append(item)
        
        return structures
    
    def extract_computational_info(self, terminal_output: str) -> Dict[str, Any]:
        """Extract computational information."""
        comp_info = {}
        
        # GPU usage.
        if "using gpu" in terminal_output.lower() or "cuda" in terminal_output.lower():
            comp_info["gpu_used"] = True
            gpu_match = re.search(r'cuda:(\d+)', terminal_output)
            if gpu_match:
                comp_info["gpu_device"] = f"cuda:{gpu_match.group(1)}"
        
        # Processing time.
        time_pattern = r'Processing time:\s*([0-9.]+)\s*seconds'
        time_matches = re.findall(time_pattern, terminal_output)
        if time_matches:
            comp_info["processing_time_seconds"] = float(time_matches[0])
        
        # Success rate.
        success_pattern = r'success_rate["\']?\s*[:=]\s*([0-9.]+)'
        success_match = re.search(success_pattern, terminal_output)
        if success_match:
            comp_info["success_rate"] = float(success_match.group(1))
        
        return comp_info
    
    def extract_simulation_parameters(self, terminal_output: str) -> Dict[str, Any]:
        """Extract simulation parameters."""
        parameters = {}
        
        # Extract resolution.
        resolution_pattern = r'resolution["\']?\s*[:=]\s*(\d+)'
        resolution_match = re.search(resolution_pattern, terminal_output)
        if resolution_match:
            parameters["resolution"] = int(resolution_match.group(1))
        
        # Extract GPU-use flag.
        gpu_pattern = r'use_gpu["\']?\s*[:=]\s*(true|false|True|False)'
        gpu_match = re.search(gpu_pattern, terminal_output)
        if gpu_match:
            parameters["use_gpu"] = gpu_match.group(1).lower() == "true"
        
        # Extract mode.
        mode_pattern = r'mode["\']?\s*[:=]\s*["\']([^"\']+)["\']'
        mode_match = re.search(mode_pattern, terminal_output)
        if mode_match:
            parameters["mode"] = mode_match.group(1)
            
        return parameters
    
    def calculate_completeness_score(self, parsed_output: Dict) -> float:
        """Calculate a completeness score in the range 0-1."""
        score = 0.0
        
        # Workflow completeness.
        if parsed_output["flow_analysis"]["total_agents"] >= 3:
            score += 0.3
        
        # Tool-use completeness.
        if parsed_output["flow_analysis"]["total_tool_calls"] >= 2:
            score += 0.3
        
        # Result completeness.
        if parsed_output["simulation_results"]:
            score += 0.2
        
        # Performance-metric completeness.
        if parsed_output["performance_metrics"]:
            score += 0.2
        
        return min(score, 1.0)
    
    def calculate_efficiency_score(self, parsed_output: Dict) -> float:
        """Calculate an efficiency score."""
        total_rounds = parsed_output["flow_analysis"]["total_rounds"]
        total_agents = parsed_output["flow_analysis"]["total_agents"]
        
        if total_rounds == 0:
            return 0.0
        
        # Ideally, every agent replies once.
        efficiency = min(total_agents / total_rounds, 1.0)
        return efficiency
    
    def assess_result_reliability(self, parsed_output: Dict) -> str:
        """Assess result reliability."""
        reliability_score = 0
        
        # Simulation results are present.
        if parsed_output["simulation_results"]:
            reliability_score += 2
        
        # Quantitative data are present.
        if parsed_output["performance_metrics"]:
            reliability_score += 2
        
        # The workflow is complete.
        if parsed_output["flow_analysis"]["total_agents"] >= 4:
            reliability_score += 1
        
        if reliability_score >= 4:
            return "high"
        elif reliability_score >= 2:
            return "medium"
        else:
            return "low"
    
    def create_case_entry(self, session_id: str, user_requirement: str, 
                         terminal_output: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a unified case entry."""
        parsed_output = self.parse_terminal_output(terminal_output)
        
        case_entry = {
            "case_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user_requirement": {
                "original_text": user_requirement,
                "category": self.categorize_requirement(user_requirement),
                "keywords": self.extract_keywords(user_requirement),
                "complexity_level": self.assess_complexity(user_requirement)
            },
            "workflow_info": {
                "workflow_type": self.determine_workflow_type(user_requirement),
                "agent_sequence": parsed_output["flow_analysis"]["agent_sequence"],
                "agent_reply_count": parsed_output["flow_analysis"]["agent_reply_count"],
                "total_agents_involved": parsed_output["flow_analysis"]["total_agents"],
                "total_rounds": parsed_output["flow_analysis"]["total_rounds"],
                "total_interactions": len(parsed_output["agent_interactions"]),
                "workflow_pattern": self.analyze_workflow_pattern(parsed_output["flow_analysis"])
            },
            "tool_usage": {
                "tools_called": self.extract_unique_tools(parsed_output["tool_calls"]),
                "total_tool_calls": parsed_output["flow_analysis"]["total_tool_calls"],
                "tool_details": parsed_output["tool_calls"],
                "successful_executions": len([t for t in parsed_output["tool_calls"] if t.get("status") == "executed"]),
                "failed_executions": len([t for t in parsed_output["tool_calls"] if t.get("status") == "failed"])
            },
            "results": {
                "simulation_results": parsed_output["simulation_results"],
                "performance_metrics": parsed_output["performance_metrics"],
                "success": result.get("status") == "completed",
                "final_status": result.get("status", "unknown"),
                "key_findings": self.extract_key_findings(terminal_output),
                "quantitative_results": self.extract_quantitative_results(parsed_output)
            },
            "technical_details": {
                "structures_analyzed": self.extract_structure_names(parsed_output["performance_metrics"]),
                "simulation_types": [sim["type"] for sim in parsed_output["simulation_results"]],
                "key_parameters": self.extract_simulation_parameters(terminal_output),
                "computational_info": self.extract_computational_info(terminal_output)
            },
            "quality_metrics": {
                "completeness_score": self.calculate_completeness_score(parsed_output),
                "data_richness": len(parsed_output["performance_metrics"]),
                "workflow_efficiency": self.calculate_efficiency_score(parsed_output),
                "result_reliability": self.assess_result_reliability(parsed_output)
            },
            "satisfaction": True,  # Saving the entry implies satisfaction.
            "raw_data": {
                "terminal_output": terminal_output,
                "session_result_path": result.get("session_result_path"),
                "summary_report_path": result.get("summary_report_path")
            }
        }
        
        return case_entry
    
    def save_case(self, case_entry: Dict[str, Any]) -> bool:
        """Save a case to the case library."""
        try:
            # Read the existing case library.
            existing_cases = []
            if os.path.exists(self.unified_cases_path):
                try:
                    with open(self.unified_cases_path, "r", encoding="utf-8") as f:
                        existing_cases = json.load(f)
                        if not isinstance(existing_cases, list):
                            existing_cases = []
                except Exception as e:
                    self.logger.error(f"Failed to read the existing case library: {e}")
                    existing_cases = []

            # Add the new case.
            existing_cases.append(case_entry)

            # Save the updated case library.
            with open(self.unified_cases_path, "w", encoding="utf-8") as f:
                json.dump(existing_cases, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Case saved to the unified case library: {self.unified_cases_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save the unified case library: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Return case-library statistics."""
        if not os.path.exists(self.unified_cases_path):
            return {"total_cases": 0, "categories": {}, "tools_usage": {}, "agents_usage": {}}
        
        try:
            with open(self.unified_cases_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
            
            stats = {
                "total_cases": len(cases),
                "categories": {},
                "workflow_types": {},
                "workflow_patterns": {},
                "complexity_distribution": {},
                "tools_usage": {},
                "agents_usage": {},
                "agent_reply_patterns": {},
                "simulation_types": {},
                "success_rate": 0.0,
                "avg_completeness_score": 0.0,
                "recent_cases": [],
                "performance_insights": {},
                "computational_stats": {}
            }
            
            total_completeness = 0
            successful_cases = 0
            total_rounds = []
            gpu_usage_count = 0
            
            for case in cases:
                # Basic statistics.
                category = case["user_requirement"]["category"]
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                
                workflow_type = case["workflow_info"]["workflow_type"]
                stats["workflow_types"][workflow_type] = stats["workflow_types"].get(workflow_type, 0) + 1
                
                # Workflow-pattern statistics.
                workflow_pattern = case["workflow_info"].get("workflow_pattern", "unknown")
                stats["workflow_patterns"][workflow_pattern] = stats["workflow_patterns"].get(workflow_pattern, 0) + 1
                
                # Complexity distribution.
                complexity = case["user_requirement"].get("complexity_level", "unknown")
                stats["complexity_distribution"][complexity] = stats["complexity_distribution"].get(complexity, 0) + 1
                
                # Tool-use statistics.
                for tool in case["tool_usage"]["tools_called"]:
                    stats["tools_usage"][tool] = stats["tools_usage"].get(tool, 0) + 1
                
                # Agent-use statistics.
                for agent in case["workflow_info"]["agent_sequence"]:
                    stats["agents_usage"][agent] = stats["agents_usage"].get(agent, 0) + 1
                
                # Agent reply-count patterns.
                agent_reply_count = case["workflow_info"].get("agent_reply_count", {})
                for agent, count in agent_reply_count.items():
                    if agent not in stats["agent_reply_patterns"]:
                        stats["agent_reply_patterns"][agent] = []
                    stats["agent_reply_patterns"][agent].append(count)
                
                # Simulation-type statistics.
                for sim in case["results"]["simulation_results"]:
                    sim_type = sim["type"]
                    stats["simulation_types"][sim_type] = stats["simulation_types"].get(sim_type, 0) + 1
                
                # Success-rate and completeness statistics.
                if case["results"]["success"]:
                    successful_cases += 1
                
                completeness = case.get("quality_metrics", {}).get("completeness_score", 0)
                total_completeness += completeness
                
                # Round-count statistics.
                rounds = case["workflow_info"].get("total_rounds", 0)
                if rounds > 0:
                    total_rounds.append(rounds)
                
                # Computational-resource statistics.
                comp_info = case["technical_details"].get("computational_info", {})
                if comp_info.get("gpu_used", False):
                    gpu_usage_count += 1
                
                # Collect recent cases.
                if len(stats["recent_cases"]) < 5:
                    stats["recent_cases"].append({
                        "case_id": case["case_id"],
                        "timestamp": case["timestamp"],
                        "requirement": case["user_requirement"]["original_text"][:100] + "..." if len(case["user_requirement"]["original_text"]) > 100 else case["user_requirement"]["original_text"],
                        "workflow_pattern": workflow_pattern,
                        "tools_count": len(case["tool_usage"]["tools_called"])
                    })
            
            # Calculate aggregate statistics.
            if cases:
                stats["success_rate"] = successful_cases / len(cases)
                stats["avg_completeness_score"] = total_completeness / len(cases)
            
            # Analyze agent reply patterns.
            for agent, counts in stats["agent_reply_patterns"].items():
                stats["agent_reply_patterns"][agent] = {
                    "avg_replies": sum(counts) / len(counts),
                    "max_replies": max(counts),
                    "min_replies": min(counts)
                }
            
            # Performance insights.
            if total_rounds:
                stats["performance_insights"] = {
                    "avg_workflow_rounds": sum(total_rounds) / len(total_rounds),
                    "min_workflow_rounds": min(total_rounds),
                    "max_workflow_rounds": max(total_rounds)
                }
            
            # Computational-resource statistics.
            stats["computational_stats"] = {
                "gpu_usage_rate": gpu_usage_count / len(cases) if cases else 0,
                "most_used_tools": sorted(stats["tools_usage"].items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get case-library statistics: {e}")
            return {"total_cases": 0, "categories": {}, "tools_usage": {}, "agents_usage": {}, "error": str(e)}
    
    def search_similar_cases(self, user_requirement: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar cases."""
        if not os.path.exists(self.unified_cases_path):
            return []
        
        try:
            with open(self.unified_cases_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
            
            # Extract features from the current requirement.
            keywords = self.extract_keywords(user_requirement)
            category = self.categorize_requirement(user_requirement)
            complexity = self.assess_complexity(user_requirement)
            workflow_type = self.determine_workflow_type(user_requirement)
            
            scored_cases = []
            for case in cases:
                score = 0
                
                # Category match (highest weight).
                if case["user_requirement"]["category"] == category:
                    score += 5
                
                # Workflow-type match.
                if case["workflow_info"]["workflow_type"] == workflow_type:
                    score += 4
                    
                # Complexity match.
                case_complexity = case["user_requirement"].get("complexity_level", "unknown")
                if case_complexity == complexity:
                    score += 2
                
                # Keyword match.
                case_keywords = case["user_requirement"]["keywords"]
                common_keywords = set(keywords) & set(case_keywords)
                score += len(common_keywords) * 2
                
                # Text similarity (word overlap).
                case_text = case["user_requirement"]["original_text"].lower()
                user_text = user_requirement.lower()
                common_words = set(case_text.split()) & set(user_text.split())
                score += len(common_words) * 0.5
                
                # Bonus for successful cases.
                if case.get("results", {}).get("success", False):
                    score += 3
                
                # Bonus for high-quality cases.
                quality_metrics = case.get("quality_metrics", {})
                completeness = quality_metrics.get("completeness_score", 0)
                reliability = quality_metrics.get("result_reliability", "low")
                
                if completeness > 0.8:
                    score += 2
                if reliability == "high":
                    score += 2
                elif reliability == "medium":
                    score += 1
                
                if score > 0:
                    scored_cases.append((score, case))
            
            # Sort by score and return the top N cases.
            scored_cases.sort(key=lambda x: x[0], reverse=True)
            return [case for score, case in scored_cases[:limit]]
            
        except Exception as e:
            self.logger.error(f"Failed to search for similar cases: {e}")
            return []
    
    def get_workflow_guidance(self, user_requirement: str) -> Dict[str, Any]:
        """Return workflow guidance."""
        similar_cases = self.search_similar_cases(user_requirement, limit=3)
        
        if not similar_cases:
            return {"guidance": "no_cases", "recommendation": "use_default_workflow"}
        
        # Analyze workflow patterns of similar cases.
        workflow_patterns = {}
        agent_sequences = []
        success_patterns = []
        
        for case in similar_cases:
            workflow_info = case.get("workflow_info", {})
            pattern = workflow_info.get("workflow_pattern", "unknown")
            sequence = workflow_info.get("agent_sequence", [])
            success = case.get("results", {}).get("success", False)
            
            workflow_patterns[pattern] = workflow_patterns.get(pattern, 0) + 1
            agent_sequences.append(sequence)
            
            if success:
                success_patterns.append({
                    "pattern": pattern,
                    "sequence": sequence,
                    "rounds": workflow_info.get("total_rounds", 0),
                    "efficiency": case.get("quality_metrics", {}).get("workflow_efficiency", 0)
                })
        
        # Recommend the best pattern.
        best_pattern = max(workflow_patterns.items(), key=lambda x: x[1])[0] if workflow_patterns else "unknown"
        
        # Generate agent-invocation guidance.
        if success_patterns:
            # Select the most efficient successful pattern.
            best_success = min(success_patterns, key=lambda x: x.get("rounds", 999))
            recommended_sequence = best_success["sequence"]
        else:
            # Generate guidance from the most common sequence.
            if agent_sequences:
                # Select the most common first agent.
                first_agents = [seq[0] for seq in agent_sequences if seq]
                recommended_first = max(set(first_agents), key=first_agents.count) if first_agents else "RequirementParser"
                recommended_sequence = [recommended_first]
            else:
                recommended_sequence = ["RequirementParser"]
        
        return {
            "guidance": "cases_available",
            "similar_cases_count": len(similar_cases),
            "most_common_pattern": best_pattern,
            "recommended_sequence": recommended_sequence,
            "success_rate": len(success_patterns) / len(similar_cases) if similar_cases else 0,
            "average_rounds": sum(p.get("rounds", 0) for p in success_patterns) / len(success_patterns) if success_patterns else 0
        }
    
    def display_stats(self, stats: Dict[str, Any]):
        """Display case-library statistics."""
        if stats["total_cases"] > 0:
            print("\nCase Library Statistics:")
            print(f"  Total cases: {stats['total_cases']}")
            print(f"  Success rate: {stats.get('success_rate', 0):.2%}")
            print(f"  Average completeness score: {stats.get('avg_completeness_score', 0):.2f}")
            
            print(f"\nRequirement categories: {dict(stats['categories'])}")
            print(f"Workflow patterns: {dict(stats.get('workflow_patterns', {}))}")
            print(f"Complexity distribution: {dict(stats.get('complexity_distribution', {}))}")
            
            print("\nMost frequently used tools (top 5):")
            top_tools = sorted(stats['tools_usage'].items(), key=lambda x: x[1], reverse=True)[:5]
            for tool, count in top_tools:
                print(f"     {tool}: {count} uses")
            
            print("\nAgent activity:")
            for agent, count in sorted(stats['agents_usage'].items(), key=lambda x: x[1], reverse=True)[:5]:
                reply_pattern = stats.get('agent_reply_patterns', {}).get(agent, {})
                avg_replies = reply_pattern.get('avg_replies', 0) if isinstance(reply_pattern, dict) else 0
                print(f"     {agent}: {count} participations, {avg_replies:.1f} average replies")
            
            if stats.get("performance_insights"):
                insights = stats["performance_insights"]
                print("\nPerformance metrics:")
                print(f"     Average workflow rounds: {insights.get('avg_workflow_rounds', 0):.1f}")
                print(f"     Minimum rounds: {insights.get('min_workflow_rounds', 0)}")
                print(f"     Maximum rounds: {insights.get('max_workflow_rounds', 0)}")
            
            comp_stats = stats.get("computational_stats", {})
            if comp_stats:
                print("\nComputational resource usage:")
                print(f"     GPU utilization rate: {comp_stats.get('gpu_usage_rate', 0):.1%}")
            
            if stats["recent_cases"]:
                print("\nRecent cases:")
                for case in stats["recent_cases"]:
                    print(f"  - {case['case_id']}: {case['requirement']}")
                    print(f"    Pattern: {case.get('workflow_pattern', 'unknown')}, tools: {case.get('tools_count', 0)}")
        else:
            print("The case library is empty.")
    
    def display_case_library_header(self):
        """Display the case-library system header."""
        print("=== Microstructure Design Case Library ===")
        stats = self.get_stats()
        self.display_stats(stats)
        print("\n" + "="*80)
    
    def search_and_display_similar_cases(self, user_input: str, limit: int = 3):
        """Search for and display similar cases."""
        print(f"\nUser requirement: {user_input}")
        similar_cases = self.search_similar_cases(user_input, limit=limit)
        if similar_cases:
            print(f"\nFound {len(similar_cases)} similar cases:")
            for i, case in enumerate(similar_cases, 1):
                print(f"  {i}. {case['case_id']}: {case['user_requirement']['original_text'][:80]}...")
                print(f"     Workflow: {case['workflow_info']['workflow_type']}, tools: {', '.join(case['tool_usage']['tools_called'][:3])}")
        return similar_cases
    
    def run_design_workflow_with_capture(self, system, user_input: str):
        """Run the design workflow and capture terminal output."""
        with capture_terminal_output() as cap:
            result = asyncio.run(system.design_microstructure(user_input))

            print("Design completed.")
            print(f"Session ID: {result['session_id']}")
            print(f"Final result: {result['final_result']}")
            if result.get('summary_report_path'):
                print(f"Summary report generated: {result['summary_report_path']}")

        return result, cap.get("text", "")
    
    def prompt_and_save_case(self, session_id: str, user_input: str, terminal_output: str, result: Dict[str, Any], system):
        """Ask whether the user is satisfied and save the case."""
        try:
            answer = input("Are you satisfied with this design? [Y/n]: ").strip().lower()
            satisfied = answer in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            satisfied = False

        if satisfied:
            # Session-result path saved by the system.
            session_result_path = None
            try:
                # Save again to retrieve its path; this operation is idempotent.
                session_result_path = system._save_session_result(session_id, result)
            except Exception:
                pass

            # Create a structured case entry.
            case_entry = self.create_case_entry(
                session_id, user_input, terminal_output, result
            )

            # Save the case to the library.
            if self.save_case(case_entry):
                print(f"Case saved to the unified case library: {self.unified_cases_path}")
                
                # Get updated statistics.
                updated_stats = self.get_stats()
                print(f"The case library now contains {updated_stats['total_cases']} cases.")
                
                # Display key information from this case.
                print("Case analysis:")
                print(f"  - Workflow type: {case_entry['workflow_info']['workflow_type']}")
                print(f"  - Agents involved: {', '.join(case_entry['workflow_info']['agent_sequence'])}")
                print(f"  - Tool calls: {case_entry['tool_usage']['total_tool_calls']}")
                print(f"  - Structures analyzed: {', '.join(case_entry['technical_details']['structures_analyzed'])}")
        else:
            print("This case was not saved.")
