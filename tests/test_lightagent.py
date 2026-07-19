"""Regression tests for the local trusted-tool agent runtime."""

import json
import unittest
from types import SimpleNamespace

from lightagent import LightAgent


def ping(value: str):
    return {"value": value}


ping.tool_info = {
    "tool_name": "ping",
    "tool_title": "Ping",
    "tool_description": "Return the supplied value.",
    "tool_params": [
        {"name": "value", "description": "Value to return.", "type": "string", "required": True}
    ],
}


class LightAgentTests(unittest.TestCase):
    def test_trusted_tool_call_round_trip(self):
        agent = LightAgent(
            name="test",
            instructions="test",
            role="test",
            model="test",
            api_key="test-key",
            base_url="http://127.0.0.1:1",
            tools=[ping],
            temperature=0.2,
            max_tokens=256,
        )
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="ping", arguments=json.dumps({"value": "ok"})
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
            ),
        ]
        requests = []

        def create(**request):
            requests.append(request)
            return responses.pop(0)

        agent.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        self.assertEqual(agent.run("go"), "done")
        self.assertEqual(agent.get_tool("ping")("x"), {"value": "x"})
        self.assertEqual(agent.get_tools()[0]["function"]["name"], "ping")
        self.assertEqual(requests[0]["temperature"], 0.2)
        self.assertEqual(requests[0]["max_tokens"], 256)
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertIn('"value": "ok"', requests[1]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
