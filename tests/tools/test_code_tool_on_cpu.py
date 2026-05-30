# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

from verl.tools.code_tool import CodeTool
from verl.tools.tool_registry import load_all_tools


def _make_tool(**config) -> CodeTool:
    merged = {"type": "native", "default_timeout": 1, "max_timeout": 2}
    merged.update(config)
    return CodeTool(config=merged, tool_schema=None)


def test_code_tool_executes_python_and_returns_stdout():
    async def _drive():
        tool = _make_tool()
        instance_id, _ = await tool.create()
        try:
            response, reward, metrics = await tool.execute(instance_id, {"code": "print(2 + 3)"})
        finally:
            await tool.release(instance_id)

        assert "status: success" in response.text
        assert "stdout:\n5" in response.text
        assert reward == 0.0
        assert metrics["status"] == "success"
        assert metrics["returncode"] == 0

    asyncio.run(_drive())


def test_code_tool_extracts_markdown_fenced_python():
    async def _drive():
        tool = _make_tool()
        instance_id, _ = await tool.create()
        try:
            response, _, _ = await tool.execute(instance_id, {"code": "```python\nprint('ok')\n```"})
        finally:
            await tool.release(instance_id)

        assert "stdout:\nok" in response.text

    asyncio.run(_drive())


def test_code_tool_reports_timeout():
    async def _drive():
        tool = _make_tool(default_timeout=0.1, max_timeout=0.1)
        instance_id, _ = await tool.create()
        try:
            response, _, metrics = await tool.execute(
                instance_id,
                {"code": "import time\ntime.sleep(1)\nprint('late')"},
            )
        finally:
            await tool.release(instance_id)

        assert "status: timeout" in response.text
        assert metrics["timed_out"] is True

    asyncio.run(_drive())


def test_code_tool_rejects_unsupported_language():
    async def _drive():
        tool = _make_tool()
        instance_id, _ = await tool.create()
        try:
            response, _, metrics = await tool.execute(instance_id, {"code": "puts 1", "language": "ruby"})
        finally:
            await tool.release(instance_id)

        assert "Unsupported language 'ruby'" in response.text
        assert metrics["status"] == "unsupported_language"

    asyncio.run(_drive())


def test_code_tool_loads_from_native_tool_yaml(tmp_path: Path):
    yaml_path = tmp_path / "tools.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            tools:
              - class_name: "verl.tools.code_tool.CodeTool"
                config:
                  type: native
                  default_timeout: 1
                  max_timeout: 2
            """
        )
    )

    tools = load_all_tools(tool_config_path=str(yaml_path), function_tool_path=None)

    assert len(tools) == 1
    assert isinstance(tools[0], CodeTool)
    assert tools[0].name == "code_interpreter"
    schema = tools[0].tool_schema.model_dump(exclude_unset=True, exclude_none=True)
    assert schema["function"]["parameters"]["required"] == ["code"]
