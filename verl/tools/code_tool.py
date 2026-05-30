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
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


_FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class CodeTool(BaseTool):
    """Execute Python code snippets as a native verl tool."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config=config, tool_schema=tool_schema)
        self.default_timeout = float(self.config.get("default_timeout", 10))
        self.max_timeout = float(self.config.get("max_timeout", 30))
        self.max_output_chars = int(self.config.get("max_output_chars", 12000))
        self.python_executable = self.config.get("python_executable", sys.executable)
        self.allowed_languages = set(self.config.get("allowed_languages", ["python", "py"]))
        self._instance_dirs: dict[str, str] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "Execute a Python code snippet and return stdout, stderr, and exit status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. Markdown fenced code blocks are accepted.",
                            },
                            "stdin": {
                                "type": "string",
                                "description": "Optional standard input passed to the program.",
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Optional execution timeout in seconds.",
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language. Only python/py is supported by this local tool.",
                            },
                        },
                        "required": ["code"],
                    },
                },
            }
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        self._instance_dirs[instance_id] = tempfile.mkdtemp(prefix=f"verl-code-tool-{instance_id}-")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = self._normalize_code(parameters.get("code", ""))
        language = str(parameters.get("language", self.config.get("default_language", "python"))).lower()
        stdin = parameters.get("stdin", "")
        timeout = self._normalize_timeout(parameters.get("timeout"))

        if language not in self.allowed_languages:
            text = f"Unsupported language '{language}'. Supported languages: {sorted(self.allowed_languages)}"
            return ToolResponse(text=text), 0.0, {"status": "unsupported_language", "language": language}

        if not code.strip():
            return ToolResponse(text="No code provided."), 0.0, {"status": "empty_code"}

        workdir = self._instance_dirs.get(instance_id)
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix=f"verl-code-tool-{instance_id}-")
            self._instance_dirs[instance_id] = workdir

        result = await asyncio.to_thread(self._run_python, code, str(stdin), timeout, workdir)
        text = self._format_result(result)
        metrics = {
            "status": result["status"],
            "returncode": result["returncode"],
            "timed_out": result["timed_out"],
            "stdout_chars": len(result["stdout"]),
            "stderr_chars": len(result["stderr"]),
        }
        return ToolResponse(text=text), 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        workdir = self._instance_dirs.pop(instance_id, None)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    def _normalize_code(self, code: Any) -> str:
        if not isinstance(code, str):
            code = str(code)
        match = _FENCED_CODE_RE.search(code)
        return match.group(1).strip() if match else code

    def _normalize_timeout(self, timeout: Any) -> float:
        if timeout is None:
            return self.default_timeout
        try:
            return min(max(float(timeout), 0.1), self.max_timeout)
        except (TypeError, ValueError):
            return self.default_timeout

    def _run_python(self, code: str, stdin: str, timeout: float, workdir: str) -> dict[str, Any]:
        code_path = os.path.join(workdir, "main.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            completed = subprocess.run(
                [self.python_executable, code_path],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                check=False,
            )
            status = "success" if completed.returncode == 0 else "error"
            return {
                "status": status,
                "returncode": completed.returncode,
                "stdout": self._truncate(completed.stdout),
                "stderr": self._truncate(completed.stderr),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ""
            stderr = e.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return {
                "status": "timeout",
                "returncode": None,
                "stdout": self._truncate(stdout),
                "stderr": self._truncate(stderr),
                "timed_out": True,
            }

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        keep = max(self.max_output_chars - len("\n...(truncated)"), 0)
        return text[:keep] + "\n...(truncated)"

    def _format_result(self, result: dict[str, Any]) -> str:
        parts = [f"status: {result['status']}"]
        if result["returncode"] is not None:
            parts.append(f"returncode: {result['returncode']}")
        if result["stdout"]:
            parts.append(f"stdout:\n{result['stdout'].rstrip()}")
        if result["stderr"]:
            parts.append(f"stderr:\n{result['stderr'].rstrip()}")
        return "\n".join(parts)
