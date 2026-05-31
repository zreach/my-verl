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

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def prepend_system_prompt(prompt: Any, system_prompt: str) -> Any:
    if not isinstance(prompt, list):
        return prompt

    messages = list(prompt)
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": system_prompt + "\n\n" + str(messages[0].get("content", "")),
        }
        return messages

    return [{"role": "system", "content": system_prompt}] + messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject a system prompt into AIME parquet rows.")
    parser.add_argument("--source", required=True, help="Input AIME parquet.")
    parser.add_argument("--target", required=True, help="Output parquet.")
    parser.add_argument("--system-prompt", required=True, help="Path to system prompt text.")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    target = Path(args.target).expanduser()
    system_prompt = Path(args.system_prompt).expanduser().read_text(encoding="utf-8").strip()

    df = pd.read_parquet(source)
    df["prompt"] = df["prompt"].apply(lambda prompt: prepend_system_prompt(prompt, system_prompt))

    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target)
    print(f"Wrote {len(df)} AIME rows with system prompt to {target}")


if __name__ == "__main__":
    main()
