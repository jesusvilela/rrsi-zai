# Copyright 2026 The rrsi Authors.
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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM client for the three search roles (proposer, analyst, critic).

Two providers speak the same Anthropic Messages API:

* ``vertex``: Claude Opus 4.8 via AnthropicVertex, round-robin over the
  configured GCP projects (RRSI_VERTEX_PROJECTS) with retry-and-rotate.
* ``zai``: GLM via Z.ai's Anthropic-compatible endpoint, keyed by ZAI_API_KEY.

RRSI_LLM_PROVIDER picks one explicitly; otherwise ``vertex`` when Vertex
projects are configured and ``zai`` when only ZAI_API_KEY is set.
`cache_prefix` sends a large stable leading block (constitution + harness
source) as its own ephemeral-cached content block so repeated turns pay to
read it once.
"""

from __future__ import annotations

import itertools
import os
import re
import threading
import time

_PROJECTS = [
    {"project_id": p.strip(), "region": os.environ.get("RRSI_VERTEX_REGION", "global")}
    for p in os.environ.get("RRSI_VERTEX_PROJECTS", "").split(",")
    if p.strip()
]
ZAI_BASE_URL = os.environ.get("ZAI_ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
PROVIDER = os.environ.get("RRSI_LLM_PROVIDER") or (
    "vertex" if _PROJECTS or not os.environ.get("ZAI_API_KEY") else "zai")
MODEL = os.environ.get("RRSI_SEARCH_MODEL",
                       "glm-5.3" if PROVIDER == "zai" else "claude-opus-4-8")
MAX_TOKENS = 20_000

_clients: dict = {}
_clients_lock = threading.Lock()
_rr = itertools.count()
_JSON_SUFFIX = ("\n\nOutput ONLY a single valid JSON object. No prose before or "
                "after, no markdown fences.")


def _n_routes() -> int:
    return 1 if PROVIDER == "zai" else max(len(_PROJECTS), 1)


def _client_for(idx: int):
    if PROVIDER == "zai":
        from anthropic import Anthropic
        key = os.environ.get("ZAI_API_KEY")
        if not key:
            raise RuntimeError("RRSI_LLM_PROVIDER=zai needs ZAI_API_KEY in the environment")
        with _clients_lock:
            c = _clients.get("zai")
            if c is None:
                c = Anthropic(api_key=key, base_url=ZAI_BASE_URL)
                _clients["zai"] = c
            return c
    from anthropic import AnthropicVertex
    if not _PROJECTS:
        raise RuntimeError("set RRSI_VERTEX_PROJECTS to a comma-separated list of GCP "
                           "projects with Claude on Vertex AI enabled")
    with _clients_lock:
        c = _clients.get(idx)
        if c is None:
            c = AnthropicVertex(**_PROJECTS[idx])
            _clients[idx] = c
        return c


def extract_json(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{") and not t.startswith("["):
        start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)
        if start != -1:
            t = t[start:]
    if t and t[0] == "{" and not t.endswith("}"):
        end = t.rfind("}")
        if end != -1:
            t = t[:end + 1]
    if t and t[0] == "[" and not t.endswith("]"):
        end = t.rfind("]")
        if end != -1:
            t = t[:end + 1]
    return t


def generate(prompt: str, system: str | None = None, max_retries: int = 6,
             json_only: bool = False, model: str | None = None,
             max_tokens: int = MAX_TOKENS, cache_prefix: str | None = None) -> str:
    mdl = model or MODEL
    if PROVIDER == "zai" and mdl.startswith("claude-"):
        mdl = MODEL     # a config written for Vertex; Z.ai serves GLM
    sys_prompt = (system or "") + (_JSON_SUFFIX if json_only else "")
    content = ([{"type": "text", "text": cache_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt}] if cache_prefix else prompt)
    n = _n_routes()
    start = next(_rr)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        idx = (start + attempt) % n
        try:
            client = _client_for(idx)
            kwargs = {"model": mdl, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": content}]}
            if sys_prompt:
                kwargs["system"] = sys_prompt
            resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text")
            if text:
                return extract_json(text) if json_only else text
            last_err = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001 - rotate to the other project / retry
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"generate failed after {max_retries} tries: {last_err}")


if __name__ == "__main__":
    print(generate("Reply with exactly: OK", max_tokens=16))
