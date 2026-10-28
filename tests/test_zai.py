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
"""Provider selection and Z.ai policy routing (no network)."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PROBE = """
import json
from rrsi import llm, zai
print(json.dumps({
    "provider": llm.PROVIDER, "model": llm.MODEL,
    "policy_vertex": zai.policy_model("vertex_ai/claude-opus-4-8"),
    "policy_other": zai.policy_model("openai/gpt-x"),
    "env_keys": sorted(zai.litellm_env()),
}))
"""


def _probe(**env):
    base = {k: v for k, v in os.environ.items()
            if k not in ("ZAI_API_KEY", "RRSI_VERTEX_PROJECTS", "RRSI_LLM_PROVIDER",
                         "RRSI_SEARCH_MODEL", "ZAI_POLICY_MODEL")}
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, env={**base, **env},
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_defaults_to_vertex_without_zai_key():
    r = _probe()
    assert r["provider"] == "vertex" and r["model"] == "claude-opus-4-8"
    assert r["policy_vertex"] == "vertex_ai/claude-opus-4-8" and r["env_keys"] == []


def test_zai_key_alone_selects_zai():
    r = _probe(ZAI_API_KEY="test")
    assert r["provider"] == "zai" and r["model"] == "glm-5.3"
    assert r["policy_vertex"] == "anthropic/glm-5.3"
    assert r["policy_other"] == "openai/gpt-x"          # explicit non-Vertex models are kept
    assert r["env_keys"] == ["ANTHROPIC_API_BASE", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]


def test_vertex_projects_win_unless_forced():
    assert _probe(ZAI_API_KEY="t", RRSI_VERTEX_PROJECTS="p")["provider"] == "vertex"
    assert _probe(ZAI_API_KEY="t", RRSI_VERTEX_PROJECTS="p",
                  RRSI_LLM_PROVIDER="zai")["provider"] == "zai"


def test_model_overrides():
    r = _probe(ZAI_API_KEY="t", RRSI_SEARCH_MODEL="glm-5.3-flash",
               ZAI_POLICY_MODEL="anthropic/glm-5.3-flash")
    assert r["model"] == "glm-5.3-flash" and r["policy_vertex"] == "anthropic/glm-5.3-flash"
