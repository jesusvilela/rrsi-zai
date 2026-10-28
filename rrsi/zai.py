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
"""Z.ai (GLM) routing for the frozen policy model.

The benchmark runners reach the policy through LiteLLM. On Z.ai the policy is
LiteLLM's ``anthropic`` provider pointed at Z.ai's Anthropic-compatible
endpoint. The key and base URL go only into the runner subprocess's
environment, so a global ANTHROPIC_API_KEY (e.g. another tool's) is untouched.

Active when rrsi.llm.PROVIDER is ``zai`` (ZAI_API_KEY set, no Vertex projects,
or RRSI_LLM_PROVIDER=zai). ZAI_POLICY_MODEL overrides the policy model.
"""

from __future__ import annotations

import os

from .llm import PROVIDER, ZAI_BASE_URL

POLICY_MODEL = os.environ.get("ZAI_POLICY_MODEL", "anthropic/glm-5.3")


def enabled() -> bool:
    return PROVIDER == "zai"


def policy_model(configured: str) -> str:
    """The configured LiteLLM model string, or GLM when it names a Vertex/Claude model."""
    if enabled() and (configured.startswith("vertex_ai/") or configured.startswith("claude-")):
        return POLICY_MODEL
    return configured


def litellm_env() -> dict:
    """Subprocess env entries that point LiteLLM's anthropic provider at Z.ai."""
    if not enabled():
        return {}
    key = os.environ.get("ZAI_API_KEY", "")
    return {"ANTHROPIC_API_KEY": key, "ANTHROPIC_API_BASE": ZAI_BASE_URL,
            "ANTHROPIC_BASE_URL": ZAI_BASE_URL}
