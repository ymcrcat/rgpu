#!/usr/bin/env bash
# Install the repository's rGPU skill for Claude Code and Codex.
#
# Override CLAUDE_SKILLS_DIR or CODEX_SKILLS_DIR to use non-default locations.
set -euo pipefail

cd "$(dirname "$0")/.."

source_file=skills/rgpu/SKILL.md
claude_skills_dir=${CLAUDE_SKILLS_DIR:-${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills}
codex_skills_dir=${CODEX_SKILLS_DIR:-${CODEX_HOME:-$HOME/.codex}/skills}

for skills_dir in "$claude_skills_dir" "$codex_skills_dir"; do
  install -d "$skills_dir/rgpu"
  install -m 0644 "$source_file" "$skills_dir/rgpu/SKILL.md"
  echo "installed rGPU skill in $skills_dir/rgpu"
done
