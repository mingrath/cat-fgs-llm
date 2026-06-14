# Pointer to Handoff for Next Fresh Session

**Created:** 2026-06-13

The full handoff document lives in the OS temp dir (per handoff skill rules — not committed here):

`/var/folders/1_/spr6kzz52zzdgvplwbcwts2r0000gn/T/handoff-cat-fgs-llm-pipeline-mcp-research.md`

**Next fresh context session must:**
1. Read the handoff file above first (it contains the complete briefing, artifact references, MCP protocol, dynamic workflow instructions, and "suggested skills").
2. Immediately execute the user's request: spawn as many subagents as possible, set up dynamic chat/polling/cross-critique workflow, use grep MCP (searchGitHub) + context7 MCP (resolve + query-docs) + firecrawl + local tools to thoroughly read this github repo (and public priors), and research concrete ways to make the cat-fgs-llm training pipeline (folds, cache, CORN, VLM, aug, eval, wrapper, gates) **better** or **more unique** than previous models built by others (Steagall, Feighelstein, Martvel, etc.).

See the handoff for verbatim text, key /tmp/ seeds (DYNAMIC_SUBAGENT_MCP_RESEARCH__PIPELINE_IMPROVEMENTS.md + all subagent_*_findings.md + chat + mcp_results + critic_hub), project context (FINAL_DIRECTION.md, README, src/..., configs/, scripts/, tests/), and exact "how to spawn + MCP + poll" protocol.

This pointer file is a convenience breadcrumb only. The authoritative handoff is the temp file. Do not duplicate the full content here.

**This session (handoff creation) did not run any subagents or MCP research.** All spawning/orchestration/MCP work is for the fresh next session after reading the handoff.
