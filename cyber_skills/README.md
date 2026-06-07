# OpenTeddy Cyber-Skills knowledge index

A merged catalogue of expert workflows, indexed for fast retrieval by
the `cyber_skill_lookup` tool. Two upstream sources:

- **754 cybersecurity workflows** from
  [mukul975/Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills)
  (Apache 2.0), each mapped to MITRE ATT&CK / NIST CSF / D3FEND /
  ATLAS / NIST AI RMF.
- **1 multi-platform trend-research workflow** from
  [mvanhorn/last30days-skill](https://github.com/mvanhorn/last30days-skill)
  (MIT), covering Reddit / X / YouTube / HN / Polymarket with
  engagement-weighted ranking.

## How it integrates

These aren't executable Python skills (the format is the
[agentskills.io](https://agentskills.io) `SKILL.md` standard:
YAML frontmatter + structured workflow markdown). Instead the agent
calls a **lookup tool** (`cyber_skill_lookup`) at planning time —
the matched skill's `Workflow` section is returned as guidance markdown
that gets folded into the agent's reasoning. No automatic injection;
no context pollution on non-security tasks.

```
cyber_skills/
├── README.md       (this file)
├── index.json      (generated: [{name, description, domain, tags,
│                    frameworks, body}, ...])
└── update.py       (fetch / refresh from upstream)
```

## Refreshing the index

```bash
python cyber_skills/update.py
```

Pulls the latest SKILL.md files from the upstream repo via the GitHub
API + rebuilds `index.json`. Run after the upstream releases a new
version (they tag minor versions ~monthly).

## Licence

`cyber_skills/index.json` is a **derivative work** that combines the
SKILL.md text from the two upstream sources above (unmodified body
content, restructured for machine lookup). Each indexed skill carries
its own `source_repo` field so consumers can trace any entry back to
its origin.

Upstream licences:

- mukul975/Anthropic-Cybersecurity-Skills — **Apache 2.0**
- mvanhorn/last30days-skill — **MIT**

OpenTeddy itself is MIT-licensed; redistributing this index follows
both upstream licences (attribution preserved here + in every entry's
`source_repo` / `upstream_url` fields). We do NOT vendor the upstreams'
`scripts/` or `references/` directories — those are pulled fresh by
the user if they need them (script content is executable code,
security-sensitive in itself, better not to silently ship).
