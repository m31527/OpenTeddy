# OpenTeddy Cyber-Skills knowledge index

754 cybersecurity workflows from
[mukul975/Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills)
(Apache 2.0), each mapped to MITRE ATT&CK / NIST CSF / D3FEND / ATLAS /
AI RMF.

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

Each indexed skill carries its own header but the upstream is uniformly
Apache 2.0; redistribution conditions met by `cyber_skills/README.md`
linking to the source repo. We do NOT vendor the upstream's `scripts/`
or `references/` directories — those are pulled fresh by the user if
they need them (script content is executable code, security-sensitive
in itself, better not to silently ship).
