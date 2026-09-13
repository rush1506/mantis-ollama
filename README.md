# Mantis: Portable Toolkit for Building Secure Software

> [!CAUTION] **USE AT YOUR OWN RISK. BE EXTREMELY CAREFUL.** This suite is
> designed to generate and execute autonomously generated code that may be
> unstable or perform unexpected actions. **USE THIS ONLY IN ISOLATED,
> RESTRICTED ENVIRONMENTS.** Never run this suite on a machine with access to
> production systems, sensitive data, or internal networks.

> [!IMPORTANT] **RESPONSIBLE USE** AI models are non-deterministic and can
> hallucinate findings or generate incorrect patches. **All findings must be
> manually verified by a security expert before being reported.** Do not
> mass-file unverified, AI-generated reports to open-source maintainers. A
> failure to automatically reproduce a vulnerability does not definitively mean
> it is a false positive, nor does a successful reproducer guarantee the bug is
> exploitable in all contexts. Use Mantis responsibly.

Mantis is a set of skills along with an ADK reference harness for building
secure software in the new AI era of software development.

## Quick Start (Ollama Cloud — DeepSeek)

The default model is **DeepSeek** (`ollama/deepseek-v4-flash:cloud`). This guide
gets Mantis running against **Ollama Cloud** so you can use hosted DeepSeek models
with no GPU and no local daemon. To run entirely locally instead, see
[Getting Started (Local)](#getting-started-local).

### 1. Prerequisites

- Python 3.10+ and `python3-venv`
- An **Ollama Cloud** account and API key from
  <https://ollama.com/settings/keys>
- (Recommended) the `ollama` CLI, to test connectivity: `ollama login`

### 2. Install the reference harness

```bash
cd reference && ./install.sh
```

### 3. Provide your API key (never committed)

Mantis loads credentials from a git-ignored `.env` file. Copy the template and
fill in your key:

```bash
cp reference/.env.example reference/.env
# edit reference/.env
#   OLLAMA_API_KEY=your-ollama-cloud-key
```

> Existing shell environment variables always win over `.env`, so you can also
> run `export OLLAMA_API_KEY=your-key` instead.

### 4. Verify the model routes to Ollama Cloud

The default model `ollama/deepseek-v4-flash:cloud` is auto-routed to
`https://ollama.com` (the Ollama-native `:cloud` endpoint). Confirm with a live
probe after configuration (step 5) — the preflight `--probe` flag will verify
credentials and reachability.

### 5. Configure and preflight

```bash
cd reference && source .venv/bin/activate

# Fast config & capability auto-detection (uses deepseek-v4-flash:cloud by default)
python3 scripts/configure.py --auto

# Preflight validation + live reachability probe (verifies your API key)
python3 scripts/configure.py --test --probe
```

If the probe reports `Unauthorized`, double-check the `OLLAMA_API_KEY` in your
`.env` / shell and rerun.

### 6. Launch a review campaign

```bash
# From reference/
./run.sh /path/to/target           # a source file or a repository directory

# Optional: target a specific objective
./run.sh /path/to/target --objective "Audit for Server-Side Request Forgery and SSRF in webhook handlers"
```

### Changing the model

To use a different DeepSeek model (e.g. `deepseek-v4-pro`), or any other model,
set it on the command line or in `workflow.json`:

```bash
./run.sh . --model ollama/deepseek-v4-pro:cloud
```

> **Routing refresher:** `ollama/<m>:cloud` → native Ollama Cloud endpoint
> `https://ollama.com`; `ollama.cloud/<m>` → OpenAI-compatible endpoint
> `https://ollama.com/v1`; a plain `ollama/<m>` (no `:cloud`) → your **local**
> Ollama daemon (`http://localhost:11434/v1`), which needs a downloaded model.

---

## Getting Started (Local)

Prefer an entirely local setup with no cloud account? First, install
python3-venv such as with `sudo apt install python3-venv`, then run the install
script. Mantis comes with automated configuration and launcher tools
(`mantis-configure` and `mantis-launch`):

```bash
cd reference && ./install.sh

# Start a local Ollama daemon and pull a real, downloadable DeepSeek model.
# (Local Ollama needs no cloud credentials.)
ollama serve &
ollama pull deepseek-v4-flash      # local (non-cloud) build of the default model

# Fast Configuration & Capability Auto-Detection
python3 scripts/configure.py --auto --model ollama/deepseek-v4-flash

# Fast Preflight Validation + Live Reachability Probe
python3 scripts/configure.py --test --probe

# Launch Vulnerability Review Campaign (file or repository)
./run.sh path/to/code              # a file or a directory

# (Optional) Research Graph Synthesis for a Specific Objective
./run.sh path/to/code --objective "Audit for Server-Side Request Forgery and SSRF in webhook handlers"
```


Mantis is roughly designed to:

- Review history of a codebase to look for historical vulnerabilities we do not
  wish to repeat
- Build a semantic index (and/or summaries) of the codebase for efficient code
  navigation
- Automatically generate a threat model
- Build up a set of hypotheses for individual agents to research (or simply
  "scan every file")
- Execute on those research plans
- Deduplicate existing findings
- Triage/critique findings to combat hallucination and statically verify
  production viability
- Reproduce the vulnerability to varying degrees, depending on available
  environments, anywhere from a static guess at what an exploit might look up up
  to a unit test or even spinning up a mock server to attempt to exploit
- Look across known vulnerabilities to attempt to build more impactful chains
- Patch discovered vulnerabilities, using an adversarial loop to verify the
  vulnerability is really fixed
- Calibrate all findings based on an established rubric to combat LLM inflation
  of severity (surfacing the most critical risks to humans instead of spewing
  thousands of "criticals")
- Reflect on each iteration of the loop to look for things we've learned in the
  now completed round of research
- Based on all of the collected learnings, threat model, and knowledge base, use
  the `/mantis-advise` skill to develop code more securely and ensure that
  during development you do not repeat prior mistakes or trigger edge cases in
  code that were previously protected by some guard that was removed

Additionally, the ADK reference harness shows some of the neat ways in which we
can build agentic workflows. Specifically, `reference/workflow.json` shows how a
deep review is done, and research graph synthesis
(`./run.sh target --objective "..."`) allows generating custom graph topologies
on the fly. This is a very powerful construct because you can use this to build
any kind of agentic deep dive security review you might imagine.

Mantis is intended to be a starting point rather than a rigid set of
instructions. You should adapt, tune, and extend this harness to fit your
organization's specific software or hardware stack. We provide an ADK-based
reference harness which works out of the box, but any competent coding agent
should be able to convert this to use your framework of choice.

The Mantis skills can be adapted to
[specialized domains](README_AGENTS.md#adaptability--specialized-domains) (such
as Hardware/RTL, Infrastructure as Code, ML pipelines, or compiled firmware).

We strongly recommend using AI to iterate on these skills and using your
internal documentation, coding standards, and build systems to augment the
threat models you use for scanning. We also strongly recommend adapting risk
calibration to your environment and risk tolerance.

Above all, while orchestrated vulnerability discovery is incredibly powerful and
useful, it is even more important to use this in a suitably isolated environment
to prevent impacting production systems. The ADK reference harness provides some
example sandboxes for testing, but if you use the most advanced frontier models
you should go beyond this and additionally set up an additional sandboxing layer
that itself contains very strong monitoring to look for escape attempts.

## Disclaimers

This is not an officially supported Google product. This project is not eligible
for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

This project is intended for demonstration purposes only. It is not intended for
use in a production environment.
