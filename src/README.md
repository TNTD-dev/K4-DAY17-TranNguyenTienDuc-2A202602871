# Student Scaffold

This `src/` folder is the student version of the lab.

- It keeps the same high-level structure
- The Python files are intentionally incomplete and contain pseudocode / TODOs
- The benchmark structure should include: standard benchmark + long-context stress benchmark
- The runtime should support these providers: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`

Suggested flow:

1. Start with `config.py`
2. Implement `memory_store.py`
3. Finish `agent_baseline.py`
4. Finish `agent_advanced.py`
5. Implement `benchmark.py`
6. Make `test_agents.py` pass

Datasets are available at the repo root in `data/`.

## Configuration (Step 3)

`load_config()` resolves paths from the repo root, creates `state/`, and reads
the root `.env` file. Existing process environment variables take precedence.
It never constructs an SDK model, so configuration works without API keys.
See `.env.example` at the repo root for optional live-mode settings.

- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`: main model (defaults: OpenAI,
  `gpt-4o-mini`, temperature `0`). Model names can be overridden for live use.
- `COMPACT_THRESHOLD_TOKENS=1000`, `COMPACT_KEEP_MESSAGES=4`: initial compact
  settings, to be validated against the long-context benchmark.
- `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_TEMPERATURE`, `JUDGE_API_KEY`,
  `JUDGE_BASE_URL`: optional independent judge settings. With the same provider,
  omitted values inherit from the main model. A different provider uses its own
  defaults and credentials.
- Keys: `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`),
  `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `CUSTOM_API_KEY`, `OLLAMA_API_KEY`.
- Endpoints: `<PROVIDER>_BASE_URL`; Ollama defaults to `http://localhost:11434`.
  The custom provider requires a URL when constructing a live model.
- `LLM_API_KEY` and `LLM_BASE_URL` override provider-specific key/URL settings.

Provider SDKs are imported only by `build_chat_model()` when live mode needs
them. Cloud model construction requires a key; offline configuration does not.
Configuration/provider checks run separately from the unfinished memory tests:

```bash
python -m pytest src/test_config_provider.py -v
```

## Memory layer (Step 4)

`UserProfileStore` stores UTF-8 Markdown at `state/profiles/<user>/User.md`.
Safe IDs keep their readable directory name; other IDs use a sanitized name
plus a hash, in a separate namespace. Resolved paths must stay inside the store.
Writes replace the file atomically. `edit_text()` replaces one occurrence;
`facts()` and `upsert_fact()` / `upsert_facts()` update fields while preserving
unrelated Markdown notes. Repeated identical facts do not rewrite the file.

The deterministic Vietnamese extractor recognizes explicit profile assertions,
current-location/job corrections, favorites, pets, technical interests and
response style. It skips questions, hypothetical/quoted claims and identifiable
temporary requests. These conservative regexes are a lab heuristic, not a
general Vietnamese entity extraction model.

`CompactMemoryManager` keeps state per thread. Its trigger includes both the
existing summary and messages. Once above the threshold, older messages merge
into a bounded extractive summary and the latest `keep_messages` messages remain
verbatim. Summary size is capped at 25% of the trigger budget. Compaction can
repeat without accumulating nested transcripts. Context snapshots are copies.

The threshold is a trigger, not a hard context limit: an oversized recent
message remains intact. Extractive summaries can lose details; durable profile
facts must be stored separately in `User.md`. No LLM calls are used by this layer.
The agent integration tests are still unfinished until the later lab steps.

```bash
python -m pytest src/test_memory_store.py src/test_config_provider.py -v
```

## Baseline Agent (Step 5)

`BaselineAgent.reply(user_id, thread_id, message)` returns `answer`,
`agent_tokens`, and `prompt_tokens` for that turn. `token_usage(thread_id)` and
`prompt_token_usage(thread_id)` return cumulative counters. Unknown threads
return zero. The prompt estimate includes the system prompt, full thread
history and current user message; output tokens count only the generated answer.

Offline mode extracts facts from user messages in the current thread and uses
the shared deterministic responder in `offline_response.py`. It never accesses
`User.md`, and the same user starting a new thread loses all previous facts.
Corrections within a thread replace earlier values. No history is trimmed and
`compaction_count()` always returns zero. A thread ID cannot be shared by
different users.

Use `force_offline=True` for repeatable benchmark/tests. Otherwise, missing
cloud credentials select offline mode. Configured cloud models, Ollama, or a
custom endpoint can use `create_agent` with an `InMemorySaver` checkpointer.
Only the current input is sent to the graph; its checkpointer supplies the
thread's prior messages. Live token metadata is used when provided, with a
character-based fallback. Live failures propagate instead of silently switching
to offline. The live path is tested with fake models, not real API requests.

```bash
python -m pytest src/test_baseline_agent.py src/test_memory_store.py src/test_config_provider.py -v
```

## Advanced Agent (Step 6)

`AdvancedAgent` uses the same reply result and cumulative-token interface as
Baseline. Every turn extracts explicit user facts, updates `User.md`, appends
the input to compact memory, builds context, generates an answer and appends the
answer. Profiles are user-scoped; thread histories and counters are thread-scoped.
The profile persists across new threads and process restarts. Unknown recall
questions do not create a profile file.

Prompt estimates include the full system prompt, profile, summary and recent
messages, including the current user input. Current profile facts override stale
summary/history mentions. The shared offline responder ensures the same facts
produce the same output for either agent. Partial style updates preserve other
constraints, explicit changes replace the corresponding constraint, and response
formatting honors supported bullet counts (1-6). General domain answering remains
outside this deterministic offline simulation.

The optional live graph uses an `InMemorySaver`, a dynamic profile/summary prompt
and middleware that replaces old checkpoint messages with the actual compact
context before each turn's first model call. Tool calls/results within that turn
are preserved. Profile tools obtain the user ID from injected runtime context;
the write tool only accepts a fact extracted from the current user message.
Every new model call, including tool requests, contributes to usage. Provider
metadata is preferred; text and tool arguments are estimated when it is absent.
Actual tool-schema/protocol overhead is not covered by the heuristic estimate.
Live API calls have not been validated against external services.

Step 6 integration checks cover profile persistence in a separate Python process,
cross-thread/user isolation, corrections, summary precedence, style, cumulative
accounting, real compact context in the graph, tool scoping and both supplied
datasets. The four required lab tests in `test_agents.py` are completed in Step 8.

```bash
python -m pytest src/test_advanced_agent.py -v
python -m pytest src/test_config_provider.py src/test_memory_store.py src/test_baseline_agent.py src/test_advanced_agent.py -v
```

## Benchmark (Step 7)

From the repository root, activate `.venv` and run:

```bash
source .venv/bin/activate
python src/benchmark.py
python -m pytest src/test_benchmark.py -v
```

The command prints Standard Benchmark and Long-Context Stress Benchmark with
both agents and all six required metrics. It forces offline mode even when API
keys are configured. Both agents receive identical turns, questions and thread
IDs. Each conversation is followed immediately by recall in its own fresh thread,
before later conversations can change the user's facts. All recall questions for
that conversation share that new thread. Expected answers only enter scoring.

Agent and prompt token totals include both training and recall turns. They are
character-based estimates, not provider billing counts. Recall is averaged per
question: 0 for no expected substrings, 0.5 for partial matches and 1 for all.
Response quality is the mean fraction of expected substrings found per question
(0..1). Both scores use case-insensitive, Unicode-normalized matching. This
quality proxy measures fact coverage; it does not evaluate fluency, negation,
formatting or general reasoning and does not call a judge model.

Memory growth is final minus initial profile bytes, counting each unique user
once. Compactions include chat and recall threads. The runner expects fresh agent
instances and unused benchmark threads. `main()` resets only the dataset users'
`User.md` files inside its own suite namespace before each run; histories and
counters start fresh with new agents. Profiles remain visible after completion:

```text
state/benchmarks/standard/profiles/dungct/User.md
state/benchmarks/long_context/profiles/dungct_stress/User.md
```

Profiles outside the benchmark namespace are preserved. The benchmark checks
cover input validation, partial scores, matching input order and fresh threads,
per-turn accounting, existing memory and repeated users, stress compactions and
prompt savings, and identical output across reruns without live model calls.

## Required memory tests (Step 8)

`test_agents.py` implements all four tests from the lab guide. `make_config()`
sets all seven config fields, uses `tmp_path` for state, stub provider configs,
an 80-token compaction threshold and two recent messages. Every agent in these
tests uses `force_offline=True`; no API key or live model request is needed.

The tests check UTF-8 profile creation and persistent edits, actual context
compression with the newest messages preserved, same-thread recall by both
agents followed by cross-thread recall only by Advanced, and lower accumulated
prompt load on a long identical conversation. The last test also compares
Advanced against an instance with compaction disabled: prompt load falls while
output token totals and profile content remain identical, and both still recall
the supplied facts. Test profiles live in pytest's temporary directories and
do not modify existing profiles in repository `state/`.

```bash
python -m pytest src/test_agents.py -v
python -m pytest -q
```

## Benchmark verification (Step 9)

Measured results and the compact-disabled comparison are saved in
`results/benchmark_results.md`; raw metrics, per-turn stress prompt counts and
dataset/profile hashes are in `results/benchmark_results.json`. Two fresh-process
runs with different Python hash seeds produced identical output and profiles.
The stress comparison uses separate empty profile directories for each variant,
with the same dataset and evaluation rules.

To reproduce the compact-disabled run, override the threshold for one command,
then run again with the normal configuration:

```bash
COMPACT_THRESHOLD_TOKENS=1000000 python src/benchmark.py
python src/benchmark.py
```

These commands reset only benchmark-owned dataset profiles. Other profiles
are preserved. The environment override does not edit `.env` or config defaults.
The measured default threshold was 1000 with four recent messages. Disabling
compact raised Advanced's stress prompt total from 12,617 to 28,618 while its
recall, output tokens and final profile size stayed the same. Baseline used
27,257 prompt tokens. Full result analysis and the bonus evidence are documented
in Step 10 below.

## Result analysis and bonus (Step 10)

`results/memory_analysis.md` answers all four analysis questions, ties each
conclusion to measured metrics and implementation, and maps the evidence to
`Rubric.md`. The chosen bonus is conflict handling, already implemented through
fact upserts, deduplication, current-profile precedence and partial style merges.

`benchmark_conflicts.py` evaluates this behavior against an experimental store
that keeps the first location/profession while leaving all other behavior intact.
Each variant runs offline in a separate temporary directory on unchanged input.
The optional JSON output includes all six metrics, recall answers, final profiles
and two extraction regression probes. Step 11 added historical-clause and
third-party-subject guards; both probes now return the intended updates, with
regression tests. General temporal/coreference parsing remains unsupported.

```bash
python src/benchmark_conflicts.py --output results/conflict_results.json
python -m pytest src/test_benchmark_conflicts.py -v
```

The control's recall is 64.3% on Standard and 66.7% on Stress, versus 100% with
current conflict handling. Its main benefit is recall; token savings are not
guaranteed. The comparison is isolated from normal benchmark and user profiles.
The complete suite passed 138 tests after the Step 11 regression fixes. Tested
dependency versions are pinned in `requirements.txt`; GitHub CI runs the full
suite, both benchmarks from clean state and the bonus comparison. The learner
submits the final repository URL on VLearn.
