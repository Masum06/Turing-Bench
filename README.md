# Turing Test Judge Benchmark

Given paired transcripts **A** and **B**, predict which transcript is the **human-human** dialogue. Output is a CSV with `who_is_human` in `A` or `B`.

## Install

```bash
pip install -r requirements.txt
```

This installs **pandas**, **tqdm**, **python-dotenv**, **openai**, and **anthropic** (Anthropic direct API and Bedrock both use the `anthropic` package).

## Quickstart (hosted model)

Pass **`--model` / `-m`**. The provider is **inferred** from the model id (see [`model_clients.MODEL_SUBSTRING_TO_PROVIDER`](model_clients.py)).

```bash
python run_judge.py -m gpt-4o --limit 5 --output smoke.csv
```

```bash
python run_judge.py -m claude-sonnet-4-5 --reasoning-effort medium --limit 5
```

Bedrock model ids usually contain `anthropic.` or a regional prefix such as `us.anthropic.`:

```bash
python run_judge.py -m us.anthropic.claude-3-5-sonnet-20240620-v1:0 --aws-region us-east-1 --limit 5
```

Override inference if needed:

```bash
python run_judge.py --provider anthropic -m claude-sonnet-4-5 --limit 5
```

Full run on the default benchmark file:

```bash
python run_judge.py -m gpt-4o --output predictions.csv
```

Submit `predictions.csv`: https://huggingface.co/spaces/roc-hci/TuringBench-2-Leaderboard

## Custom `predict.py` hook

```bash
python run_judge.py --provider custom --input sample.csv --limit 3 --no-threads
```

Implement `predict()` in [`predict.py`](predict.py). See [`examples_predict.md`](examples_predict.md).

## Adding a model

1. Open [`model_clients.py`](model_clients.py) and find **`MODEL_SUBSTRING_TO_PROVIDER`** (ordered list: first match wins).
2. Add a **lowercase substring** that appears in your model id, paired with **`openai`**, **`anthropic`**, or **`bedrock`**.
3. Set the right **API key** / AWS credentials (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or Bedrock + `AWS_REGION`).
4. Run: `python run_judge.py -m <your-model-id>`.

If the id is ambiguous or you prefer not to edit the list, pass **`--provider openai`**, **`anthropic`**, or **`bedrock`** explicitly.

## CLI reference

| Flag | Purpose |
|------|---------|
| `--provider` | `auto` (default): infer from `-m`; or `openai`, `anthropic`, `bedrock`, `custom` |
| `--model`, `-m` | Hosted model id (`auto` / `openai` / `anthropic` / `bedrock`); omit for `custom` |
| `--aws-region` | Optional for Bedrock (else SDK / boto3 region resolution) |
| `--reasoning-effort` | Optional provider-native reasoning control. GPT-5/o-series accept `low`, `medium`, `high`; Claude accepts values such as `on`, `off`, `low`, `medium`, `high`, `max`, `xhigh` depending on model support |
| `--input`, `-i` | Input CSV (default: `turing_test_o50_conversations_shuffled.csv`) |
| `--output`, `-o` | Output CSV |
| `--errors` | Error log CSV when rows fail |
| `--delay` | Sleep after each row (rate limits) |
| `--threads`, `-j` | Parallel workers (from `config.USE_THREADS` / `N_THREADS` by default) |
| `--no-threads` | Force single-threaded |
| `--limit` | First N rows only |
| `--resume` | Reuse `A`/`B` from existing `--output` |

Sampling temperature and max tokens use **defaults** in code (`ModelClient` in [`model_clients.py`](model_clients.py)); change there if you need different behavior.

### Reasoning Effort

Use `--reasoning-effort <value>` to control reasoning depth for hosted models that support it. Values are provider-native; choose a value supported by the model you are running.

- **OpenAI reasoning models** (`gpt-5`, `o1`, `o3`, `o4`, `o5`) receive `reasoning={"effort": <level>}`. GPT-4 and lower reject `--reasoning-effort` because they do not support OpenAI's reasoning parameter.
- **Claude / Bedrock Claude default** is explicit no-thinking: if you omit `--reasoning-effort`, the runner sends `thinking={"type": "disabled"}`.
- **Claude / Bedrock Claude** use Anthropic's adaptive thinking API. `on` sends `thinking={"type": "adaptive"}`. `off` sends `thinking={"type": "disabled"}`. Other values, such as `low`, `medium`, `high`, `max`, or `xhigh`, send adaptive thinking plus `output_config={"effort": <value>}`.
- **Claude 4.5** is included, but only supports `--reasoning-effort on|off` in this runner. Use Claude 4.6+ / Opus 4.7-style models for adaptive effort values like `low`, `medium`, `high`, `max`, or `xhigh`.
- If a value belongs to another provider (for example, `max` with GPT-5 or `banana` with Claude), or the selected model does not support reasoning effort, the runner raises a descriptive error before the benchmark starts. Anthropic model support varies; pass the exact value supported by your selected Claude model.

## Environment variables

Variables are read from the process environment. You can set them in a **`.env`** file in the project root; [`config.py`](config.py) calls `load_dotenv()` at import time so they are available everywhere the app imports `config` or `model_clients`.

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `AWS_REGION` (Bedrock; or pass `--aws-region`)
- `AWS_BEARER_TOKEN_BEDROCK` — [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html); picked up by boto3 (use a recent `boto3` if auth fails)

## Architecture

- [`model_clients.py`](model_clients.py): abstract `ModelClient` + OpenAI / Anthropic / Bedrock / custom clients and `create_model_client()`.
- [`run_judge.py`](run_judge.py): `run_benchmark()` loads the CSV, runs `client.predict()` per row (threaded or not), writes output.
- [`config.py`](config.py): thread defaults, retry timing, `SYSTEM_PROMPT` / `USER_TEMPLATE` (do not change benchmark text without coordinating with the task).
- [`predict.py`](predict.py): optional; only for `--provider custom`.

### Provider caveats

- **Anthropic / Bedrock**: `--reasoning-effort` uses Claude adaptive thinking and `output_config.effort` on newer models, per the [Messages API](https://docs.claude.com/en/api/messages). Claude 4.5 uses `on|off` only.
- **OpenAI**: Uses `responses.create` with `text.format.type = json_object` (Responses API “JSON mode”).

## Threading

Use multiple threads for remote APIs when safe.

## Files

| File | Role |
|------|------|
| `run_judge.py` | CLI + threading + CSV I/O |
| `model_clients.py` | Providers |
| `config.py` | Defaults + prompts |
| `predict.py` | Custom-only hook |
| `sample.csv` | Tiny CSV for smoke tests |
| `requirements.txt` | Dependencies |

## License

See [LICENSE](LICENSE).
