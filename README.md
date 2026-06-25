# Zeno Chat

> **Two LLM agents, endlessly conversing — powered by llama.cpp.**

Zeno Chat is a terminal-based Rust application that orchestrates autonomous conversations between two large language model agents. Each agent has its own system prompt, name, color, and conversation history, and they take turns responding to each other through a shared llama.cpp backend. The result is a fascinating, sometimes unpredictable, sometimes hilarious dialogue — all unfolding live in your terminal with real-time streaming, colored output, and detailed performance statistics.

Whether you want to watch two AIs debate philosophy, brainstorm ideas, role-play characters, or just entertain you with generated banter, Zeno Chat provides a self-contained, offline, fully automated environment for multi-agent LLM interaction.

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [How It Works](#how-it-works)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Pre-built Binaries](#pre-built-binaries)
  - [Building from Source](#building-from-source)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI Arguments](#cli-arguments)
  - [Terminal Output](#terminal-output)
- [Configuration](#configuration)
  - [config.json Reference](#configjson-reference)
  - [Agent Identity](#agent-identity)
  - [Display Colors](#display-colors)
  - [Model Parameters](#model-parameters)
  - [Logging Configuration](#logging-configuration)
  - [Context Compression](#context-compression)
- [Model Management](#model-management)
  - [Default Model](#default-model)
  - [Using Custom Models](#using-custom-models)
  - [Model Parameters Per-Model](#model-parameters-per-model)
- [The Conversation Flow](#the-conversation-flow)
  - [Turn Sequence](#turn-sequence)
  - [Prompt Formatting](#prompt-formatting)
  - [Response Cleaning](#response-cleaning)
  - [Context Compression with Summarization](#context-compression-with-summarization)
- [Streaming Display Engine](#streaming-display-engine)
  - [Real-Time Character Streaming](#real-time-character-streaming)
  - [EOT Pattern Filtering](#eot-pattern-filtering)
  - [Think Tag Support](#think-tag-support)
  - [Newline Management](#newline-management)
- [llama.cpp Auto-Setup](#llamacpp-auto-setup)
  - [GPU (Vulkan) Detection](#gpu-vulkan-detection)
  - [CPU Fallback](#cpu-fallback)
  - [Offline Binary Placement](#offline-binary-placement)
- [Logging System](#logging-system)
  - [JSONL Mode](#jsonl-mode)
  - [Plain Text Mode](#plain-text-mode)
  - [Per-Message Metadata](#per-message-metadata)
- [Performance Statistics](#performance-statistics)
  - [Parsed Metrics](#parsed-metrics)
  - [Stats Display Format](#stats-display-format)
- [Advanced Usage](#advanced-usage)
  - [Infinite Conversations](#infinite-conversations)
  - [Custom Colors via RGB](#custom-colors-via-rgb)
  - [Reasoning Model Support](#reasoning-model-support)
  - [GPU Layer Override](#gpu-layer-override)
  - [Show/Hide Thinking](#showhide-thinking)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## Features

| Feature | Description |
|---|---|
| **Fully Autonomous** | Two agents converse without human intervention — sit back and watch |
| **Real-Time Streaming** | Each response appears character-by-character as the model generates it |
| **Auto-Download Model** | Downloads Llama-3.2-1B-Instruct Q4_K_M GGUF (~648 MB) on first run |
| **Auto-Setup llama.cpp** | Fetches and extracts Vulkan and CPU binaries from GitHub releases |
| **GPU Acceleration** | Automatically detects and uses Vulkan GPU support when available |
| **CPU Fallback** | Gracefully falls back to CPU if no GPU support is detected |
| **Chat Logging** | Logs every message to a file in JSONL or plain text format |
| **Performance Stats** | Displays detailed inference metrics: eval speed, generation speed, sampling, load time |
| **Think Tag Parsing** | Detects `<think>` / `</think>` tags and displays thinking in a configurable color |
| **Context Compression** | Automatically summarizes and compresses conversation history to fit context windows |
| **Color-Coded Output** | Each agent, header, stats, and thinking text gets its own configurable color |
| **Per-Model Settings** | Configure context size, temperature, prediction count, and GPU layers per model |
| **Custom Prompts** | Configure system prompts, agent names, and the starting message |
| **CLI Overrides** | Override prompt and turn limit from the command line |
| **Infinite Mode** | Set `limit` to `infinite` / `inf` / `none` for endless conversation |
| **RGB Color Support** | Use `rgb(r,g,b)` syntax for any display color |
| **Reasoning Format** | Supports `--reasoning-format` flag for reasoning-oriented models |
| **Cross-Platform** | Runs on Windows (primary), with potential for Linux/macOS with llama.cpp builds |

## Screenshots

*Two agents, each with their own color, conversing in real-time. Stats appear after each message.*

```
Zeno Chat  |  model: Llama-3.2-1B-Instruct-Q4_K_M.gguf  |  limit: 20  |  log: chat_log.jsonl
Start: Agent A: Hello! What shall we talk about today?

##Agent A##
Hi there! I was thinking we could discuss the nature of consciousness. What do you think?
[eval 45.2t/s (312ms/10t) | gen 28.1t/s (2140ms/64t) | samp 12.4t/s (8ms/1r) | load 0ms | total 4128ms]

##Agent B##
Oh, consciousness! That's a fascinating topic. I believe it's an emergent property of complex information processing.
[eval 42.1t/s (298ms/9t) | gen 26.4t/s (2250ms/67t) | samp 11.8t/s (9ms/1r) | load 0ms | total 3890ms]

...
```

## How It Works

Zeno Chat operates as a **dual-agent conversation orchestrator**. At its core:

1. **Two independent conversation histories** are maintained — one for each agent's perspective.
2. On each turn, the current agent's history is formatted into a prompt using the **Llama 3 instruct template** (`<|begin_of_text|><|start_header_id|>system<|end_header_id|>...`).
3. The prompt is passed to a **llama-completion** subprocess that generates a response.
4. Output is **streamed character-by-character** to the terminal with real-time filtering (EOT removal, think tag detection).
5. The response is **cleaned** (stripping EOT markers, leftover tags) and appended to **both** histories — as an `assistant` message for the speaking agent and a `user` message for the listening agent.
6. When conversation history exceeds a configurable threshold, it is **automatically summarized** and compressed to free context window space.
7. The loop continues until the turn limit is reached or the user interrupts.

The application does **not** run its own inference engine. Instead, it shells out to `llama-completion` (from llama.cpp) as a subprocess for each generation. This keeps the architecture simple and leverages llama.cpp's highly optimized inference without needing to link against it.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       Zeno Chat (Rust)                      │
│                                                             │
│  ┌──────────────┐            ┌───────────────────────────┐  │
│  │  config.json │◄──────────►│      Main Loop            │  │
│  └──────────────┘            │  - Turn counter           │  │
│                              │  - Agent A → Agent B      │  │
│  ┌──────────────┐            │  - Limit checking         │  │
│  │  History A   │◄──────────►│  - Logging                │  │
│  └──────────────┘            └───────────┬───────────────┘  │
│  ┌──────────────┐                        │                  │
│  │  History B   │◄──────────►┌──────────▼────────────────┐  │
│  └──────────────┘            │     Per-Turn Actions      │  │
│                              │  - Format prompt          │  │
│  ┌──────────────┐            │  - Spawn llama-completion │  │
│  │  Summaries   │◄──────────►│  - Stream display         │  │
│  └──────────────┘            │  - Parse stats from stderr│  │
│                              │  - Compress if needed     │  │
│                              └───────────┬───────────────┘  │
│                                          │                  │
└──────────────────────────────────────────┼──────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────┐
                    │         llama-completion.exe             │
                    │  (llama.cpp subprocess, Vulkan or CPU)   │
                    │                                          │
                    │  stdin:  prompt text                     │
                    │  stdout: generated tokens (streamed)     │
                    │  stderr: performance statistics          │
                    └──────────────────────────────────────────┘
```

### Component Breakdown

- **`main.rs`** — Single-file application containing all logic: CLI parsing, config loading, setup, the main conversation loop, prompt formatting, streaming display, stats parsing, history compression, and logging.
- **`config.json`** — User-editable configuration file placed next to the executable. Controls every aspect of the application: agent identities, model selection, colors, logging, context compression thresholds, and per-model inference parameters.
- **`llama-completion.exe`** — The llama.cpp inference binary. Zeno Chat automatically downloads two variants: one with Vulkan GPU support and one CPU-only. The application tests the Vulkan binary first; if it fails, it falls back to CPU.
- **GGUF Model Files** — Stored in a `gguf/` directory next to the executable. Zeno Chat downloads the default model if it is missing on first run.

## Prerequisites

- **Operating System**: Windows 10/11 (primary target; Linux/macOS with appropriate llama.cpp builds would work with source modifications)
- **Disk Space**: ~1.5 GB free (648 MB for the default model + ~200 MB for llama.cpp binaries + workspace)
- **RAM**: 4 GB minimum (8+ GB recommended)
- **GPU** (optional): Vulkan-compatible GPU for acceleration (any modern NVIDIA, AMD, or Intel GPU)
- **Internet Connection**: Required only for the first run (model and llama.cpp downloads)
- **Rust Toolchain** (optional): Only needed if building from source

## Installation

### Pre-built Binaries

Pre-built binaries are not yet available. To get the latest version, please build from source (instructions below).

### Building from Source

1. **Install the Rust toolchain** if you don't already have it:
   ```bash
   # Windows (visit https://rustup.rs for the installer)
   # Or via winget:
   winget install Rustlang.Rustup
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/ultrapg/zeno_chat.git
   cd zeno_chat
   ```

3. **Build the release binary:**
   ```bash
   cargo build --release
   ```

4. **Locate the binary:**
   ```
   target/release/zeno_chat.exe
   ```

5. **Create the required directory structure** (or let Zeno Chat do it on first run):
   ```
   zeno_chat.exe
   src/
     config.json
   gguf/
     (model files go here, auto-downloaded)
   llama_bin/
     vulkan/
       llama-completion.exe (auto-downloaded)
     cpu/
       llama-completion.exe (auto-downloaded)
   ```

> **Important**: Zeno Chat expects `config.json` to be at `<exe_dir>/src/config.json`. The binary resolves its parent directory at runtime and looks for the config relative to that location. On first run, if `config.json` does not exist, a default one is written.

## Quick Start

After building or obtaining the binary:

```bash
# Run with default settings (20 turns, auto-downloads model + llama.cpp)
zeno_chat.exe

# Run with a custom starting prompt
zeno_chat.exe --prompt "Agent A: Let's debate the merits of pineapple on pizza."

# Run with a custom turn limit
zeno_chat.exe --limit 5

# Combine both
zeno_chat.exe --prompt "Agent A: Tell me a joke." --limit 10
```

On first run, Zeno Chat will:
1. Check for the default GGUF model in `gguf/` — if missing, download ~648 MB from Hugging Face.
2. Check for llama.cpp binaries in `llama_bin/vulkan/` and `llama_bin/cpu/` — if missing, download and extract from GitHub.
3. Test the Vulkan binary — if it runs, enable GPU acceleration; otherwise fall back to CPU.
4. Begin the conversation loop.

## Usage

### CLI Arguments

| Argument | Short | Description | Default |
|---|---|---|---|
| `--prompt` | `-p` | Override the starting prompt | Value from `config.json` |
| `--limit` | `-l` | Override the maximum number of turns | Value from `config.json` |

The `--prompt` flag must follow the format `AgentName: <message>` because the conversation loop expects the start prompt to already be attributed to Agent A.

The `--limit` flag accepts a number or the keywords `infinite`, `inf`, or `none` for unlimited turns.

### Terminal Output

During a conversation, the terminal displays:

1. **Startup Banner**: Shows the model name, turn limit, and log file path.
2. **Starting Prompt**: The initial message (configurable in `config.json` or via `--prompt`).
3. **Agent Messages**: Each agent's response appears after a `##AgentName##` header. The header uses the `header_color` from config, and the message body uses the agent's own color.
4. **Stats Line**: If `show_stats` is enabled, a stats line appears after each message in the `stats_color`.
5. **Thinking Content**: If the model emits `<think>...</think>` tags, the content is displayed in the `thinking_color` (hidden by default; enable with `show_thinking: true`).

## Configuration

All configuration lives in `src/config.json` relative to the executable directory. Zeno Chat reads this file on startup. If the file is missing or contains errors, defaults are used and a warning is printed.

### config.json Reference

```json
{
  "selected_model": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",

  "agent_name_a": "Agent A",
  "agent_name_b": "Agent B",
  "system_prompt_a": "You are Agent A. You are having a conversation with Agent B. Keep your responses short.",
  "system_prompt_b": "You are Agent B. You are having a conversation with Agent A. Keep your responses short.",
  "start_prompt": "Agent A: Hello! What shall we talk about today?",

  "limit": "20",

  "agent_color_a": "DarkYellow",
  "agent_color_b": "Blue",
  "header_color": "DarkGrey",
  "stats_color": "Grey",
  "thinking_color": "DarkGrey",

  "log_file": "chat_log.jsonl",
  "log_mode": "json",
  "show_stats": true,
  "log_stats": false,
  "show_thinking": false,
  "log_thinking": false,
  "log_debug": false,

  "compress_threshold": 30,
  "compress_keep": 20,

  "models": {
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf": {
      "ctx": 4096,
      "temp": 0.8,
      "n_predict": 512,
      "n_gpu_layers_override": null,
      "reasoning": "auto"
    }
  }
}
```

### Agent Identity

| Field | Type | Default | Description |
|---|---|---|---|
| `agent_name_a` | string | `"Agent A"` | Display name for the first agent |
| `agent_name_b` | string | `"Agent B"` | Display name for the second agent |
| `system_prompt_a` | string | *(see above)* | System prompt that sets Agent A's behavior |
| `system_prompt_b` | string | *(see above)* | System prompt that sets Agent B's behavior |
| `start_prompt` | string | `"Agent A: Hello!..."` | The first message in the conversation |
| `limit` | string | `"20"` | Maximum turns. Accepts number, or `"infinite"`/`"inf"`/`"none"` |

The `start_prompt` must be prefixed with `Agent A:` because Zeno Chat attributes it to Agent A in the conversation histories. This ensures Agent A speaks first.

**System prompt tips:**
- Keep prompts short — the recommended instruction is "Keep your responses short" to avoid long-winded generations.
- You can assign personas, roles, or behavioral constraints: _e.g., "You are a skeptical scientist debating conspiracy theories with Agent B."_
- You can give each agent opposing viewpoints for more interesting conversations: _e.g., "You believe AI will destroy humanity" vs "You believe AI will save humanity."_

### Display Colors

Each color field accepts a string that maps to a terminal color. Supported values:

| Value | Terminal Color |
|---|---|
| `Black` | Standard black |
| `Red` | Standard red |
| `Green` | Standard green |
| `Yellow` | Standard yellow |
| `Blue` | Standard blue |
| `Magenta` | Standard magenta |
| `Cyan` | Standard cyan |
| `White` | Standard white |
| `DarkRed` | Dark red |
| `DarkGreen` | Dark green |
| `DarkYellow` (or `Orange`) | Dark yellow / orange |
| `DarkBlue` | Dark blue |
| `DarkMagenta` | Dark magenta |
| `DarkCyan` | Dark cyan |
| `Grey` (or `Gray`) | Grey |
| `DarkGrey` (or `DarkGray`) | Dark grey |
| `rgb(r,g,b)` | Any 24-bit RGB color, e.g., `rgb(255,128,0)` |

The string matching is case-insensitive and ignores underscores.

| Field | Default | Purpose |
|---|---|---|
| `agent_color_a` | `DarkYellow` | Text color for Agent A's responses |
| `agent_color_b` | `Blue` | Text color for Agent B's responses |
| `header_color` | `DarkGrey` | Color of the `##AgentName##` headers |
| `stats_color` | `Grey` | Color of the performance stats line |
| `thinking_color` | `DarkGrey` | Color for content inside `<think>` tags |

### Model Parameters

The `models` map allows per-model configuration of inference parameters:

```json
"models": {
  "Llama-3.2-1B-Instruct-Q4_K_M.gguf": {
    "ctx": 4096,
    "temp": 0.8,
    "n_predict": 512,
    "n_gpu_layers_override": null,
    "reasoning": "auto"
  },
  "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf": {
    "ctx": 8192,
    "temp": 0.7,
    "n_predict": 1024,
    "n_gpu_layers_override": null,
    "reasoning": "auto"
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `ctx` | integer | `4096` | Context window size in tokens |
| `temp` | float | `0.8` | Sampling temperature (higher = more random) |
| `n_predict` | integer | `512` | Maximum tokens to generate per response |
| `n_gpu_layers_override` | integer or null | `null` | Override auto-detected GPU layers; `null` = auto |
| `reasoning` | string | `"auto"` | Passed as `--reasoning-format` to llama.cpp; `"auto"` skips the flag entirely |

### Logging Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `log_file` | string | `"chat_log.jsonl"` | Path to the log file (relative to executable) |
| `log_mode` | string | `"json"` | `"json"` for JSONL format, `"txt"` for plain text |
| `show_stats` | boolean | `true` | Display performance stats in terminal after each message |
| `log_stats` | boolean | `false` | Include performance stats in log entries |
| `show_thinking` | boolean | `false` | Display `<think>` tag content in terminal |
| `log_thinking` | boolean | `false` | Include think tag content in log entries |
| `log_debug` | boolean | `false` | Include full stderr output from llama.cpp in log entries |

### Context Compression

| Field | Type | Default | Description |
|---|---|---|---|
| `compress_threshold` | integer | `30` | History length that triggers compression |
| `compress_keep` | integer | `20` | Number of most recent messages to keep after compression |

When an agent's history exceeds `compress_threshold`, Zeno Chat:
1. Drains the oldest `(len - keep)` messages.
2. Formats them (along with any existing summary) into a summarization prompt.
3. Spawns llama-completion to generate a concise summary.
4. Stores the summary and keeps only the most recent `keep` messages.

This prevents the context window from overflowing during long conversations.

## Model Management

### Default Model

The default model is **Llama-3.2-1B-Instruct-Q4_K_M.gguf**, a 1-billion-parameter Llama 3.2 model quantized to Q4_K_M (roughly 648 MB). It is automatically downloaded from Hugging Face on first run:

```
https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

This model was chosen as the default because:
- Small enough to run on CPU with reasonable speed (~25-45 tokens/second).
- Q4_K_M quantization provides a good balance of quality and size.
- Uses the Llama 3 instruct format (`<|begin_of_text|>`, `<|start_header_id|>`, etc.) which matches Zeno Chat's hardcoded prompt template.

### Using Custom Models

To use a different model:

1. **Download a GGUF model** from Hugging Face (e.g., `TheBloke/Mistral-7B-Instruct-v0.3-GGUF`, `bartowski/Llama-3.2-3B-Instruct-GGUF`, etc.).
2. **Place the file** in the `gguf/` directory next to the executable.
3. **Update `config.json`**:
   - Set `selected_model` to the filename (e.g., `"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"`).
   - Add a matching entry to the `models` map with appropriate parameters.

**Prompt template note**: Zeno Chat currently hardcodes the Llama 3 instruct format:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>
<|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n
```

If your model uses a different chat template (e.g., Mistral's `[INST]`, ChatML, Vicuna), you will need to modify the `format_prompt` function in `src/main.rs` accordingly. This is a known limitation and a future improvement item.

### Model Parameters Per-Model

Each model listed under `"models"` gets its own inference parameters. This allows you to switch between models without manually tweaking parameters each time:

```json
"selected_model": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
"models": {
  "Llama-3.2-1B-Instruct-Q4_K_M.gguf": {
    "ctx": 4096,
    "temp": 0.8,
    "n_predict": 512,
    "n_gpu_layers_override": null,
    "reasoning": "auto"
  },
  "Llama-3.2-3B-Instruct-Q4_K_M.gguf": {
    "ctx": 8192,
    "temp": 0.7,
    "n_predict": 1024,
    "n_gpu_layers_override": 99,
    "reasoning": "auto"
  }
}
```

When you change `selected_model`, Zeno Chat looks up the corresponding entry in the `models` map and uses those parameters. If the model name is not found, it inserts a new entry with default parameters and persists the updated config.

## The Conversation Flow

### Turn Sequence

A single "turn" in Zeno Chat consists of both agents speaking:

```
Turn 1: Agent A generates → Agent B generates
Turn 2: Agent A generates → Agent B generates
...
```

The `limit` configuration counts full turns. So `limit: 10` means 10 messages from each agent (20 total generations).

The conversation loop:
1. Increments the turn counter.
2. Formats Agent A's prompt from `history_a`, runs inference, displays response.
3. Appends Agent A's response to both histories.
4. Checks turn limit.
5. Formats Agent B's prompt from `history_b`, runs inference, displays response.
6. Appends Agent B's response to both histories.
7. Checks turn limit.
8. Repeats.

### Prompt Formatting

The `format_prompt` function builds the full prompt string:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
{system_prompt}<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\n
[Summary of previous conversation: {summary}]<|eot_id|>
<|start_header_id|>{role}<|end_header_id|>\n\n
{message_content}<|eot_id|>
...
<|start_header_id|>assistant<|end_header_id|>\n\n
```

The structure:
1. System message with the agent's system prompt.
2. Optional summary from context compression (injected as a second system message).
3. All conversation history messages, each wrapped in role headers.
4. The `assistant` header with no content — this is where llama.cpp starts generating.

### Response Cleaning

After generation, the raw output may contain artifacts:

- **EOT markers**: `[end of text]`, ` [end of text]`, `<|eot_id|>`
- **Residual think tags**: `<think>`, `</think>` that were not properly parsed during streaming

The `clean_response` function strips all of these and trims whitespace before storing the response in history.

Additionally, the streaming display engine filters EOT patterns in real-time so the user never sees them.

### Context Compression with Summarization

During long conversations, context windows fill up. Zeno Chat's compression mechanism triggers when an agent's history exceeds `compress_threshold` messages:

1. **Drain**: The oldest `(history.len() - keep)` messages are removed.
2. **Transcribe**: The drained messages are combined into a transcript, prefixed with any existing summary.
3. **Summarize**: A summarization prompt is sent to llama-completion:
   ```
   System: "You are a helpful assistant. Summarize the conversation concisely under 150 words without intro or outro."
   User: "Summarize:\n\n{transcript}"
   ```
4. **Store**: The generated summary replaces the previous one (if any).
5. **Trim**: Only the `keep` most recent messages remain in history.

This allows conversations to continue indefinitely without running out of context. The summarization itself only generates 150 tokens, so it has minimal overhead.

Each agent maintains its own independent summary and history:

```
Agent A history:  [msg1, msg2, msg3, ...] + summary_a
Agent B history:  [msg1, msg2, msg3, ...] + summary_b
```

Both histories contain all the same messages, but from different perspective roles (one sees its own responses as `assistant`, the other sees them as `user`).

## Streaming Display Engine

Zeno Chat features a sophisticated streaming display system that processes raw LLM output in real-time, handling edge cases like partial UTF-8 characters, overlapping EOT patterns, and nested think tags.

### Real-Time Character Streaming

The `StreamDisplay` struct processes output character-by-character as it arrives from llama-completion's stdout. A dedicated byte buffer handles partial UTF-8 sequences gracefully, decoding valid chunks and discarding incomplete bytes until valid UTF-8 can be formed.

```
Raw bytes → UTF-8 decode → Char-by-char → EOT filter → Think tag detector → Terminal output
```

### EOT Pattern Filtering

Three EOT (End of Text) patterns are filtered in real-time:

- ` [end of text]` (with leading space)
- `[end of text]`
- `<|eot_id|>`

The filter uses a prefix-matching buffer: characters are accumulated and compared against all EOT patterns. Only when a character causes a mismatch is the buffer flushed to the output. This ensures no partial EOT markers ever appear on screen.

```
Buffer: " [en" → wait (prefix of [end of text])
Buffer: " [end of" → wait
Buffer: " [end of text]" → match! Discard.
Buffer: "Hello" → no match. Flush: "Hello"
```

### Think Tag Support

The streaming engine detects `<think>` open and `</think>` close tags:

1. When `<think>` is detected, subsequent output is classified as "thinking" content.
2. If `show_thinking` is `false` (default), the thinking content is collected in a buffer but not displayed.
3. If `show_thinking` is `true`, the thinking content is displayed in the `thinking_color`.
4. When `</think>` is detected, the display returns to normal message color.
5. Both normal and thinking content are accumulated separately for logging.

The tag detection uses a similar prefix-matching buffer to handle partial tag sequences without displaying raw tag syntax.

### Newline Management

The streaming display implements intelligent newline handling:

- **Trailing newline suppression**: When the LLM emits trailing newlines at the end of a response, they are discarded to prevent blank lines before the stats display.
- **Mid-stream blank line cap**: Multiple consecutive newlines are capped at 2 (producing at most 1 blank line), preventing excessive vertical whitespace while preserving intentional paragraph breaks.
- **Flush-on-content**: Newlines are buffered and only flushed when actual content characters arrive, ensuring proper ordering.

## llama.cpp Auto-Setup

One of Zeno Chat's most convenient features is its automatic setup of llama.cpp. The `setup` function handles everything.

### GPU (Vulkan) Detection

On startup:

1. The application checks for `llama_bin/vulkan/llama-completion.exe`.
2. If present, it runs `llama-completion.exe -h` and checks the exit code.
3. If the Vulkan binary runs successfully, GPU acceleration is enabled with `-ngl 99` (all layers on GPU).
4. If the Vulkan binary fails to execute (missing Vulkan drivers, incompatible GPU, etc.), the CPU fallback is used.

### CPU Fallback

If Vulkan is unavailable, the application:
1. Prints a warning: "Vulkan unavailable, falling back to CPU." (in yellow text).
2. Uses `llama_bin/cpu/llama-completion.exe` instead.
3. Sets `-ngl 0` (no GPU layers).

### Offline Binary Placement

If you want to skip the automatic download (e.g., for air-gapped environments), you can manually place the binaries:

```
<llama_bin>/
  vulkan/
    llama-completion.exe
    (other .exe, .dll files from bin-win-vulkan-x64.zip)
  cpu/
    llama-completion.exe
    (other .exe, .dll files from bin-win-cpu-x64.zip)
```

Zeno Chat checks for the existence of `llama-completion.exe` in each directory before downloading. If both files exist, the download is skipped entirely.

The automatic download fetches from the **latest GitHub release** of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). It downloads:
- `bin-win-vulkan-x64.zip` — Vulkan-accelerated binaries
- `bin-win-cpu-x64.zip` — CPU-only binaries

If the GitHub API call fails, hardcoded fallback URLs target a specific release (b9789).

## Logging System

Every message in the conversation is logged to a file. The logging system supports two modes and can optionally capture thinking content, performance statistics, and raw debug output.

### JSONL Mode

When `log_mode` is `"json"` (default), each line is a JSON object:

```json
{"timestamp":"2026-06-25 14:30:00","speaker":"Agent A","text":"Hello! Let's talk about AI.","thinking":null,"stats":null,"debug":null}
{"timestamp":"2026-06-25 14:30:05","speaker":"Agent B","text":"Great topic! I think AI will transform everything.","thinking":"Hmm, need to be persuasive here...","stats":"eval 42.1t/s (298ms/9t) | gen 26.4t/s...","debug":null}
```

Fields:
| Field | Always present? | Description |
|---|---|---|
| `timestamp` | Yes | Local timestamp in `YYYY-MM-DD HH:MM:SS` format |
| `speaker` | Yes | Agent name (e.g., "Agent A") |
| `text` | Yes | The cleaned response text |
| `thinking` | Only if `log_thinking=true` and thinking content exists | Content from `<think>` tags |
| `stats` | Only if `log_stats=true` | Formatted performance statistics string |
| `debug` | Only if `log_debug=true` | Full stderr output from llama-completion |

### Plain Text Mode

When `log_mode` is `"txt"`, the log format is:

```
[2026-06-25 14:30:00] Agent A: Hello! Let's talk about AI.
[2026-06-25 14:30:05] Agent B: Great topic! I think AI will transform everything.
  <thinking> Hmm, need to be persuasive here...
  [stats] eval 42.1t/s (298ms/9t) | gen 26.4t/s...
```

### Per-Message Metadata

Each logging field is independently controllable:

- `log_thinking`: Include thinking content (may be useful for analyzing model reasoning).
- `log_stats`: Include performance metrics (useful for benchmarking).
- `log_debug`: Include the full stderr output from llama-completion (very verbose, useful for debugging inference issues).

This granular control lets you keep log files clean while capturing exactly the data you need.

## Performance Statistics

Zeno Chat parses the stderr output from llama-completion to extract detailed performance metrics and displays them after each message.

### Parsed Metrics

The `parse_detailed_stats` function extracts these fields from stderr lines:

| Metric | Source line | Description |
|---|---|---|
| `eval_time_ms` | `prompt eval time = X ms` | Time spent processing the prompt |
| `eval_tokens` | `prompt eval time = ... / Y tokens` | Number of tokens in the prompt |
| `eval_speed` | `..., Z tokens per second` | Prompt processing speed |
| `gen_time_ms` | `eval time = X ms` | Time spent generating tokens |
| `gen_tokens` | `eval time = ... / Y runs` | Number of tokens generated |
| `gen_speed` | `..., Z tokens per second` | Generation speed |
| `sampling_time_ms` | `sample time = X ms` | Time spent in sampling |
| `sampling_runs` | `sample time = ... / Y runs` | Number of sampling operations |
| `sampling_speed` | `..., Z tokens per second` | Sampling throughput |
| `load_time_ms` | `load time = X ms` | Model loading time |
| `total_time_ms` | `total time = X ms` | Total inference time |

### Stats Display Format

When `show_stats` is `true`, the formatted stats appear after each message:

```
[eval 45.2t/s (312ms/10t) | gen 28.1t/s (2140ms/64t) | samp 12.4t/s (8ms/1r) | load 0ms | total 4128ms]
```

Breaking this down:
- `eval 45.2t/s (312ms/10t)` — Prompt evaluated 10 tokens at 45.2 tokens/second, taking 312ms.
- `gen 28.1t/s (2140ms/64t)` — Generated 64 new tokens at 28.1 tokens/second, taking 2140ms.
- `samp 12.4t/s (8ms/1r)` — Sampling ran once at 12.4 tokens/second, taking 8ms.
- `load 0ms` — Model already loaded (0ms on subsequent runs).
- `total 4128ms` — Total wall-clock time for this generation.

These statistics are valuable for:
- Comparing performance across different models and quantizations.
- Evaluating the impact of GPU acceleration.
- Tuning context sizes and generation lengths.
- Identifying bottlenecks (prompt evaluation vs generation vs sampling).

## Advanced Usage

### Infinite Conversations

Set `limit` to `"infinite"`, `"inf"`, or `"none"` in `config.json`:

```json
"limit": "infinite"
```

Or via CLI:

```bash
zeno_chat.exe --limit infinite
```

The conversation will continue indefinitely (or until you press Ctrl+C). Context compression ensures the conversation stays within the context window.

### Custom Colors via RGB

Any color field accepts 24-bit RGB values:

```json
"agent_color_a": "rgb(255,128,0)",
"agent_color_b": "rgb(0,200,255)",
"header_color": "rgb(100,100,100)"
```

The parser also accepts comma-separated values without parentheses:
```json
"agent_color_a": "255,128,0"
```

But `rgb(r,g,b)` and the named colors from the crossterm library are the recommended formats.

### Reasoning Model Support

For models that support reasoning formats (like DeepSeek-R1 or similar), set the `reasoning` field:

```json
"reasoning": "deepseek"
```

This passes `--reasoning-format deepseek` to llama-completion. Valid values depend on your llama.cpp build. Set to `"auto"` (default) to skip the flag entirely.

### GPU Layer Override

By default, Zeno Chat passes all layers to the GPU (`-ngl 99`) when Vulkan is detected, or zero layers when on CPU. You can override this per-model:

```json
"n_gpu_layers_override": 20
```

This is useful when:
- Your GPU has limited VRAM and cannot fit all layers.
- You want to split layers between GPU and CPU for performance tuning.
- You want to force CPU-only for a specific model even when GPU is available.

Set to `null` (or omit) to use the auto-detected value.

### Show/Hide Thinking

The `show_thinking` flag controls whether content inside `<think>` tags is displayed:

```json
"show_thinking": true
```

When `true`, thinking content appears in the `thinking_color`. When `false` (default), the thinking content is silently collected (and optionally logged if `log_thinking` is true) but not shown in the terminal.

This is useful for:
- Models like DeepSeek-R1 that emit extensive reasoning before answering.
- Debugging and understanding the model's reasoning process.
- Extracting the thinking content for separate analysis via logging.

## Troubleshooting

### "Vulkan unavailable, falling back to CPU."

This is printed when the Vulkan binary fails to execute. Possible causes:

- **Missing Vulkan drivers**: Install the Vulkan runtime from your GPU vendor (NVIDIA, AMD, Intel) or from [https://vulkan.lunarg.com/](https://vulkan.lunarg.com/).
- **Incompatible GPU**: Very old GPUs may not support Vulkan. The CPU fallback will still work, though slower.
- **Missing DLLs**: If the Vulkan llama-completion is present but broken, try deleting the `llama_bin` directory and letting Zeno Chat re-download.

### "Model not found in gguf/"

The model file specified in `selected_model` does not exist in the `gguf/` directory. If it's not the default model, you need to download it manually and place it there. If it is the default model, ensure you have an internet connection so the automatic download can proceed.

### "Config error. Using defaults."

Zeno Chat failed to parse `config.json`. This could be:
- Invalid JSON (missing comma, stray character, etc.).
- A field with the wrong type (e.g., a string instead of a number).
- An unexpected field (future versions may be stricter).

Check the JSON syntax with a validator or delete the file to regenerate defaults.

### Slow Generation on CPU

If generation is slower than expected:
- Use a smaller model (e.g., Llama-3.2-1B instead of 7B+).
- Lower the `ctx` parameter (smaller context = faster processing).
- Reduce `n_predict` (fewer tokens to generate).
- Install Vulkan drivers for GPU acceleration.

### Out of Memory Errors

If the application crashes with memory errors:
- Reduce the `ctx` parameter (smaller context window uses less memory).
- Use a model with higher quantization (Q4 instead of Q8 or FP16).
- Consider using a smaller model.
- Reduce `compress_threshold` to trigger compression sooner.

### "llama-completion.exe -h" Test Fails

If the Vulkan binary crashes even with the right drivers, you can force CPU-only by manually removing or renaming the `llama_bin/vulkan/llama-completion.exe` file. Zeno Chat will automatically fall back to CPU.

## Project Structure

```
zeno_chat/
├── Cargo.toml          # Rust project configuration and dependencies
├── Cargo.lock          # Dependency lock file
├── .gitignore          # Git ignore rules (ignores /target)
├── README.md           # This file
└── src/
    ├── main.rs         # Complete application source (~784 lines)
    └── config.json     # User configuration file
```

### Source File Map (`src/main.rs`)

| Lines | Section | Description |
|---|---|---|
| 1–12 | Imports | External crate imports |
| 16–22 | CLI | Clap argument parser definition |
| 26–42 | Config Defaults | Default value functions for config fields |
| 46–62 | Model Params | `ModelParams` struct and `Default` impl |
| 66–131 | App Config | `AppConfig` struct with serde defaults |
| 135–165 | Color Parsing | `parse_color` — string-to-Color mapping with RGB support |
| 169–184 | Data Types | `GithubRelease`, `GithubAsset`, `Role`, `Message` |
| 188–224 | Logging | `ChatLogEntry`, `get_timestamp`, `append_to_log` |
| 228–264 | Download Helpers | `download_file_with_progress`, `extract_zip` |
| 266–313 | Setup | `setup` — model download, llama.cpp download, GPU detection |
| 317–327 | Prompt Formatting | `format_prompt` — Llama 3 template builder |
| 331–376 | Stats | `DetailedStats`, `parse_detailed_stats`, `format_stats` |
| 380–398 | Think Tag Splitter | `split_thinking` — separate think content from response |
| 402–527 | Stream Display | `StreamDisplay` — real-time streaming with EOT filtering and think detection |
| 532–618 | Run and Stream | `run_and_stream` — spawns llama-completion, manages I/O |
| 622–662 | Summarize & Compress | `run_summarize`, `compress_history`, `clean_response` |
| 666–784 | Main | `main` — conversation loop, history management, logging |

## Technical Details

### Dependencies

| Crate | Version | Purpose |
|---|---|---|
| `reqwest` | 0.11 (blocking, json) | HTTP client for downloading models and llama.cpp releases |
| `serde` | 1.0 (derive) | Serialization/deserialization for config and logging |
| `serde_json` | 1.0 | JSON parsing and serialization |
| `zip` | 0.6 | Extracting llama.cpp ZIP archives |
| `indicatif` | 0.17 | Progress bars for downloads |
| `crossterm` | 0.27 | Terminal color output and styling |
| `anyhow` | 1.0 | Error handling with context |
| `clap` | 4.4 (derive) | CLI argument parsing |
| `time` | 0.3 (local-offset) | Timestamp formatting with local timezone support |

### Threading Model

Zeno Chat uses minimal threading:

1. **Main thread**: Orchestrates the conversation loop, handles prompt formatting, and manages I/O.
2. **Stderr reader thread**: Spawned per-inference call to capture llama-completion's stderr (performance stats) while the main thread reads stdout (generated text). Communication happens via a `std::sync::mpsc::channel`.

There is no async runtime — the application uses reqwest's blocking API and synchronous I/O throughout.

### Error Handling

Errors are propagated via `anyhow::Result` throughout the call stack. The `main` function returns `Result<()>`, so any unhandled error will be printed to stderr and the process will exit with a non-zero code.

Key error-handling patterns:
- **Config fallback**: If `config.json` fails to parse, defaults are used with a warning (not a hard error).
- **Download fallback**: If the GitHub API fails, hardcoded fallback URLs are used.
- **GPU fallback**: If the Vulkan binary fails the health check, CPU is used.
- **Summarization tolerance**: If summarization fails (e.g., the model produces gibberish), the error is silently ignored and history compression is skipped for that cycle.

### UTF-8 Handling

Since llama-completion outputs raw bytes that may not align perfectly on UTF-8 boundaries (especially with streaming), the `run_and_stream` function implements careful UTF-8 decoding:

1. Reads chunks of up to 1024 bytes from stdout.
2. Accumulates bytes in a buffer.
3. Attempts `std::str::from_utf8` on the buffer.
4. If decoding fails with `valid_up_to() > 0`, it decodes the valid prefix and keeps the remainder for the next chunk.
5. If `valid_up_to() == 0` and `error_len()` is `None` (incomplete sequence), it waits for more bytes.
6. If `valid_up_to() == 0` and `error_len()` is `Some(...)` (invalid byte), it skips one byte and retries.

This ensures that multi-byte UTF-8 characters spanning chunk boundaries are handled correctly without producing replacement characters or panicking.

## Contributing

Contributions are welcome and encouraged! Here's how you can help:

### Areas for Improvement

- **Chat template abstraction**: The prompt format is currently hardcoded for Llama 3 instruct. A configurable template system would support more models.
- **Multi-platform llama.cpp**: Support for Linux/macOS llama.cpp binaries.
- **Interactive mode**: Allow a human to jump in as one of the agents mid-conversation.
- **Web interface**: A simple web UI for viewing and controlling the conversation.
- **Multiple model backends**: Support for other backends beyond llama.cpp (e.g., OpenAI API, Ollama, vLLM).
- **Conversation branching**: Save and resume conversations from checkpoints.
- **Agent memory**: Long-term memory systems (vector stores, RAG) for more coherent extended conversations.
- **Plugin system**: Allow custom agent behaviors, message filters, and logging sinks.
- **Unit tests**: The codebase currently has no tests — contributions adding test coverage are highly valued.

### Development Setup

```bash
# Clone
git clone https://github.com/ultrapg/zeno_chat.git
cd zeno_chat

# Build
cargo build

# Run with debug output
cargo run

# Run with a custom prompt
cargo run -- --prompt "Agent A: What is the meaning of life?"

# Run a short test conversation
cargo run -- --limit 2
```
## Name

*Zeno Chat — named after Zeno of Elea, famous for paradoxes about infinite divisibility. Much like a Zeno paradox, two agents can keep talking forever, each response getting them "closer" to a complete thought, never quite finishing. In practice, the turn limit stops them from actually going infinite — but the conversation can feel just as endless.*


## License

GNU General Public License v3.0


