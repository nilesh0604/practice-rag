# Free LLM Models Available via NVIDIA NIM

> Deep-research reference of every free, OpenAI-compatible chat / reasoning /
> embedding / vision / coding / safety model hosted on NVIDIA's
> [build.nvidia.com](https://build.nvidia.com) catalog.
>
> **Research date:** 2026-08-15
> **Catalog snapshot:** 2026-08-06 (freellm.net) + 2026-08-15 (build.nvidia.com)
> **Total models documented:** 125 (99 confirmed online, 26 "check provider")

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [Access & Authentication](#2-access--authentication)
3. [Global Constraints & Rate Limits](#3-global-constraints--rate-limits)
4. [OpenAI Compatibility](#4-openai-compatibility)
5. [Quick-Start Code](#5-quick-start-code)
6. [Featured / Flagship Models](#6-featured--flagship-models)
7. [All Free Chat Models (Online)](#7-all-free-chat-models-online)
8. [Models Requiring Provider Verification](#8-models-requiring-provider-verification)
9. [Specialized Models by Category](#9-specialized-models-by-category)
   - [9.1 Reasoning Models](#91-reasoning-models)
   - [9.2 Coding Models](#92-coding-models)
   - [9.3 Vision / Multimodal Models](#93-vision--multimodal-models)
   - [9.4 Embedding / RAG Models](#94-embedding--rag-models)
   - [9.5 Safety / Guardrail Models](#95-safety--guardrail-models)
   - [9.6 Domain-Specific Models](#96-domain-specific-models)
10. [Kimi K2.5 / K2.6 Deep-Dive](#10-kimi-k25--k26-deep-dive)
11. [Programmatic Availability Detection](#11-programmatic-availability-detection)
12. [Caveats, Risks & Best Practices](#12-caveats-risks--best-practices)
13. [Relevance to practice-rag Project](#13-relevance-to-practice-rag-project)
14. [Sources](#14-sources)

---

## 1. Platform Overview

**NVIDIA NIM (NVIDIA Inference Microservices)** provides API access to 100+
open-weight models hosted on NVIDIA infrastructure. The free tier is available
to all NVIDIA Developer Program members (free sign-up).

| Attribute | Value |
|---|---|
| Provider name | NVIDIA NIM |
| Provider type | API Provider |
| Base URL | `https://integrate.api.nvidia.com/v1` |
| OpenAI-compatible | Yes (live-tested Chat Completions) |
| Free tier | 99 free models online (125 total listed) |
| Credit card required | No |
| Phone verification | Yes (one-time OTP) |
| Streaming | Yes |
| Function calling | Yes |
| Vision | Yes |
| Context range | 8K – 1.0M tokens |
| FreeLLM score | 88/100 (Best for easy signup) |
| Last updated | 2026-08-06 |

### Score breakdown

| Dimension | Score |
|---|---|
| Generosity (free limits/access) | 65 |
| Access (signup & key availability) | 100 |
| Model breadth | 95 |
| Reliability | 70 |
| Compatibility (SDK/endpoint) | 100 |
| Quality (model capability) | 95 |

### Use-case coverage

| Use case | Model count |
|---|---|
| Chat | 125 |
| Reasoning | 12 |
| Embedding | 11 |
| Coding | 9 |
| Vision | 2 |

---

## 2. Access & Authentication

### Step-by-step (5 minutes)

1. **Register** at [build.nvidia.com](https://build.nvidia.com) — sign in with
   email or Google.
2. **Verify** email and phone (one-time OTP).
3. **Generate API key** — avatar (top-right) → **API Keys** → Generate API key.
   - No request limits, no key expiration.
   - Key format: `nvapi-xxxxxxxxxxxxxxxx`
4. **Point your client** at NVIDIA — set base URL to
   `https://integrate.api.nvidia.com/v1` and paste the key.
5. **Pick a model** — use a model ID like `moonshotai/kimi-k2.6` or
   `z-ai/glm-5.2` and start sending requests.

### API key location

- Web UI: <https://build.nvidia.com/settings/api-keys>
- Docs: <https://docs.api.nvidia.com/nim/>

### Per-model-family registration

Some models require **additional per-model-family registration**. If you get an
HTTP 403, navigate to the model's page on build.nvidia.com and click
**"Try API"** to register for that specific model family.

---

## 3. Global Constraints & Rate Limits

| Constraint | Value |
|---|---|
| Primary rate limit | **~40 RPM** (requests per minute) |
| Rate-limit scope | **Global across all models** (shared per API key, not per-model) |
| Daily token cap | None stated |
| Credit card | Not required |
| Phone verification | Required (one-time) |
| Context window range | 8K – 1.0M tokens |
| Key expiration | None |

> **Important:** The ~40 RPM limit is **shared across all model calls** on your
> API key. If you use multiple models in parallel, the combined request rate
> cannot exceed ~40 RPM.

---

## 4. OpenAI Compatibility

All NVIDIA NIM chat endpoints are **OpenAI-compatible** — drop-in replacement
for most existing code that targets the OpenAI Chat Completions API.

| Feature | Supported |
|---|---|
| `POST /v1/chat/completions` | Yes |
| `POST /v1/completions` (legacy) | Model-dependent |
| `POST /v1/embeddings` | Yes (embedding models) |
| `GET /v1/models` | Yes (returns full catalog, 189 entries) |
| Streaming (`stream: true`) | Yes |
| Function / tool calling | Yes (model-dependent) |
| Vision (image_url content) | Yes (vision models) |
| `max_tokens`, `temperature`, `top_p` | Yes |

### Supported AI coding assistants

NVIDIA NIM is fully compatible with: Cursor, Zed, OpenCode, Hermes, Claude
Code, Codex, Gemini CLI, Cherry Studio, Lobe Chat, ChatGPT Next Web, DeepChat,
AionUI, OpenCat, and any OpenAI-compatible client.

---

## 5. Quick-Start Code

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-xxxxxxxxxxxxxxxx",
)

response = client.chat.completions.create(
    model="moonshotai/kimi-k2.6",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in one sentence."},
    ],
    max_tokens=512,
    temperature=0.1,
    stream=False,
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="z-ai/glm-5.2",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### curl

```bash
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/kimi-k2.6",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="moonshotai/kimi-k2.6",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-xxxx",
    temperature=0.1,
    max_tokens=512,
)
```

---

## 6. Featured / Flagship Models

The leading open models built by the community, optimized and accelerated by
NVIDIA's enterprise-ready inference runtime (from build.nvidia.com/explore).

| Model ID | Developer | Context | Speciality |
|---|---|---|---|
| `z-ai/glm-5.2` | Z AI | 1.0M | Flagship LLM for agentic workflows, coding, long-horizon reasoning |
| `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA | 1.0M | 550B MoE flagship chat |
| `thinkingmachines/inkling` | Thinking Machines | 256K | Mamba-hybrid 256-expert MoE, text+image reasoning, tool use |
| `deepseek-ai/deepseek-v4-pro` | DeepSeek | 1.0M | 1M-token context, coding, MoE architecture |
| `moonshotai/kimi-k2.6` | Kimi | 262K | Successor to K2.5; agentic + tool use |

---

## 7. All Free Chat Models (Online)

All models below are **confirmed online** with free-tier access as of
2026-08-06. All are OpenAI-compatible, support streaming, and share the
global ~40 RPM rate limit.

### Flagship / Large Models (>100B or 1M context)

| Model ID | Developer | Context | Speciality | Notes |
|---|---|---|---|---|
| `z-ai/glm-5.2` | Z AI | 1.0M | chat, agentic, coding, reasoning | Flagship |
| `deepseek-ai/deepseek-v4-flash-0731` | DeepSeek | 1.0M | chat, coding, agents | 284B MoE, fast |
| `minimaxai/minimax-m3` | MiniMax | 1.0M | chat, multimodal VLM | MoE |
| `moonshotai/kimi-k2.6` | Kimi | 262K | chat, agentic, tool use | Successor to K2.5 |
| `stepfun-ai/step-3.7-flash` | StepFun | 262K | chat | |
| `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA NIM | 1.0M | chat | 550B MoE flagship |
| `poolside/laguna-xs-2.1` | NVIDIA NIM | 262K | chat, coding | 33B MoE agentic coding |
| `google/gemma-4-31b-it` | NVIDIA NIM | 262K | chat, reasoning | Dense 31B, frontier reasoning |
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | NVIDIA NIM | 131K | chat, reasoning | 253B |
| `nvidia/nemotron-3-super-120b-a12b` | NVIDIA NIM | 262K | chat | 120B MoE |
| `nvidia/nemotron-4-340b-instruct` | NVIDIA NIM | 131K | chat, reasoning | 340B |
| `nvidia/nemotron-4-340b-reward` | NVIDIA NIM | 131K | chat, reasoning (reward model) | 340B reward |
| `openai/gpt-oss-120b` | NVIDIA NIM | 131K | chat, reasoning | MoE, fits 80GB GPU |
| `mistralai/mistral-large-2-instruct` | Mistral | 131K | chat | |
| `mistralai/mistral-large` (`mistral-large-3-675b-instruct-2512`) | NVIDIA NIM | 8K | chat | Mistral Large 3 675B |
| `writer/palmyra-creative-122b` | NVIDIA NIM | 131K | chat, creative | 122B |
| `meta/llama-3.1-70b-instruct` | Meta | 131K | chat | |
| `meta/llama-3.3-70b-instruct` | NVIDIA NIM | 128K | chat, reasoning, math, function calling | |
| `nvidia/llama-3.1-nemotron-70b-instruct` | NVIDIA | 131K | chat, reasoning | |
| `nvidia/llama-3.1-nemotron-51b-instruct` | NVIDIA NIM | 131K | chat, reasoning | |
| `mistralai/mixtral-8x22b-v0.1` | NVIDIA NIM | 131K | chat | 8x22B MoE |
| `01-ai/yi-large` | NVIDIA NIM | 131K | chat | |
| `meta/llama-3.2-90b-vision-instruct` | NVIDIA NIM | 8K | chat, vision | 90B VLM |
| `meta/muse-glimmer-30b` | NVIDIA NIM | 131K | chat, multimodal reasoning | Text+image, Onyx tool-calling |

### Mid-Size Models (10B – 100B)

| Model ID | Developer | Context | Speciality | Notes |
|---|---|---|---|---|
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | NVIDIA NIM | 131K | chat, reasoning | High efficiency |
| `nvidia/llama-3.3-nemotron-super-49b-v1` | NVIDIA NIM | 131K | chat | |
| `nvidia/nemotron-nano-3-30b-a3b` | NVIDIA NIM | 131K | chat, reasoning | |
| `nvidia/nemotron-3-nano-30b-a3b` | NVIDIA NIM | 262K | chat | |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA NIM | 256K | chat, reasoning | |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | NVIDIA NIM | 8K | chat | |
| `nvidia/ising-calibration-1.5-31b` | NVIDIA NIM | 8K | chat, quantum calibration VLM | Gemma 4 31B base |
| `meta/llama-3.2-11b-vision-instruct` | Meta | 131K | chat, vision | |
| `meta/codellama-70b` | NVIDIA NIM | 131K | chat, coding | |
| `nvidia/llama3-chatqa-1.5-70b` | NVIDIA NIM | 131K | chat | QA-tuned |
| `writer/palmyra-fin-70b-32k` | NVIDIA NIM | 131K | chat, finance | |
| `writer/palmyra-med-70b` | NVIDIA NIM | 131K | chat, medical | |
| `writer/palmyra-med-70b-32k` | NVIDIA NIM | 131K | chat, medical | 32K context |
| `abacusai/dracarys-llama-3.1-70b-instruct` | NVIDIA NIM | 8K | chat, coding, summarization | Fine-tuned Llama 3.1 70B |
| `mistralai/codestral-22b-instruct-v0.1` | NVIDIA NIM | 131K | chat, coding | |
| `ibm/granite-34b-code-instruct` | NVIDIA NIM | 131K | chat, coding | |
| `nv-mistralai/mistral-nemo-12b-instruct` | NVIDIA NIM | 131K | chat | |
| `mistralai/mistral-7b-instruct-v0.3` | NVIDIA NIM | 131K | chat | |
| `ai21labs/jamba-1.5-large-instruct` | AI21 Labs | 131K | chat | Jamba hybrid |
| `microsoft/phi-3.5-moe-instruct` | NVIDIA NIM | 131K | chat | MoE |
| `meta/llama-guard-4-12b` | NVIDIA NIM | 1.0M | chat, safety | Multimodal safety classifier |
| `mistralai/ministral-14b-instruct-2512` | NVIDIA NIM | 8K | chat, VLM | |
| `nvidia/nemotron-nano-12b-v2-vl` | NVIDIA NIM | 128K | chat, vision | |
| `bigcode/starcoder2-15b` | NVIDIA NIM | 131K | chat, coding | |
| `upstage/solar-10.7b-instruct` | NVIDIA NIM | 8K | chat | |

### Small Models (<10B)

| Model ID | Developer | Context | Speciality | Notes |
|---|---|---|---|---|
| `nvidia/nvidia-nemotron-nano-9b-v2` | NVIDIA NIM | 8K | chat | |
| `nvidia/llama-3.1-nemotron-nano-8b-v1` | NVIDIA NIM | 8K | chat | |
| `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | NVIDIA NIM | 8K | chat, vision | |
| `meta/llama-3.2-3b-instruct` | Meta | 131K | chat | |
| `meta/llama-3.2-1b-instruct` | Meta | 60K | chat | |
| `meta/llama-3.1-8b-instruct` | NVIDIA NIM | 8K | chat | |
| `ibm/granite-3.0-8b-instruct` | NVIDIA NIM | 131K | chat | |
| `ibm/granite-8b-code-instruct` | NVIDIA NIM | 131K | chat, coding | |
| `ibm/granite-3.0-3b-a800m-instruct` | NVIDIA NIM | 131K | chat | |
| `nvidia/mistral-nemo-minitron-8b-8k-instruct` | NVIDIA NIM | 131K | chat | |
| `zyphra/zamba2-7b-instruct` | NVIDIA NIM | 131K | chat | |
| `aisingapore/sea-lion-7b-instruct` | NVIDIA NIM | 131K | chat | Southeast Asian languages |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | NVIDIA NIM | 131K | chat, coding | |
| `google/codegemma-7b` | NVIDIA NIM | 131K | chat, coding | |
| `google/codegemma-1.1-7b` | NVIDIA NIM | 131K | chat, coding | |
| `google/gemma-2b` | NVIDIA NIM | 131K | chat | |
| `google/recurrentgemma-2b` | NVIDIA NIM | 131K | chat | |
| `google/diffusiongemma-26b-a4b-it` | NVIDIA NIM | 8K | chat | Diffusion-based 26B, parallel token gen |
| `nvidia/nemotron-mini-4b-instruct` | NVIDIA NIM | 128K | chat | |
| `nvidia/cosmos-reason2-8b` | NVIDIA NIM | 131K | chat, physical-world reasoning | Vision on videos/images |
| `adept/fuyu-8b` | NVIDIA NIM | 131K | chat, vision | |

### Other / Utility Chat Models

| Model ID | Developer | Context | Speciality | Notes |
|---|---|---|---|---|
| `microsoft/kosmos-2` | NVIDIA NIM | 131K | chat, vision | Grounded multimodal |
| `microsoft/phi-3-vision-128k-instruct` | NVIDIA NIM | 131K | chat, vision | |
| `nvidia/neva-22b` | NVIDIA NIM | 131K | chat, vision | |
| `nvidia/vila` | NVIDIA NIM | 131K | chat, vision | |
| `nvidia/nvclip` | NVIDIA NIM | 131K | chat, vision | |
| `google/deplot` | NVIDIA NIM | 131K | chat, vision | Chart understanding |
| `nvidia/nemoretriever-parse` | NVIDIA NIM | 131K | chat, parsing | |
| `nvidia/nemotron-parse` | NVIDIA NIM | 131K | chat, reasoning, parsing | |
| `nvidia/riva-translate-4b-instruct` | NVIDIA NIM | 131K | chat, translation | 37 languages |
| `nvidia/riva-translate-4b-instruct-v1.1` | NVIDIA NIM | 8K | chat, translation | 12 languages |
| `nvidia/riva-translate-4b-instruct-v2` | NVIDIA NIM | 8K | chat, translation | |
| `mistralai/mistral-nemotron` | NVIDIA NIM | 8K | chat | |
| `meta/llama2-70b` | Meta | 131K | chat | Legacy |
| `databricks/dbrx-instruct` | Databricks | 131K | chat | |
| `mistralai/mixtral-8x7b-instruct-v0.1` | NVIDIA NIM | 8K | chat | |
| `thinkingmachines/inkling` | NVIDIA NIM | 256K | chat, multimodal reasoning | Mamba-hybrid 256-expert MoE |

---

## 8. Models Requiring Provider Verification

These models are listed in the catalog with free-tier access but their live
availability should be verified before use (status: "Check provider"). They may
require additional per-model-family registration or may be temporarily
unavailable.

| Model ID | Developer | Context | Speciality | Notes |
|---|---|---|---|---|
| `deepseek-ai/deepseek-v4-pro` | DeepSeek | 1.0M | chat, coding | 1M context, MoE |
| `z-ai/glm-5.1` | Z AI | 203K | chat | Previous GLM flagship |
| `minimaxai/minimax-m2.7` | MiniMax | 205K | chat, coding, reasoning, office | 230B params |
| `qwen/qwen3.5-397b-a17b` | Alibaba | 262K | chat | 397B MoE |
| `qwen/qwen3.5-122b-a10b` | Alibaba | 262K | chat | 122B MoE |
| `qwen/qwen3-next-80b-a3b-instruct` | NVIDIA NIM | 8K | chat | |
| `stepfun-ai/step-3.5-flash` | StepFun | 262K | chat | |
| `abacusai/dracarys-llama-3.1-70b-instruct` | NVIDIA NIM | 8K | chat, coding | |
| `mistralai/mistral-medium-3.5-128b` | NVIDIA NIM | 8K | chat | |
| `bytedance/seed-oss-36b-instruct` | NVIDIA NIM | 8K | chat | |
| `google/gemma-2-2b-it` | NVIDIA NIM | 8K | chat | |
| `google/gemma-3n-e2b-it` | NVIDIA NIM | 8K | chat | |
| `google/gemma-3n-e4b-it` | NVIDIA NIM | 8K | chat | |
| `meta/llama-4-maverick-17b-128e-instruct` | NVIDIA NIM | 8K | chat | Llama 4 Maverick |
| `mistralai/mistral-small-4-119b-2603` | NVIDIA NIM | 8K | chat | |
| `microsoft/phi-4-multimodal-instruct` | Microsoft | 131K | chat, multimodal | |
| `microsoft/phi-4-mini-instruct` | Microsoft | 8K | chat | |
| `mistralai/ministral-14b-instruct-2512` | NVIDIA NIM | 8K | chat, VLM | |
| `sarvamai/sarvam-m` | NVIDIA NIM | 8K | chat | Indic languages |
| `stockmark/stockmark-2-100b-instruct` | NVIDIA NIM | 8K | chat, Japanese | 100B |
| `upstage/solar-10.7b-instruct` | NVIDIA NIM | 8K | chat | |
| `nvidia/gliner-pii` | NVIDIA NIM | 8K | chat, PII detection | |
| `nvidia/ising-calibration-1-35b-a3b` | NVIDIA NIM | 8K | chat, quantum calibration VLM | |
| `nvidia/nemotron-3-content-safety` | NVIDIA NIM | 128K | chat, safety | |
| `nvidia/nemotron-content-safety-reasoning-4b` | NVIDIA NIM | 128K | chat, safety, reasoning | |

---

## 9. Specialized Models by Category

### 9.1 Reasoning Models

Models explicitly tagged for reasoning / chain-of-thought tasks.

| Model ID | Context | Speciality |
|---|---|---|
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | 131K | 253B reasoning |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | 131K | High-efficiency reasoning |
| `nvidia/llama-3.1-nemotron-70b-instruct` | 131K | Reasoning |
| `nvidia/llama-3.1-nemotron-51b-instruct` | 131K | Reasoning |
| `nvidia/nemotron-nano-3-30b-a3b` | 131K | Nano reasoning |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | 256K | Omni reasoning |
| `nvidia/nemotron-4-340b-instruct` | 131K | 340B reasoning |
| `nvidia/nemotron-4-340b-reward` | 131K | 340B reward model |
| `nvidia/nemotron-parse` | 131K | Reasoning + parsing |
| `nvidia/nemotron-3.5-content-safety` | 128K | Safety reasoning |
| `moonshotai/kimi-k2-thinking` | 256K | Open reasoning, INT4 quant, tool use |
| `openai/gpt-oss-120b` | 131K | MoE reasoning |
| `openai/gpt-oss-20b` | 131K | Smaller MoE reasoning/math |

### 9.2 Coding Models

Models specialized for code generation, completion, and agentic coding.

| Model ID | Context | Speciality |
|---|---|---|
| `deepseek-ai/deepseek-v4-flash` | 1.0M | Fast coding + agents, 284B MoE |
| `deepseek-ai/deepseek-v4-pro` | 1.0M | Coding, 1M context |
| `poolside/laguna-xs-2.1` | 262K | 33B MoE agentic coding/terminal |
| `meta/codellama-70b` | 131K | Code generation, multi-language |
| `abacusai/dracarys-llama-3.1-70b-instruct` | 8K | Fine-tuned Llama 3.1 70B for code |
| `mistralai/codestral-22b-instruct-v0.1` | 131K | Code generation |
| `ibm/granite-34b-code-instruct` | 131K | Enterprise code |
| `ibm/granite-8b-code-instruct` | 131K | Enterprise code |
| `bigcode/starcoder2-15b` | 131K | Open code generation |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | 131K | Code generation |
| `google/codegemma-7b` | 131K | Code completion/generation |
| `google/codegemma-1.1-7b` | 131K | Code completion/generation |
| `nvidia/nv-embedcode-7b-v1` | 131K | Code embedding + coding |

### 9.3 Vision / Multimodal Models

Models that accept image inputs (via `image_url` content type) in addition to
text.

| Model ID | Context | Speciality |
|---|---|---|
| `moonshotai/kimi-k2.5` | — | Multimodal VLM, vision + language, agentic |
| `minimaxai/minimax-m3` | 1.0M | Multimodal MoE VLM |
| `thinkingmachines/inkling` | 256K | Mamba-hybrid MoE, text+image reasoning |
| `meta/muse-glimmer-30b` | 131K | Multimodal reasoning, Onyx tool-calling |
| `meta/llama-3.2-90b-vision-instruct` | 8K | 90B VLM |
| `meta/llama-3.2-11b-vision-instruct` | 131K | 11B VLM |
| `microsoft/phi-3-vision-128k-instruct` | 131K | Vision + 128K context |
| `microsoft/phi-4-multimodal-instruct` | 131K | Multimodal |
| `microsoft/kosmos-2` | 131K | Grounded multimodal |
| `nvidia/neva-22b` | 131K | 22B VLM |
| `nvidia/vila` | 131K | VLM |
| `nvidia/nvclip` | 131K | Vision-language |
| `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | 8K | 8B VLM |
| `nvidia/nemotron-nano-12b-v2-vl` | 128K | 12B VLM |
| `google/deplot` | 131K | Chart/plot understanding |
| `adept/fuyu-8b` | 131K | 8B VLM |
| `nvidia/ising-calibration-1.5-31b` | 8K | Quantum calibration chart VLM |
| `nvidia/ising-calibration-1-35b-a3b` | 8K | Quantum calibration VLM |
| `nvidia/cosmos-reason2-8b` | 131K | Physical-world reasoning on videos/images |
| `mistralai/ministral-14b-instruct-2512` | 8K | VLM, chat/instruction |

### 9.4 Embedding / RAG Models

Models for semantic search, retrieval, and RAG (use `/v1/embeddings` endpoint).

| Model ID | Context | Speciality |
|---|---|---|
| `nvidia/llama-nemotron-embed-1b-v2` | 131K | Multilingual (26 langs), long-doc QA retrieval |
| `nvidia/llama-nemotron-embed-vl-1b-v2` | 131K | Multimodal QA retrieval (text query → image doc) |
| `nvidia/nemotron-3-embed-1b` | 131K | 1B embedding |
| `nvidia/embed-qa-4` | 131K | QA embedding |
| `nvidia/nv-embed-v1` | 131K | General embedding |
| `nvidia/nv-embedqa-e5-v5` | 131K | QA embedding |
| `nvidia/nv-embedqa-mistral-7b-v2` | 131K | 7B Mistral-based QA embedding |
| `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` | 131K | 1B VLM embedding |
| `nvidia/llama-3.2-nv-embedqa-1b-v1` | 131K | 1B QA embedding |
| `nvidia/nv-embedcode-7b-v1` | 131K | 7B code embedding |
| `baai/bge-m3` | 131K | Dense + multi-vector + sparse retrieval |
| `snowflake/arctic-embed-l` | 131K | Long-doc embedding |

### 9.5 Safety / Guardrail Models

Models for content safety, topic control, PII detection, and moderation.

| Model ID | Context | Speciality |
|---|---|---|
| `nvidia/llama-3.1-nemoguard-8b-content-safety` | 8K | Content safety guard |
| `nvidia/llama-3.1-nemoguard-8b-topic-control` | 8K | Topic control guard |
| `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | 128K | Multilingual content safety |
| `meta/llama-guard-4-12b` | 1.0M | Multimodal safety classifier |
| `nvidia/nemotron-3.5-content-safety` | 128K | Safety reasoning |
| `nvidia/nemotron-3-content-safety` | 128K | Content safety |
| `nvidia/nemotron-content-safety-reasoning-4b` | 128K | Safety reasoning, 4B |
| `nvidia/gliner-pii` | 8K | PII detection in text |

### 9.6 Domain-Specific Models

| Model ID | Context | Domain | Speciality |
|---|---|---|---|
| `writer/palmyra-med-70b` | 131K | Medical | Medical chat |
| `writer/palmyra-med-70b-32k` | 131K | Medical | Medical chat, 32K context |
| `writer/palmyra-fin-70b-32k` | 131K | Finance | Financial chat |
| `writer/palmyra-creative-122b` | 131K | Creative | Creative writing, 122B |
| `stockmark/stockmark-2-100b-instruct` | 8K | Japanese | Japanese-language chat, 100B |
| `aisingapore/sea-lion-7b-instruct` | 131K | Southeast Asian | SEA languages |
| `sarvamai/sarvam-m` | 8K | Indic | Indic languages |
| `nvidia/riva-translate-4b-instruct` | 131K | Translation | 37 languages |
| `nvidia/riva-translate-4b-instruct-v1.1` | 8K | Translation | 12 languages |
| `nvidia/riva-translate-4b-instruct-v2` | 8K | Translation | |
| `nvidia/cuopt` | — | Logistics | Route optimization |
| `nvidia/ising-calibration-1.5-31b` | 8K | Quantum computing | Calibration chart understanding |
| `nvidia/ising-calibration-1-35b-a3b` | 8K | Quantum computing | Calibration chart understanding |

---

## 10. Kimi K2.5 / K2.6 Deep-Dive

Kimi K2.5 is the model that prompted this research. Here are the full details.

### Kimi K2.5 (`moonshotai/kimi-k2.5`)

| Attribute | Value |
|---|---|
| Model ID | `moonshotai/kimi-k2.5` |
| Developer | Moonshot AI |
| Type | Multimodal Vision-Language Model (VLM) |
| API | NVIDIA Multimodal APIs (status polling + infer) |
| Modes | Instant mode + thinking mode |
| Paradigms | Conversational + agentic |
| Capabilities | Vision + language understanding, advanced agentic, tool use |
| Self-host container | `nvcr.io/nim/moonshotai/kimi-k2.5-turbo:1.0.0` |
| Container tag | `1.7.0-variant` (specialized base, subject to limitations) |
| Text-only queries | Supported (behaves like a text-only LLM) |
| Image queries | Supported via `image_url` content type |

#### Example (image + text)

```bash
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/kimi-k2.5",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "max_tokens": 1024
  }'
```

### Kimi K2.6 (`moonshotai/kimi-k2.6`)

| Attribute | Value |
|---|---|
| Model ID | `moonshotai/kimi-k2.6` |
| Developer | Kimi (Moonshot AI) |
| Context | 262K tokens |
| Availability | Online (free tier) |
| Use case | chat |
| Speciality | Agentic workflows, tool use (successor to K2.5) |

### Related Kimi models

| Model ID | Context | Speciality |
|---|---|---|
| `moonshotai/kimi-k2-instruct` | 256K | K2 instruct variant |
| `moonshotai/kimi-k2-thinking` | 256K | Open reasoning, native INT4 quant, enhanced tool use |

---

## 11. Programmatic Availability Detection

The `/v1/models` endpoint returns **all 189 catalog entries** but does not
distinguish between free hosted models and everything else. To verify which
models are actually live chat endpoints, probe each with a minimal chat
completion request.

### Detection script (Bash)

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${NVIDIA_API_KEY?Must export NVIDIA_API_KEY}"

BASE_URL="https://integrate.api.nvidia.com/v1"
TIMEOUT=5

# Fetch full catalog
curl -s --max-time "$TIMEOUT" "$BASE_URL/models" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" > /tmp/nim_models.json

# Probe each model
jq -r '.data[].id // empty' /tmp/nim_models.json | while read -r model_id; do
  payload=$(jq -n --arg m "$model_id" '{
    model: $m,
    messages: [
      {"role":"system","content":"You are a helpful assistant."},
      {"role":"user","content":"Respond only with OK."}
    ],
    max_tokens: 8
  }')
  raw=$(curl -s --max-time "$TIMEOUT" -w 'STATUS:%{http_code}' \
    "$BASE_URL/chat/completions" \
    -H "Authorization: Bearer $NVIDIA_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null || echo "STATUS:000")
  status="${raw##*STATUS:}"
  case "$status" in
    200) echo "HOSTED    $model_id" ;;
    429) echo "HOSTED*   $model_id (rate-limited)" ;;
    404|403|401|500) echo "DOWN      $model_id" ;;
    *) echo "AMBIGUOUS $model_id (HTTP $status)" ;;
  esac
done
```

### Status classification

| HTTP Status | Meaning |
|---|---|
| 200 (with `choices` or `model` in JSON) | Hosted and responding to chat completions |
| 429 | Rate-limited but **confirmed hosted** |
| 404 / 403 / 401 / 500 | Not available (removed, not a chat model, or broken) |
| 400 / 422 / 000 / timeout | Ambiguous (non-chat, temporarily down, or different format) |

### Findings from 2026-04-06 probe (189 models)

| Status | Count | Interpretation |
|---|---|---|
| 200 or 429 | 101 | Confirmed hosted |
| 404 | 62 | Removed or never hosted as chat endpoints |
| 500 | 6 | Server errors (temporarily broken) |
| Ambiguous | 20 | 400, 422, timeouts, or no response |

**Observations:**
- Retired models still in catalog: `meta/llama2-70b`, `databricks/dbrx-instruct`,
  `adept/fuyu-8b` return 404.
- Embedding models return 404 on `/chat/completions` (they only serve
  `/v1/embeddings`).
- Large models (253B, 397B) may time out within a 5-second window due to
  cold-start.
- Duplicate entries exist (e.g. `nvidia/nemotron-3-super-120b-a12b`,
  `openai/gpt-oss-120b`).

---

## 12. Caveats, Risks & Best Practices

### Caveats

| Caveat | Detail |
|---|---|
| Free isn't forever | NVIDIA likely monetizes once the catalog gets popular. Pricing is the obvious next step. Lock in testing now. |
| Shared rate limit | ~40 RPM is **global across all models** on your API key, not per-model. Parallel multi-model calls share the budget. |
| Catalog rotates | Models get retired (e.g. `meta/llama2-70b`, `databricks/dbrx-instruct` now return 404). Run a probe script periodically. |
| Large models cold-start | 253B / 397B models may time out on first request within short windows. Use longer timeouts. |
| Per-family registration | Some models require additional per-model-family registration. HTTP 403 → visit the model page and click "Try API". |
| Production reliability | Free APIs from large vendors are fine for prototyping but unreliable for production. The proof is in the SLO. |
| Open weights only | The catalog currently hosts open-weight models only. No GPT-class or Claude-class closed models. |
| Phone verification required | One-time OTP during signup. |

### Best Practices

1. **Use smaller models for high-volume tasks** — 7B/8B models (e.g.
   `meta/llama-3.1-8b-instruct`) have lower latency and are less likely to
   cold-start.
2. **Reserve large models for complex reasoning** — `z-ai/glm-5.2`,
   `moonshotai/kimi-k2.6`, `deepseek-ai/deepseek-v4-flash` for agentic/coding.
3. **Implement retry with backoff** — handle 429s gracefully; the limit is
   shared.
4. **Cache responses** — no daily token cap, but 40 RPM is tight for burst
   traffic.
5. **Probe availability periodically** — run the detection script weekly to
   track which models come and go.
6. **Use a router** — projects like `rohansx/nvidia-litellm-router` auto-route
   across 31 free NIM models with latency-based routing and failover.
7. **Keep OpenAI as fallback** — since NIM is OpenAI-compatible, you can swap
   base URLs at runtime with no code changes.

---

## 13. Relevance to practice-rag Project

This project (`practice-rag`) currently uses **Ollama with `llama3.2:3b`**
locally for generation, guardrails, and query classification. NVIDIA NIM offers
free access to much stronger models that could be used for comparison testing
or as an optional cloud provider.

### Directly relevant NIM models

| Project component | Current (Ollama) | NIM alternative | Benefit |
|---|---|---|---|
| `rag/generator.py` (answer generation) | `llama3.2:3b` / `llama3.1:8b` | `moonshotai/kimi-k2.6`, `z-ai/glm-5.2`, `deepseek-ai/deepseek-v4-flash` | Stronger reasoning, 1M context, no local GPU |
| `api/guardrails.py` (content safety) | `llama3.2:3b` LLM judge | `nvidia/llama-3.1-nemoguard-8b-content-safety`, `meta/llama-guard-4-12b` | Purpose-built safety models |
| `api/guardrails.py` (topic control) | regex fallback | `nvidia/llama-3.1-nemoguard-8b-topic-control` | Dedicated topic-control guard |
| `api/guardrails.py` (PII scrubbing) | regex | `nvidia/gliner-pii` | ML-based PII detection |
| Embedding (RAG retrieval) | local Ollama embeddings | `nvidia/llama-nemotron-embed-1b-v2`, `baai/bge-m3` | Multilingual, long-doc QA |
| Reranking | (not implemented) | `nvidia/llama-nemotron-rerank-1b-v2` | GPU-accelerated reranking |

### Integration approach

Because NIM is **OpenAI-compatible**, adding it as an optional provider in
`generator.py` is a one-line base URL + key change. The lazy-client pattern
already used in the project would allow injecting a NIM-backed client for
testing while keeping Ollama as the default for local development.

---

## 14. Sources

| Source | URL | Date accessed |
|---|---|---|
| NVIDIA NIM LLM APIs reference | <https://docs.api.nvidia.com/nim/reference/llm-apis> | 2026-08-15 |
| NVIDIA NIM Multimodal APIs reference | <https://docs.api.nvidia.com/nim/reference/multimodal-apis> | 2026-08-15 |
| build.nvidia.com models catalog | <https://build.nvidia.com/models> | 2026-08-15 |
| build.nvidia.com explore (featured) | <https://build.nvidia.com/explore> | 2026-08-15 |
| Kimi K2 Thinking model card | <https://build.nvidia.com/moonshotai/kimi-k2-thinking> | 2026-08-15 |
| Kimi K2.5 API docs (VLM) | <https://docs.nvidia.com/nim/vision-language-models/latest/examples/kimi-k2.5/api.html> | 2026-08-15 |
| Kimi K2.5 get-started (LLM turbo) | <https://docs.nvidia.com/nim/large-language-models/latest/turbo/get-started-kimi-k2-5-turbo.html> | 2026-08-15 |
| NVIDIA NIM for Developers | <https://developer.nvidia.com/nim> | 2026-08-15 |
| API Catalog Quickstart | <https://docs.api.nvidia.com/nim/docs/api-quickstart> | 2026-08-15 |
| NIM for LLMs supported models | <https://docs.nvidia.com/nim/large-language-models/latest/_include/models.html> | 2026-08-15 |
| freeLLM.net — NVIDIA NIM provider page | <https://freellm.net/providers/nvidia-nim> | 2026-08-15 |
| Kilo.ai — NVIDIA NIM + Kilo Code tutorial | <https://blog.kilo.ai/p/nvidia-nim-kilo-code-free-kimi-k25> | 2026-08-15 |
| Kargin-Utkin — "NVIDIA Just Quietly Became a Free Model API Provider" | <https://www.kargin-utkin.com/nvidia_free_models> | 2026-08-15 |
| Steve Scargall — "Using the API to Find Free Hosted Models on NVIDIA Builder" | <https://stevescargall.com/blog/2026/04/using-the-api-to-find-free-hosted-models-on-nvidia-builder/> | 2026-08-15 |
| rohansx/nvidia-litellm-router (GitHub) | <https://github.com/rohansx/nvidia-litellm-router/> | 2026-08-15 |

---

*This document is a research snapshot. The NVIDIA NIM catalog rotates
regularly — verify model availability with the detection script in
[Section 11](#11-programmatic-availability-detection) before relying on any
specific model in production.*
