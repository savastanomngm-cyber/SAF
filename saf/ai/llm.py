"""Multi-provider LLM layer with tier-based routing.
DEEP chain:    Nous Ox Alpha -> Gemini Flash -> Groq Llama 70B
INSTANT chain: Gemini Flash-Lite -> Groq Llama 8B -> Nous Ox Alpha (fallback)
"""
import json, re, time
from openai import OpenAI
from ..security import get_key

# ─── Model chains (ordered by preference) ───
DEEP_CHAIN = [
    ("nous",   "stealth/ox-alpha"),
    ("gemini", "gemini-2.5-flash"),
    ("groq",   "llama-3.3-70b-versatile"),
]
FAST_CHAIN = [
    ("gemini", "gemini-2.5-flash-lite"),
    ("groq",   "llama-3.1-8b-instant"),
    ("nous",   "stealth/ox-alpha"),
]

# ─── Provider endpoints ───
NOUS_BASE     = "https://inference-api.nousresearch.com/v1"
GROQ_BASE     = "https://api.groq.com/openai/v1"
GEMINI_BASE   = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE = "https://api.deepseek.com/v1"

MAX_RETRIES = 2
RETRY_DELAY = 2
_MODE = "auto"


def set_mode(m):
    global _MODE
    _MODE = m if m in ("deep", "instant", "auto") else "auto"


def resolve_chain(task="auto"):
    """Route to the correct provider chain.
    task='deep'  -> ALWAYS DEEP_CHAIN (Ox Alpha first) regardless of UI mode
    task='fast'  -> FAST_CHAIN unless UI mode is 'deep'
    task='auto'  -> respects UI mode (instant->FAST, deep/auto->DEEP)
    """
    if task == "deep":
        return DEEP_CHAIN
    if _MODE == "deep":
        return DEEP_CHAIN
    if _MODE == "instant":
        return FAST_CHAIN
    return FAST_CHAIN if task == "fast" else DEEP_CHAIN


def _client(provider):
    """Build an OpenAI-compatible client for the given provider."""
    if provider == "nous":
        key = get_key("NOUS_API_KEY")
        if not key:
            return None
        return OpenAI(api_key=key, base_url=NOUS_BASE)
    if provider == "gemini":
        key = get_key("GEMINI_API_KEY")
        if not key:
            return None
        return OpenAI(api_key=key, base_url=GEMINI_BASE)
    if provider == "deepseek":
        key = get_key("DEEPSEEK_API_KEY")
        if not key:
            return None
        return OpenAI(api_key=key, base_url=DEEPSEEK_BASE)
    # groq
    key = get_key("GROQ_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key, base_url=GROQ_BASE)


def _strip_wrappers(text):
    """Remove think tags and markdown fences that reasoning models add."""
    if not text:
        return text
    text = re.sub(r"", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    return text.strip()


def complete(system, user, temperature=0.7, max_tokens=4096,
             task="auto", force_provider=None, force_model=None, timeout=240):
    chain = [(force_provider, force_model)] if (force_provider and force_model) else resolve_chain(task)
    last_err = "no provider attempted"
    for provider, model in chain:
        client = _client(provider)
        if not client:
            print(f"[llm] {provider} SKIPPED — no API key")
            last_err = f"no key for {provider}"
            continue
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                )
                txt = (resp.choices[0].message.content or "").strip()
                txt = _strip_wrappers(txt)
                if txt:
                    print(f"[llm] {provider}:{model} responded ({len(txt)} chars)")
                    return txt, {"provider": provider, "model": model}
                last_err = "empty response"
            except Exception as e:
                err = str(e)
                last_err = err[:200]
                if "429" in err or "rate" in err.lower() or "TPD" in err or "tokens per day" in err.lower():
                    print(f"[llm] {provider}:{model} rate limited -> next")
                    break
                if "404" in err or "not_found" in err or "decommissioned" in err:
                    print(f"[llm] {provider}:{model} not found -> next")
                    break
                if "413" in err or "too large" in err.lower():
                    print(f"[llm] {provider}:{model} request too large -> next")
                    break
                if "timeout" in err.lower() or "timed out" in err.lower():
                    print(f"[llm] {provider}:{model} timeout (attempt {attempt}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES:
                        time.sleep(2)
                        continue
                    break
                print(f"[llm] {provider}:{model} error: {err[:100]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                break
    print(f"[llm] ALL providers exhausted. Last error: {last_err}")
    return "", {"error": last_err}


def complete_json(system, user, temperature=0.3, max_tokens=4096,
                  task="auto", force_provider=None, force_model=None, timeout=240):
    chain = [(force_provider, force_model)] if (force_provider and force_model) else resolve_chain(task)
    last_err = "no provider attempted"
    for provider, model in chain:
        client = _client(provider)
        if not client:
            print(f"[llm] {provider} SKIPPED — no API key")
            last_err = f"no key for {provider}"
            continue
        use_json = True
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                kwargs = dict(model=model,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}],
                              temperature=temperature, max_tokens=max_tokens, timeout=timeout)
                # Gemini, Groq, DeepSeek support response_format; Nous doesn't
                if use_json and provider in ("groq", "deepseek", "gemini"):
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                txt = (resp.choices[0].message.content or "").strip()
                parsed = extract_json(txt)
                if parsed is not None:
                    print(f"[llm] {provider}:{model} JSON ok ({len(txt)} chars)")
                    return parsed, {"provider": provider, "model": model}
                print(f"[llm] {provider}:{model} JSON parse failed. Raw: {txt[:120]}...")
                last_err = f"unparseable: {txt[:120]}"
            except Exception as e:
                err = str(e)
                last_err = err[:200]
                if "response_format" in err or ("json" in err.lower() and "400" in err):
                    use_json = False
                    continue
                if "429" in err or "rate" in err.lower() or "TPD" in err or "tokens per day" in err.lower():
                    print(f"[llm] {provider}:{model} rate limited -> next")
                    break
                if "404" in err or "not_found" in err or "decommissioned" in err:
                    print(f"[llm] {provider}:{model} not found -> next")
                    break
                if "413" in err or "too large" in err.lower():
                    print(f"[llm] {provider}:{model} request too large -> next")
                    break
                if "timeout" in err.lower() or "timed out" in err.lower():
                    print(f"[llm] {provider}:{model} timeout (attempt {attempt}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES:
                        time.sleep(2)
                        continue
                    break
                print(f"[llm] {provider}:{model} error: {err[:100]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                break
    print(f"[llm] ALL providers exhausted. Last error: {last_err}")
    return None, {"error": last_err}


def extract_json(text):
    """Robust JSON extraction — greedy regex approach that works."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    cleaned = _strip_wrappers(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None