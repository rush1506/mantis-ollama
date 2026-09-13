import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Tuple, Union

# Load credentials from a local, git-ignored .env file (e.g. OLLAMA_API_KEY) so
# API keys never live in source control. The first existing file wins; existing
# process environment variables always take precedence (never overwritten).
for _mantis_env in (
    Path(__file__).resolve().parent / ".env",          # reference/core/.env
    Path(__file__).resolve().parent.parent / ".env",   # reference/.env
    Path(__file__).resolve().parent.parent.parent / ".env",  # repo root .env
):
    if _mantis_env.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=str(_mantis_env), override=False, verbose=False)
        except Exception:
            pass
        break

# Suppress ADK warning regarding Gemini via LiteLLM to maintain unified error handling and backoff
os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")

try:
    import litellm
    litellm.drop_params = True
    litellm.suppress_debug_info = True
except Exception:
    pass

try:
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient, LlmCapabilities
    from pydantic import Field
except Exception:
    LiteLlm = object
    LiteLLMClient = object
    LlmCapabilities = None

    def Field(**kwargs: Any) -> Any:
        factory = kwargs.get("default_factory")
        if callable(factory):
            return factory()
        return kwargs.get("default", None)

# Default model served by a local Ollama daemon (OpenAI-compatible).
# Set OLLAMA_MODEL to override, or use a bare `ollama/<model>` / `openai/<model>`
# id together with LLM_API_BASE to point at any OpenAI-compatible endpoint
# (including Ollama Cloud at https://ollama.com/v1).
DEFAULT_MODEL = "ollama/deepseek-v4-flash:cloud"
# Local Ollama daemon (default port 11434) exposed through its OpenAI-compatible /v1.
DEFAULT_API_BASE = os.environ.get("DEFAULT_API_BASE") or "http://localhost:11434/v1"
# Ollama Cloud native completion base (LiteLLM's 'ollama' provider appends
# /api/generate, so the base must NOT carry a trailing /v1).
OLLAMA_CLOUD_API_BASE = "https://ollama.com"
# Ollama Cloud OpenAI-compatible base (used when a 'ollama.cloud/<m>' alias is
# rewritten to the 'openai/' provider, which appends /chat/completions).
OLLAMA_CLOUD_OPENAI_BASE = "https://ollama.com/v1"
OLLAMA_CLOUD_PREFIX = "ollama.cloud/"
SUPPORTED_SANDBOXES = ("static-only", "static", "gvisor", "microsandbox", "gce")
RECOMMENDED_MODELS = (
    "ollama/deepseek-v4-flash:cloud",
    "ollama/deepseek-v4.1-flash",
    "ollama/glm-5.3",
    "ollama/glm-5.3-flash",
    "ollama/minimax-m3",
    "ollama/kimi-k3",
    "ollama/qwen3.5",
)

PLACEHOLDER_STRINGS = {
    "YOUR_PROJECT_ID",
    "YOUR_PROJECT",
    "YOUR_GCP_PROJECT",
    "<YOUR_PROJECT_ID>",
    "<PROJECT_ID>",
    "YOUR_API_KEY",
    "<YOUR_API_KEY>",
    "TODO",
    "CHANGE_ME",
    "REPLACE_ME",
    "",
}


def is_placeholder(val: Any) -> bool:
    """Checks if a string or value represents an unconfigured placeholder."""
    if val is None:
        return True
    s = str(val).strip()
    if not s or s.upper() in PLACEHOLDER_STRINGS:
        return True
    if s.upper().startswith(("YOUR_", "<YOUR_", "CHANGE_ME", "REPLACE_ME")):
        return True
    # Token-level check for composite paths/models (e.g. openai/YOUR_API_KEY)
    for token in s.upper().split("/"):
        t = token.strip()
        if t in PLACEHOLDER_STRINGS or t.startswith(("YOUR_", "<YOUR_", "CHANGE_ME", "REPLACE_ME")):
            return True
    return False


def _is_unconfigured_api_base(val: Any) -> bool:
    """Returns True for an api_base that is empty, a placeholder, or not a URL.

    A valid HTTP(S) URL is never treated as unconfigured. This is checked before
    is_placeholder, because is_placeholder would otherwise flag the empty segment
    produced by the '://' in any URL.
    """
    if val is None:
        return True
    s = str(val).strip()
    if not s:
        return True
    if s.lower().startswith(("http://", "https://")):
        return False
    return True


def normalize_model_id(model_id: str) -> str:
    """Normalizes model names and routes model ids to an OpenAI-compatible endpoint.

    Routing is local-first and removes the Google/Vertex AI dependency:
      * `ollama/<model>` -> local Ollama daemon (OpenAI-compatible /v1 at
        DEFAULT_API_BASE, usually http://localhost:11434/v1).
      * `ollama.cloud/<model>` -> Ollama Cloud's OpenAI-compatible endpoint
        (https://ollama.com/v1).
      * `openai/<model>` -> any OpenAI-compatible endpoint resolved at call time
        (LLM_API_BASE / api_base / default api base), e.g. vLLM, LM Studio, or
        Ollama Cloud.
      * A bare model name with no provider prefix is assumed to be a local
        Ollama model and routed accordingly.
      * Direct Anthropic remains supported only when ANTHROPIC_API_KEY is set.
    """
    if not model_id:
        return DEFAULT_MODEL
    cleaned = model_id.strip()
    if cleaned.startswith("ollama/"):
        return cleaned
    if cleaned.startswith(OLLAMA_CLOUD_PREFIX):
        return cleaned.replace(OLLAMA_CLOUD_PREFIX, "openai/", 1)
    if cleaned.startswith("openai/"):
        return cleaned
    if cleaned.startswith("claude-") and os.environ.get("ANTHROPIC_API_KEY"):
        return f"anthropic/{cleaned}"
    if "/" not in cleaned:
        # Fallback: treat a bare model id as a local Ollama model.
        return f"ollama/{cleaned}"
    return cleaned


def is_rate_limit_error(e: Exception) -> bool:
    """Detects whether an exception represents a 429 / RateLimitError / RESOURCE_EXHAUSTED error."""
    try:
        import litellm
        if isinstance(e, (getattr(litellm, "RateLimitError", ()), getattr(litellm.exceptions, "RateLimitError", ()))):
            return True
    except Exception:
        pass
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status in (429, 503, 529):
        return True
    resp = getattr(e, "response", None)
    if resp and getattr(resp, "status_code", None) in (429, 503, 529):
        return True
    msg = str(e).lower()
    if any(k in msg for k in ("ratelimiterror", "resource_exhausted", "quota exceeded", "rate limit", "rate_limit", "overloaded", "too many requests")):
        return True
    if "429" in msg and any(k in msg for k in ("quota", "token", "limit", "exhausted", "aiplatform", "prediction")):
        return True
    return False


def is_retryable_llm_error(e: Exception) -> bool:
    """Detects whether an exception represents a retryable transient error (429, 408, 5xx, timeout, or network reset)."""
    if isinstance(e, MantisEmptyTurnError):
        return True
    if is_auth_error(e) or isinstance(e, (MantisAuthError, PermissionError, FileNotFoundError)):
        return False
    if is_rate_limit_error(e):
        return True
    if isinstance(e, json.decoder.JSONDecodeError):
        return True
    try:
        import litellm
        timeout_classes = (
            getattr(litellm, "Timeout", ()),
            getattr(getattr(litellm, "exceptions", None), "Timeout", ()),
            getattr(litellm, "InternalServerError", ()),
            getattr(getattr(litellm, "exceptions", None), "InternalServerError", ()),
            getattr(litellm, "ServiceUnavailableError", ()),
            getattr(getattr(litellm, "exceptions", None), "ServiceUnavailableError", ()),
            getattr(litellm, "BadGatewayError", ()),
            getattr(getattr(litellm, "exceptions", None), "BadGatewayError", ()),
        )
        valid_timeout_classes = tuple(c for c in timeout_classes if isinstance(c, type))
        if valid_timeout_classes and isinstance(e, valid_timeout_classes):
            return True
    except Exception:
        pass
    if isinstance(e, (asyncio.TimeoutError, TimeoutError, ConnectionResetError, BrokenPipeError)):
        return True
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status in (408, 429, 500, 502, 503, 504, 529):
        return True
    resp = getattr(e, "response", None)
    if resp and getattr(resp, "status_code", None) in (408, 429, 500, 502, 503, 504, 529):
        return True
    msg = str(e).lower()
    if any(
        k in msg
        for k in (
            "timeout",
            "timed out",
            "connection reset",
            "connection closed",
            "sockettimeout",
            "remotepathreset",
            "broken pipe",
            "internal server error",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
    ):
        return True
    return False


def extract_retry_after(e: Exception) -> Optional[float]:
    """Extracts suggested retry delay in seconds from response headers if present."""
    headers = getattr(e, "headers", None)
    if not headers:
        resp = getattr(e, "response", None)
        if resp:
            headers = getattr(resp, "headers", None)
    if headers and isinstance(headers, (dict, Mapping)):
        for k, v in headers.items():
            if k.lower() == "retry-after":
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
    return None


def extract_rate_limit_detail(e: Exception) -> str:
    """Extracts a succinct summary of the rate limit reason from the exception."""
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg:
        m = re.search(r"Quota exceeded for ([^.\s]+(?:\.[^.\s]+)*)", msg)
        if m:
            metric = m.group(1).split("/")[-1]
            return f"Quota exceeded: {metric}"
        return "RESOURCE_EXHAUSTED"
    if "tokens_per_minute" in msg:
        return "Tokens per minute limit exceeded"
    if "requests_per_minute" in msg:
        return "Requests per minute limit exceeded"
    return "Rate limit (429)"


def compute_full_jitter_delay(
    attempt: int,
    initial_delay: float = 5.0,
    max_delay: float = 60.0,
    min_offset: float = 5.0,
    backoff_factor: float = 2.0,
    retry_after: Optional[float] = None,
    max_remaining: Optional[float] = None,
) -> float:
    """Computes backoff delay using full jitter with a minimum offset."""
    upper_bound = min(max_delay, max(min_offset, initial_delay * (backoff_factor ** attempt)))
    if upper_bound <= min_offset:
        delay = min_offset
    else:
        delay = random.uniform(min_offset, upper_bound)

    if retry_after is not None and retry_after > 0:
        delay = min(max_delay, max(delay, retry_after))

    if max_remaining is not None:
        delay = min(delay, max(1.0, max_remaining))

    return delay


class MantisAuthError(RuntimeError):
    """Raised when an unrecoverable LLM authentication or token refresh failure occurs."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


class MantisStreamingTruncationError(RuntimeError):
    """Raised when streaming tool call arguments are truncated mid-stream by output limits or premature chunk completion."""
    pass


class MantisEmptyTurnError(RuntimeError):
    """Raised when a non-streaming LLM turn finishes with STOP but zero content parts.

    ADK surfaces this case as ``MODEL_RETURNED_NO_CONTENT``. For a given conversation
    state it is often DETERMINISTIC (DeepSeek/Ollama swallowing all output on
    think=true), so the wrapper retries it only a small bounded number of times,
    then raises MantisEmptyTurnExhaustedError (non-retryable) rather than hanging
    under the generic 1h transient patience.
    """

    pass


class MantisEmptyTurnExhaustedError(RuntimeError):
    """Raised after the bounded empty-turn retry budget is exhausted.

    This is intentionally NOT retryable (unlike ``MantisEmptyTurnError``): a request
    that returns empty repeatedly for the same conversation state will not recover,
    and retrying it under the 1h patience only hangs the campaign.
    """

    pass


def is_auth_error(e: Optional[Exception]) -> bool:
    """Detects whether an exception represents an authentication or token refresh failure."""
    if e is None:
        return False
    if isinstance(e, MantisAuthError):
        return True

    # Check known exception class names
    exc_cls_name = getattr(getattr(e, "__class__", None), "__name__", "")
    if exc_cls_name in (
        "RefreshError",
        "DefaultCredentialsError",
        "ReauthError",
        "ReauthFailError",
        "ReauthSamlChallengeFailError",
        "AuthenticationError",
    ):
        return True

    # Check known exception classes
    auth_classes = []
    try:
        import google.auth.exceptions
        for name in (
            "RefreshError",
            "DefaultCredentialsError",
            "ReauthFailError",
            "ReauthSamlChallengeFailError",
            "OAuthError",
            "UserAccessTokenError",
        ):
            cls = getattr(google.auth.exceptions, name, None)
            if cls is not None and isinstance(cls, type):
                auth_classes.append(cls)
    except Exception:
        pass

    try:
        import litellm
        for name in ("AuthenticationError",):
            cls = getattr(litellm, name, None) or getattr(getattr(litellm, "exceptions", None), name, None)
            if cls is not None and isinstance(cls, type):
                auth_classes.append(cls)
    except Exception:
        pass

    valid_classes = tuple(c for c in auth_classes if isinstance(c, type))
    if valid_classes and isinstance(e, valid_classes):
        return True

    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status == 401:
        return True
    resp = getattr(e, "response", None)
    if resp and getattr(resp, "status_code", None) == 401:
        return True

    curr = e
    while curr is not None:
        msg = str(curr).lower()
        if any(
            phrase in msg
            for phrase in (
                "reauthentication is needed",
                "gcloud auth application-default login",
                "invalid_grant",
                "your default credentials were not found",
                "could not automatically determine credentials",
                "reauthentication required",
                "reauthentication challenge",
                "authenticationerror",
                "credentials are expired",
                "credentials expired",
                "failed to retrieve auth token",
                "invalid api key",
                "incorrect api key",
                "unauthenticated",
            )
        ):
            return True
        curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

    return False


def format_auth_error_message(e: Exception, model: str = "") -> str:
    """Formats an informative, clean authentication error banner without tracebacks."""
    err_text = str(e).strip()
    if err_text.startswith("=" * 10) and "[AUTHENTICATION ERROR]" in err_text:
        return err_text
    cause_msg = ""
    curr = e
    while curr is not None:
        c_str = str(curr).strip()
        if c_str and not c_str.startswith("Traceback") and not c_str.startswith("PIPELINE"):
            cause_msg = c_str
        curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

    cause_clean = cause_msg or err_text
    if "AuthenticationError:" in cause_clean:
        cause_clean = cause_clean.split("AuthenticationError:")[-1].strip()

    model_label = f" for '{model}'" if model else ""
    lines = [
        "=" * 80,
        f" ❌ [AUTHENTICATION ERROR] Cloud / LLM Authentication Failed{model_label}",
        "=" * 80,
        f"  • Cause: {cause_clean}",
    ]

    is_gcp = any(
        k in cause_clean.lower() or k in model.lower()
        for k in ("gcloud", "google", "vertex", "gemini", "adc", "application-default", "refresh_token", "rapt")
    ) or not model or model.startswith("vertex_ai/") or model.startswith("gemini/")

    is_anthropic = "anthropic" in model.lower() or "anthropic" in cause_clean.lower()
    is_openai = "openai" in model.lower() or "openai" in cause_clean.lower()

    if is_gcp:
        lines.extend([
            "  • If using Vertex AI with personal Application Default Credentials (ADC):",
            "      gcloud auth application-default login",
            "  • Or if using a Google Cloud service account key:",
            '      export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account_key.json"',
        ])
    if is_anthropic:
        lines.extend([
            "  • If using Anthropic API directly:",
            '      export ANTHROPIC_API_KEY="your-anthropic-api-key"',
        ])
    if is_openai:
        lines.extend([
            "  • If using OpenAI API directly:",
            '      export OPENAI_API_KEY="your-openai-api-key"',
        ])

    lines.append("=" * 80)
    return "\n".join(lines)


def clear_vertex_credential_caches() -> None:
    """Clears cached credentials across all LiteLLM Vertex AI handlers to force token reload."""
    try:
        import gc
        from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

        for obj in gc.get_objects():
            if isinstance(obj, VertexBase):
                mapping = getattr(obj, "_credentials_project_mapping", None)
                if isinstance(mapping, dict):
                    mapping.clear()
    except Exception:
        pass


def is_token_refreshable_auth_error(e: Exception) -> bool:
    """Checks if an auth error is due to an expired access token that can be refreshed automatically."""
    curr = e
    while curr is not None:
        msg = str(curr).lower()
        if any(
            phrase in msg
            for phrase in (
                "reauthentication is needed",
                "gcloud auth application-default login",
                "invalid_grant",
                "your default credentials were not found",
                "could not automatically determine credentials",
                "invalid api key",
                "incorrect api key",
            )
        ):
            return False
        curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status == 401:
        return True
    resp = getattr(e, "response", None)
    if resp and getattr(resp, "status_code", None) == 401:
        return True

    curr = e
    while curr is not None:
        msg = str(curr).lower()
        if any(
            phrase in msg
            for phrase in (
                "credentials are expired",
                "credentials expired",
                "unauthenticated",
                "401",
                "token has expired",
                "access token expired",
            )
        ):
            return True
        curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

    return False


def try_refresh_auth() -> bool:
    """Attempts to refresh Application Default Credentials and clear LiteLLM token caches."""
    clear_vertex_credential_caches()
    try:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        return getattr(creds, "valid", False)
    except Exception:
        return False


class ExpectedErrorLoggingFilter(logging.Filter):
    """Filters out noisy, multi-frame tracebacks for expected operational conditions

    (such as MantisAuthError or BudgetExceededError), while strictly preserving full
    tracebacks for unexpected bugs or runtime exceptions.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exc_val = record.exc_info[1]
            if exc_val is not None:
                if is_auth_error(exc_val) or isinstance(exc_val, MantisAuthError):
                    return False
                exc_cls_name = getattr(getattr(exc_val, "__class__", None), "__name__", "")
                if exc_cls_name in (
                    "BudgetExceededError",
                    "LlmCallsLimitExceededError",
                    "MantisAuthError",
                ):
                    return False
                cause = getattr(exc_val, "__cause__", None) or getattr(exc_val, "__context__", None)
                if cause is not None and (
                    is_auth_error(cause)
                    or isinstance(cause, MantisAuthError)
                    or getattr(getattr(cause, "__class__", None), "__name__", "") in (
                        "BudgetExceededError",
                        "LlmCallsLimitExceededError",
                        "MantisAuthError",
                    )
                ):
                    return False

        # Also suppress messages directly matching authentication failure signatures
        msg = record.getMessage() if hasattr(record, "getMessage") else str(record.msg)
        if any(
            pattern in msg
            for pattern in (
                "Failed to load vertex credentials",
                "Reauthentication is needed",
                "gcloud auth application-default login",
            )
        ):
            return False

        return True


_expected_error_filter = ExpectedErrorLoggingFilter()

if getattr(logging, "lastResort", None):
    logging.lastResort.addFilter(_expected_error_filter)

_orig_logger_add_handler = logging.Logger.addHandler


def _filtered_add_handler(self: logging.Logger, h: logging.Handler) -> None:
    h.addFilter(_expected_error_filter)
    return _orig_logger_add_handler(self, h)


logging.Logger.addHandler = _filtered_add_handler


def apply_expected_error_filter() -> None:
    """Attaches _expected_error_filter to root, all existing loggers, and LiteLLM/ADK handlers."""
    root_lg = logging.getLogger()
    root_lg.addFilter(_expected_error_filter)
    for h in root_lg.handlers:
        h.addFilter(_expected_error_filter)

    known_loggers = [
        "",
        "LiteLLM",
        "litellm",
        "litellm._logging",
        "google_adk",
        "google_adk.google.adk.runners",
        "google.adk",
        "google.adk.runners",
        "google.adk.workflow._node_runner",
        "google_adk.google.adk.workflow._node_runner",
        "google.adk.workflow",
        "google_adk.google.adk.workflow",
        "google_adk.google.adk.flows.llm_flows.base_llm_flow",
        "google.adk.flows.llm_flows.base_llm_flow",
    ]
    for name in known_loggers:
        lg = logging.getLogger(name)
        lg.addFilter(_expected_error_filter)
        for h in lg.handlers:
            h.addFilter(_expected_error_filter)

    for lg in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(lg, logging.Logger):
            lg.addFilter(_expected_error_filter)
            for h in lg.handlers:
                h.addFilter(_expected_error_filter)

    try:
        import litellm
        from litellm._logging import verbose_logger
        verbose_logger.addFilter(_expected_error_filter)
        for h in verbose_logger.handlers:
            h.addFilter(_expected_error_filter)
    except Exception:
        pass


apply_expected_error_filter()

# Disable ADK node retry on unrecoverable authentication errors or budget pauses
try:
    import google.adk.workflow.utils._retry_utils as _adk_retry_utils
    _orig_adk_should_retry_node = _adk_retry_utils._should_retry_node

    def _non_retryable_should_retry_node(
        exception: BaseException,
        retry_config: Any,
        node_state: Any,
    ) -> bool:
        if is_auth_error(exception) or isinstance(exception, MantisAuthError):
            return False
        exc_cls_name = getattr(getattr(exception, "__class__", None), "__name__", "")
        if exc_cls_name in (
            "BudgetExceededError",
            "LlmCallsLimitExceededError",
            "MantisAuthError",
        ):
            return False
        cause = getattr(exception, "__cause__", None) or getattr(exception, "__context__", None)
        if cause is not None and (
            is_auth_error(cause)
            or isinstance(cause, MantisAuthError)
            or getattr(getattr(cause, "__class__", None), "__name__", "") in (
                "BudgetExceededError",
                "LlmCallsLimitExceededError",
                "MantisAuthError",
            )
        ):
            return False
        return _orig_adk_should_retry_node(exception, retry_config, node_state)

    _adk_retry_utils._should_retry_node = _non_retryable_should_retry_node
except Exception:
    pass


class ResilientLiteLLMClient(LiteLLMClient):
    """LiteLLMClient with full jitter exponential backoff (min offset 5s, 1h patience) on 429/quota exhaustion."""

    async def acompletion(
        self,
        model: Any,
        messages: Any,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        import litellm

        max_patience = float(os.environ.get("MANTIS_LLM_MAX_PATIENCE_SECONDS", "3600.0"))
        initial_delay = float(os.environ.get("MANTIS_LLM_RETRY_INITIAL_DELAY", "5.0"))
        max_delay = float(os.environ.get("MANTIS_LLM_RETRY_MAX_DELAY", "60.0"))
        min_offset = float(os.environ.get("MANTIS_LLM_MIN_OFFSET", "5.0"))
        backoff_factor = float(os.environ.get("MANTIS_LLM_RETRY_BACKOFF", "2.0"))

        start_time = time.time()
        attempt = 0
        auth_refreshed = False
        while True:
            try:
                return await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    **kwargs,
                )
            except Exception as e:
                if is_auth_error(e):
                    if not auth_refreshed and is_token_refreshable_auth_error(e) and try_refresh_auth():
                        auth_refreshed = True
                        print(f"\n[AUTH REFRESH] Token refreshed for '{model}'; retrying request...", file=sys.stderr, flush=True)
                        continue
                    raise MantisAuthError(
                        format_auth_error_message(e, model=str(model)),
                        original_exception=e,
                    ) from None

                elapsed = time.time() - start_time
                if not is_retryable_llm_error(e) or elapsed >= max_patience:
                    raise

                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )

                attempt += 1
                model_name = str(model)
                if is_rate_limit_error(e):
                    err_detail = extract_rate_limit_detail(e)
                    prefix = f"[RATE LIMIT] 429 Quota Exceeded on '{model_name}'"
                else:
                    err_detail = f"{type(e).__name__}: {e}"
                    prefix = f"[LLM RETRY] Transient error on '{model_name}'"
                print(
                    f"\n{prefix}. "
                    f"Full jitter backoff: pausing {delay:.1f}s before retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(delay)

    def completion(
        self,
        model: Any,
        messages: Any,
        tools: Any = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        import litellm

        max_patience = float(os.environ.get("MANTIS_LLM_MAX_PATIENCE_SECONDS", "3600.0"))
        initial_delay = float(os.environ.get("MANTIS_LLM_RETRY_INITIAL_DELAY", "5.0"))
        max_delay = float(os.environ.get("MANTIS_LLM_RETRY_MAX_DELAY", "60.0"))
        min_offset = float(os.environ.get("MANTIS_LLM_MIN_OFFSET", "5.0"))
        backoff_factor = float(os.environ.get("MANTIS_LLM_RETRY_BACKOFF", "2.0"))

        start_time = time.time()
        attempt = 0
        auth_refreshed = False
        while True:
            try:
                return litellm.completion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=stream,
                    **kwargs,
                )
            except Exception as e:
                if is_auth_error(e):
                    if not auth_refreshed and is_token_refreshable_auth_error(e) and try_refresh_auth():
                        auth_refreshed = True
                        print(f"\n[AUTH REFRESH] Token refreshed for '{model}'; retrying request...", file=sys.stderr, flush=True)
                        continue
                    raise MantisAuthError(
                        format_auth_error_message(e, model=str(model)),
                        original_exception=e,
                    ) from None

                elapsed = time.time() - start_time
                if not is_retryable_llm_error(e) or elapsed >= max_patience:
                    raise

                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )

                attempt += 1
                model_name = str(model)
                if is_rate_limit_error(e):
                    err_detail = extract_rate_limit_detail(e)
                    prefix = f"[RATE LIMIT] 429 Quota Exceeded on '{model_name}'"
                else:
                    err_detail = f"{type(e).__name__}: {e}"
                    prefix = f"[LLM RETRY] Transient error on '{model_name}'"
                print(
                    f"\n{prefix}. "
                    f"Full jitter backoff: pausing {delay:.1f}s before retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)


class ResilientLiteLlm(LiteLlm):
    """LiteLlm wrapper with resilient full jitter retry (min 5s offset, 1h patience) on quota and rate limits."""

    llm_client: LiteLLMClient = Field(default_factory=ResilientLiteLLMClient, exclude=True)

    def __init__(self, model: str, **kwargs: Any) -> None:
        if LiteLlm is not object:
            super().__init__(model=model, **kwargs)
        else:
            self.model = model
        if getattr(self, "llm_client", None) is None:
            self.llm_client = ResilientLiteLLMClient()
        if "gemini" in str(model).lower():
            try:
                import litellm
                if getattr(litellm, "vertex_ai_safety_settings", None) is None:
                    litellm.vertex_ai_safety_settings = [
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    ]
            except Exception:
                pass

    @property
    def capabilities(self) -> Any:
        """Explicitly declare output_schema_and_tools=False.

        When models have both tools and an output schema (e.g. reproducer, reviewer, critic),
        this instructs ADK's _OutputSchemaRequestProcessor to inject SetModelResponseTool.
        The agent can then conclude and submit its final structured verdict cleanly via a tool
        call (set_model_response) instead of attempting to push JSON via probe shell commands
        or becoming trapped in thinking-token loops.
        """
        if LlmCapabilities is not None:
            return LlmCapabilities(output_schema_and_tools=False)
        return getattr(super(), "capabilities", None)

    @staticmethod
    def _inject_stage_turn_reminder(llm_request: Any) -> None:
        """Injects a termination reminder or reporting directive on every turn."""
        schema_cls = getattr(getattr(llm_request, "config", None), "response_schema", None)
        has_set_model_response = False
        tools_dict = getattr(llm_request, "tools_dict", None)
        if schema_cls is None:
            if isinstance(tools_dict, dict) and "set_model_response" in tools_dict:
                has_set_model_response = True
                schema_cls = getattr(tools_dict["set_model_response"], "output_schema", None)
        else:
            if isinstance(tools_dict, dict) and "set_model_response" in tools_dict:
                has_set_model_response = True

        has_report_findings = isinstance(tools_dict, dict) and "report_findings" in tools_dict

        if schema_cls is None and not has_report_findings:
            return

        reminder_entries: list[tuple[str, str]] = []

        if schema_cls is not None:
            schema_name = getattr(schema_cls, "__name__", str(schema_cls))
            if schema_name == "ReproVerdict":
                hint = '{"route": "success" | "failed_repro", "reason": "<explanation>"}'
            elif schema_name == "ReviewVerdict":
                hint = '{"route": "confirmed" | "false_positive", "reason": "<explanation>"}'
            elif schema_name == "CriticVerdict":
                hint = '{"route": "viable" | "non_viable", "reason": "<explanation>"}'
            else:
                hint = '{"route": "...", "reason": "..."}'

            reminder_header = f"[STAGE TURN REMINDER - {schema_name}]"
            if has_set_model_response:
                reminder = (
                    f"\n\n{reminder_header}: "
                    f"If your analysis or updates for this stage are complete, you MUST STOP calling other tools now. "
                    f"Do NOT invoke run_sandbox, write_file, or dummy shell commands to signal completion or test output. "
                    f"To conclude this stage and provide your result, submit your final verdict using the 'set_model_response' tool "
                    f"(e.g. set_model_response(route=..., reason=...)) or emit it directly as raw JSON response text conforming to {schema_name}: "
                    f"{hint}"
                )
            else:
                reminder = (
                    f"\n\n{reminder_header}: "
                    f"If your analysis or updates for this stage are complete, you MUST STOP calling tools now. "
                    f"Do NOT invoke any tools or shell commands to signal completion or test output. "
                    f"To conclude this stage and provide your result, emit your final verdict directly as raw JSON response text conforming to {schema_name}: "
                    f"{hint}"
                )
            reminder_entries.append((reminder_header, reminder))

        if has_report_findings:
            reporting_header = "[STAGE DIRECTIVE - RESEARCHER]"
            reporting_directive = (
                f"\n\n{reporting_header}: "
                f"When reporting security findings via 'report_findings', call 'report_findings' directly without "
                f"emitting intermediate conversational text, analysis essays, or markdown preambles prior to the tool call. "
                f"This ensures your entire output token budget is preserved for the full structured findings payload."
            )
            reminder_entries.append((reporting_header, reporting_directive))

        contents = getattr(llm_request, "contents", None)
        if not contents:
            return

        last_content = contents[-1]
        parts = getattr(last_content, "parts", None)
        if not parts:
            return

        fr_parts = [p for p in parts if getattr(p, "function_response", None) is not None]
        if fr_parts:
            target_fr_part = fr_parts[-1]
            old_fr = target_fr_part.function_response
            resp = getattr(old_fr, "response", None)
            needed = [txt for hdr, txt in reminder_entries if not (resp and hdr in str(resp))]
            if not needed:
                return
            full_reminder = "".join(needed)

            if isinstance(resp, dict):
                new_resp = dict(resp)
                if "output" in new_resp and isinstance(new_resp["output"], str):
                    new_resp["output"] = f"{new_resp['output']}{full_reminder}"
                elif "result" in new_resp and isinstance(new_resp["result"], str):
                    new_resp["result"] = f"{new_resp['result']}{full_reminder}"
                else:
                    new_resp["_stage_turn_reminder"] = full_reminder.strip()
            elif isinstance(resp, str):
                new_resp = f"{resp}{full_reminder}"
            else:
                new_resp = {"output": full_reminder.strip()}

            from google.genai import types
            idx = parts.index(target_fr_part)
            parts[idx] = types.Part(
                function_response=types.FunctionResponse(
                    name=getattr(old_fr, "name", "tool"),
                    response=new_resp,
                    id=getattr(old_fr, "id", None),
                )
            )
        elif getattr(last_content, "role", None) == "user":
            text_parts = [p for p in parts if getattr(p, "text", None)]
            if text_parts:
                target_p = text_parts[-1]
                needed = [txt for hdr, txt in reminder_entries if hdr not in str(target_p.text)]
                if not needed:
                    return
                full_reminder = "".join(needed)
                idx = parts.index(target_p)
                from google.genai import types
                parts[idx] = types.Part.from_text(text=f"{target_p.text}{full_reminder}")

    @staticmethod
    def _inject_active_findings_state(llm_request: Any) -> None:
        """Injects live canonical state store findings into the prompt so agents are immune to context compaction."""
        from core.context import current_run_context
        from core.database import read_findings

        ctx = current_run_context.get()
        if ctx is None or not ctx.db_path or not os.path.exists(ctx.db_path):
            return

        try:
            findings = read_findings(ctx.db_path, run_id=ctx.run_id)
        except Exception:
            return

        if not findings:
            return

        state_header = "[STATE STORE: RECORDED FINDINGS (DO NOT RE-REPORT)]"
        contents = getattr(llm_request, "contents", None)
        if not contents:
            return

        for c in contents:
            for p in getattr(c, "parts", []):
                if state_header in str(getattr(p, "text", "")):
                    return
                fr = getattr(p, "function_response", None)
                if fr and state_header in str(getattr(fr, "response", "")):
                    return

        from core.llm_gateway import SecretScrubber

        lines = []
        for f in findings:
            f_id = f.get("id")
            f_sev = f.get("severity", "UNKNOWN")
            f_fp = f.get("filepath", "")
            f_lines = f.get("line_numbers") or []
            raw_title = str(f.get("title", "")).replace("\n", " ").strip()
            clean_title = SecretScrubber.scrub(raw_title)[:120]
            f_st = f.get("status", "reported")
            lines.append(f"  • Finding #{f_id} [{f_sev}] {f_fp}:{f_lines} - {clean_title} (status: {f_st})")

        state_block = (
            f"\n\n{state_header}\n"
            f"The following {len(findings)} vulnerability finding(s) have ALREADY been recorded in knowledge.db for this run.\n"
            f"Do NOT call report_findings for these existing findings. If analyzing new code, only report NEW distinct findings:\n"
            + "\n".join(lines)
        )

        last_content = contents[-1]
        parts = getattr(last_content, "parts", None)
        if not parts:
            return

        from google.genai import types

        fr_parts = [p for p in parts if getattr(p, "function_response", None) is not None]
        if fr_parts:
            target_fr_part = fr_parts[-1]
            old_fr = target_fr_part.function_response
            resp = getattr(old_fr, "response", None)
            if isinstance(resp, dict):
                new_resp = dict(resp)
                if "output" in new_resp and isinstance(new_resp["output"], str):
                    new_resp["output"] = f"{new_resp['output']}{state_block}"
                elif "result" in new_resp and isinstance(new_resp["result"], str):
                    new_resp["result"] = f"{new_resp['result']}{state_block}"
                else:
                    new_resp["_state_store_findings"] = state_block.strip()
            elif isinstance(resp, str):
                new_resp = f"{resp}{state_block}"
            else:
                new_resp = {"output": state_block.strip()}

            idx = parts.index(target_fr_part)
            parts[idx] = types.Part(
                function_response=types.FunctionResponse(
                    name=getattr(old_fr, "name", "tool"),
                    response=new_resp,
                    id=getattr(old_fr, "id", None),
                )
            )
        elif getattr(last_content, "role", None) == "user":
            text_parts = [p for p in parts if getattr(p, "text", None)]
            if text_parts:
                target_p = text_parts[-1]
                idx = parts.index(target_p)
                parts[idx] = types.Part.from_text(text=f"{target_p.text}{state_block}")

    @staticmethod
    def _sanitize_structured_response(response: Any, schema_cls: Any) -> Any:
        """Sanitizes model responses when a structured schema is required to prevent Pydantic ValidationError on refusals."""
        if schema_cls is None or getattr(response, "partial", False):
            return response

        # If the response contains function calls (e.g. set_model_response or tool calls), let ADK handle it
        parts = getattr(getattr(response, "content", None), "parts", None)
        if parts:
            if any(getattr(p, "function_call", None) is not None for p in parts):
                return response

        # Extract text content from parts
        text = ""
        if parts:
            text = "".join(
                getattr(p, "text", "") or ""
                for p in parts
                if not getattr(p, "thought", False)
            )

        # Check if the text is valid JSON according to schema_cls
        is_valid = False
        clean_text = text.strip()
        if clean_text:
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            try:
                if hasattr(schema_cls, "model_validate_json"):
                    schema_cls.model_validate_json(clean_text)
                    is_valid = True
                elif hasattr(schema_cls, "validate_json"):
                    schema_cls.validate_json(clean_text)
                    is_valid = True
            except Exception:
                is_valid = False

        from google.genai import types
        is_safety_block = getattr(response, "finish_reason", None) in (types.FinishReason.SAFETY, "SAFETY") or getattr(response, "error_code", None) == "SAFETY"
        if is_safety_block:
            is_valid = False

        if not is_valid:
            schema_name = getattr(schema_cls, "__name__", str(schema_cls))
            clean_snippet = text.strip().replace("\n", " ")[:200] or "Model emitted non-JSON or refusal output"
            if schema_name == "ReproVerdict":
                fallback_payload = {
                    "route": "failed_repro",
                    "reason": f"Fallback: {clean_snippet}",
                }
            elif schema_name == "ReviewVerdict":
                fallback_payload = {
                    "route": "confirmed",
                    "reason": f"Fallback: {clean_snippet}",
                }
            elif schema_name == "CriticVerdict":
                fallback_payload = {
                    "route": "non_viable",
                    "reason": f"Fallback: {clean_snippet}",
                }
            else:
                fallback_payload = {
                    "route": "failed_repro",
                    "reason": f"Fallback: {clean_snippet}",
                }

            fallback_json = json.dumps(fallback_payload)
            if getattr(response, "content", None) is None:
                response.content = types.Content(role="model", parts=[types.Part.from_text(text=fallback_json)])
            else:
                response.content.parts = [types.Part.from_text(text=fallback_json)]

            if getattr(response, "finish_reason", None) in (types.FinishReason.SAFETY, "SAFETY"):
                response.finish_reason = types.FinishReason.STOP
            if getattr(response, "error_code", None):
                response.error_code = None
                response.error_message = None

            print(
                f"\n[SCHEMA RESILIENCE] Intercepted non-schema/refusal output for {schema_name}; "
                f"synthesized fallback: {fallback_json}",
                file=sys.stderr,
                flush=True,
            )

        return response

    @staticmethod
    def _is_empty_stop_turn(response: Any) -> bool:
        """Returns True for a non-streaming turn that finished with STOP but zero content parts.

        ADK treats this exact case (base_llm_flow `_postprocess_async`) as a fatal
        ``MODEL_RETURNED_NO_CONTENT`` event error that otherwise aborts the campaign.
        We detect it here, before ADK sees it, so the resilience wrapper can retry the
        turn instead of failing the workflow. Function-call turns (legitimate content
        in a different part type) are never flagged.
        """
        if getattr(response, "partial", False):
            return False
        from google.genai import types
        if getattr(response, "finish_reason", None) not in (types.FinishReason.STOP, "STOP", "stop"):
            return False
        parts = getattr(getattr(response, "content", None), "parts", None)
        if not parts:
            return True
        for part in parts:
            if (
                getattr(part, "text", None)
                or getattr(part, "function_call", None) is not None
                or getattr(part, "function_response", None) is not None
                or getattr(part, "thought", None)
            ):
                return False
        return True

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> Any:
        self._inject_stage_turn_reminder(llm_request)
        self._inject_active_findings_state(llm_request)
        schema_cls = getattr(getattr(llm_request, "config", None), "response_schema", None)
        if schema_cls is None and isinstance(getattr(llm_request, "tools_dict", None), dict):
            if "set_model_response" in llm_request.tools_dict:
                schema_cls = getattr(llm_request.tools_dict["set_model_response"], "output_schema", None)

        max_patience = float(os.environ.get("MANTIS_LLM_MAX_PATIENCE_SECONDS", "3600.0"))
        initial_delay = float(os.environ.get("MANTIS_LLM_RETRY_INITIAL_DELAY", "5.0"))
        max_delay = float(os.environ.get("MANTIS_LLM_RETRY_MAX_DELAY", "60.0"))
        min_offset = float(os.environ.get("MANTIS_LLM_MIN_OFFSET", "5.0"))
        backoff_factor = float(os.environ.get("MANTIS_LLM_RETRY_BACKOFF", "2.0"))

        # Empty-turn retries are budgeted separately and tightly. ADK emits
        # MODEL_RETURNED_NO_CONTENT when a non-streaming turn ends STOP with no
        # content parts; for a given conversation state that is usually
        # DETERMINISTIC (DeepSeek/Ollama sometimes swallows all output on
        # think=true), so retrying it hundreds of times under the 1h transient
        # patience only hangs the campaign. Keep it to a few short attempts, then
        # fail gracefully so ADK's node-level retry/resume can steer around it.
        empty_max_attempts = int(os.environ.get("MANTIS_EMPTY_TURN_MAX_ATTEMPTS", "3"))
        empty_initial_delay = float(os.environ.get("MANTIS_EMPTY_TURN_INITIAL_DELAY", "3.0"))
        empty_max_delay = float(os.environ.get("MANTIS_EMPTY_TURN_MAX_DELAY", "15.0"))
        empty_min_offset = float(os.environ.get("MANTIS_EMPTY_TURN_MIN_OFFSET", "1.0"))
        empty_backoff_factor = float(os.environ.get("MANTIS_EMPTY_TURN_BACKOFF", "1.5"))

        start_time = time.time()
        attempt = 0
        auth_refreshed = False
        current_stream = stream
        empty_attempts = 0
        while True:
            try:
                from google.genai import types as genai_types
                empty_turn_seen = False
                async for response in super().generate_content_async(llm_request, stream=current_stream):
                    if (
                        current_stream
                        and (
                            getattr(response, "error_code", None) in (getattr(genai_types.FinishReason, "MAX_TOKENS", None), "MAX_TOKENS")
                            or getattr(response, "finish_reason", None) in (getattr(genai_types.FinishReason, "MAX_TOKENS", None), "MAX_TOKENS")
                        )
                        and "Tool call arguments were truncated" in str(getattr(response, "error_message", ""))
                    ):
                        raise MantisStreamingTruncationError(str(getattr(response, "error_message", "")))
                    if not current_stream and self._is_empty_stop_turn(response):
                        # A non-streaming turn that ends with STOP but no content is emitted
                        # as a fatal MODEL_RETURNED_NO_CONTENT event by ADK's postprocess.
                        # Retry a small bounded number of times (it is usually
                        # deterministic for a given conversation state), then surface
                        # a non-retryable error rather than hang under the 1h patience.
                        if empty_attempts >= empty_max_attempts:
                            # Non-retryable: an identical request that repeatedly
                            # returns empty will not recover, and retrying it under
                            # the 1h transient patience only hangs the campaign.
                            raise MantisEmptyTurnExhaustedError(
                                "Model returned no content (finish_reason=STOP with empty parts) "
                                f"after {empty_attempts} empty-turn attempts; giving up on this request."
                            )
                        empty_attempts += 1
                        empty_turn_seen = True
                        empty_delay = compute_full_jitter_delay(
                            attempt=empty_attempts,
                            initial_delay=empty_initial_delay,
                            max_delay=empty_max_delay,
                            min_offset=empty_min_offset,
                            backoff_factor=empty_backoff_factor,
                        )
                        model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                        print(
                            f"\n[EMPTY TURN RETRY] '{model_name}' returned no content "
                            f"(finish_reason=STOP, no parts) on attempt {empty_attempts}/{empty_max_attempts}. "
                            f"Pausing {empty_delay:.1f}s before retrying...",
                            file=sys.stderr,
                            flush=True,
                        )
                        await asyncio.sleep(empty_delay)
                        # Leave the generator to re-issue the request on the while
                        # loop; mark that the empty turn (not a streamed response)
                        # is what ended this try.
                        break
                    yield self._sanitize_structured_response(response, schema_cls)
                if empty_turn_seen:
                    continue
                return
            except (json.decoder.JSONDecodeError, ValueError, MantisStreamingTruncationError) as e:
                if current_stream:
                    model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                    print(
                        f"\n[STREAM RESILIENCE] Streaming tool call parsing failed on '{model_name}' ({type(e).__name__}: {e}). "
                        f"Retrying turn with stream=False...",
                        file=sys.stderr,
                        flush=True,
                    )
                    current_stream = False
                    continue
                if not is_retryable_llm_error(e):
                    raise
                elapsed = time.time() - start_time
                if elapsed >= max_patience:
                    raise
                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )
                attempt += 1
                model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                err_detail = f"{type(e).__name__}: {e}"
                print(
                    f"\n[LLM RETRY] Transient error on '{model_name}'. "
                    f"Full jitter backoff: pausing {delay:.1f}s before retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                if current_stream and any(p in str(e).lower() for p in ("unterminated string", "jsondecodeerror", "truncated while streaming")):
                    model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                    print(
                        f"\n[STREAM RESILIENCE] Streaming tool call parsing failed on '{model_name}' ({type(e).__name__}: {e}). "
                        f"Retrying turn with stream=False...",
                        file=sys.stderr,
                        flush=True,
                    )
                    current_stream = False
                    continue

                if is_auth_error(e):
                    if not auth_refreshed and is_token_refreshable_auth_error(e) and try_refresh_auth():
                        auth_refreshed = True
                        model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                        print(f"\n[AUTH REFRESH] Token refreshed for '{model_name}'; retrying request...", file=sys.stderr, flush=True)
                        continue
                    raise MantisAuthError(
                        format_auth_error_message(
                            e,
                            model=str(getattr(self, "model", getattr(llm_request, "model", "llm"))),
                        ),
                        original_exception=e,
                    ) from None

                elapsed = time.time() - start_time
                if not is_retryable_llm_error(e) or elapsed >= max_patience:
                    raise

                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )

                attempt += 1
                model_name = str(getattr(self, "model", llm_request.model if hasattr(llm_request, "model") else "llm"))
                if is_rate_limit_error(e):
                    err_detail = extract_rate_limit_detail(e)
                    prefix = f"[RATE LIMIT] 429 Quota Exceeded on '{model_name}'"
                else:
                    err_detail = f"{type(e).__name__}: {e}"
                    prefix = f"[LLM RETRY] Transient error on '{model_name}'"
                print(
                    f"\n{prefix}. "
                    f"Full jitter backoff: pausing {delay:.1f}s before retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(delay)


def get_llm_kwargs(
    model_id: Optional[str] = None,
    default_model: str = DEFAULT_MODEL,
    api_base: Optional[str] = None,
    default_api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    default_timeout: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    default_reasoning_effort: Optional[str] = None,
    global_model_override: Optional[str] = None,
    config: Optional[dict] = None,
) -> Tuple[str, dict]:
    """Resolves the LLM mapping details cleanly with precedence: global_override > MANTIS_MODEL > node > MODEL_ID > default."""
    if global_model_override:
        raw_model = global_model_override
    elif os.environ.get("MANTIS_MODEL"):
        raw_model = os.environ.get("MANTIS_MODEL")
    else:
        raw_model = model_id or os.environ.get("MODEL_ID") or default_model

    # A model id carrying the ':cloud' suffix (e.g. ollama/deepseek-v4-flash:cloud)
    # is an Ollama-hosted model and must reach the Ollama Cloud endpoint, not the
    # local daemon. Honor both the 'ollama.cloud/' prefix and the ':cloud' suffix.
    raw_is_ollama_cloud = (
        raw_model.strip().startswith(OLLAMA_CLOUD_PREFIX)
        or raw_model.strip().endswith(":cloud")
    )
    # Whether this cloud model flows through the OpenAI provider (ollama.cloud/ alias
    # rewritten to openai/<m>) vs. the native Ollama provider (:cloud suffix).
    raw_is_ollama_cloud_openai = raw_model.strip().startswith(OLLAMA_CLOUD_PREFIX)
    resolved_model = normalize_model_id(raw_model)

    # config (global config dict, e.g. from workflow.json) may carry an api_base.
    config_api_base = None
    if config and isinstance(config, dict):
        config_api_base = config.get("api_base")
        if isinstance(config.get("config"), dict):
            config_api_base = config_api_base or config["config"].get("api_base")
        if isinstance(config.get("sandbox"), dict):
            sb_opts = config["sandbox"].get("options")
            if isinstance(sb_opts, dict) and not config_api_base:
                config_api_base = sb_opts.get("api_base")
        if config_api_base and _is_unconfigured_api_base(config_api_base):
            config_api_base = None

    # Resolve api_base: explicit > config > LLM_API_BASE > default_api_base > provider default.
    #  - Local Ollama defaults to the local daemon's OpenAI-compatible /v1.
    #  - Ollama Cloud (/v1-compatible hosted) defaults to https://ollama.com/v1.
    #  - Generic openai/ uses any configured base (vLLM, LM Studio, Ollama Cloud).
    resolved_api_base = api_base or config_api_base or os.environ.get("LLM_API_BASE") or default_api_base
    if not resolved_api_base:
        if raw_is_ollama_cloud:
            # ollama.cloud/* flows via the OpenAI provider (appends /chat/completions)
            # -> /v1 base. :cloud suffix flows via the native Ollama provider
            # (appends /api/generate) -> no /v1 base.
            resolved_api_base = (
                OLLAMA_CLOUD_OPENAI_BASE if raw_is_ollama_cloud_openai else OLLAMA_CLOUD_API_BASE
            )
        elif resolved_model.startswith("ollama/"):
            resolved_api_base = DEFAULT_API_BASE

    # Inform the operator that the endpoint was auto-resolved for this model and
    # how to override it (or drop back to a local model). Printed once per resolve;
    # harmless in normal runs, useful when debugging LLM connectivity.
    if os.environ.get("MANTIS_DEBUG_LLM_BASE") in ("1", "true", "True"):
        if raw_is_ollama_cloud and not (api_base or config_api_base or os.environ.get("LLM_API_BASE")):
            print(
                f"[LLM BASE] '{resolved_model}' is an Ollama Cloud model; auto-using "
                f"api_base={resolved_api_base}. Override with LLM_API_BASE, workflow "
                f"config 'api_base', or use a non-':cloud' model for the local daemon "
                f"({DEFAULT_API_BASE}).",
                file=sys.stderr,
                flush=True,
            )

    raw_timeout = timeout if timeout is not None else (
        os.environ.get("LLM_TIMEOUT")
        or os.environ.get("MANTIS_TIMEOUT")
        or os.environ.get("LLM_REQUEST_TIMEOUT")
        or default_timeout
    )
    effort = reasoning_effort or os.environ.get("REASONING_EFFORT") or default_reasoning_effort
    if raw_timeout is None and (effort in ("high", "medium") or "claude" in resolved_model):
        raw_timeout = 300.0

    llm_kwargs = {"model": resolved_model}
    if resolved_api_base:
        llm_kwargs["api_base"] = resolved_api_base

    # Inject provider API keys for hosted/cloud endpoints (never send secrets with
    # credentials to a local daemon). Keys come from env or the git-ignored .env
    # file loaded at module import. Order: model-specific generic key fallback.
    _key = None
    if raw_is_ollama_cloud or raw_is_ollama_cloud_openai:
        # Ollama Cloud requires an API key.
        _key = (
            os.environ.get("OLLAMA_API_KEY")
            or os.environ.get("OLLAMA_CLOUD_API_KEY")
            or ""
        )
    elif resolved_model.startswith("openai/"):
        _key = os.environ.get("OPENAI_API_KEY") or ""
    elif resolved_model.startswith("anthropic/"):
        _key = os.environ.get("ANTHROPIC_API_KEY") or ""
    if _key:
        llm_kwargs["api_key"] = _key

    if effort:
        llm_kwargs["reasoning_effort"] = str(effort).lower().strip()
    if raw_timeout is not None:
        try:
            timeout_val = float(raw_timeout)
            if timeout_val > 0:
                llm_kwargs["timeout"] = timeout_val
        except (ValueError, TypeError):
            pass

    max_tokens_val = os.environ.get("LLM_MAX_TOKENS") or os.environ.get("MANTIS_MAX_TOKENS")
    if max_tokens_val:
        try:
            llm_kwargs["max_tokens"] = int(max_tokens_val)
        except (ValueError, TypeError):
            pass
    elif effort in ("high", "medium") or "claude" in resolved_model:
        # Provide a generous output token budget so thinking tokens do not starve response text
        llm_kwargs.setdefault("max_tokens", 32768)

    return resolved_model, llm_kwargs

