"""Langfuse and OpenTelemetry tracing setup for Pipecat observability.

Optional by default for local WebRTC runs: missing keys log a notice and return
None without breaking the agent. If required=True is passed, missing keys or
failed initialization will raise ValueError or RuntimeError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping
from loguru import logger


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse and OpenTelemetry tracing."""

    public_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = None
    session_id: str | None = None
    environment: str | None = None
    release: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_environment(
        cls,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> LangfuseConfig:
        """Load configuration from environment variables."""
        public_key = (
            os.getenv("LANGFUSE_PUBLIC_KEY")
            or os.getenv("PIPECAT_LANGFUSE_PUBLIC_KEY")
        )
        secret_key = (
            os.getenv("LANGFUSE_SECRET_KEY")
            or os.getenv("PIPECAT_LANGFUSE_SECRET_KEY")
        )
        base_url = (
            os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or os.getenv("PIPECAT_LANGFUSE_BASE_URL")
            or os.getenv("PIPECAT_LANGFUSE_HOST")
        )
        environment = os.getenv("LANGFUSE_ENVIRONMENT") or os.getenv("ENVIRONMENT")
        release = os.getenv("LANGFUSE_RELEASE") or os.getenv("RELEASE")

        meta = dict(metadata or {})
        return cls(
            public_key=public_key.strip() if public_key else None,
            secret_key=secret_key.strip() if secret_key else None,
            base_url=base_url.strip() if base_url else None,
            session_id=session_id,
            environment=environment,
            release=release,
            metadata=meta,
            tags=list(tags or []),
        )

    def is_configured(self) -> bool:
        """Return True if all required Langfuse keys (public_key, secret_key, base_url) are present."""
        return bool(self.public_key and self.secret_key and self.base_url)

    def missing_keys(self) -> list[str]:
        """Return list of missing required key names."""
        missing = []
        if not self.public_key:
            missing.append("LANGFUSE_PUBLIC_KEY")
        if not self.secret_key:
            missing.append("LANGFUSE_SECRET_KEY")
        if not self.base_url:
            missing.append("LANGFUSE_BASE_URL (or LANGFUSE_HOST)")
        return missing


class LangfuseTracer:
    """Wrapper holding initialized Langfuse client and OpenTelemetry tracer provider."""

    def __init__(
        self,
        config: LangfuseConfig,
        client: Any = None,
        tracer_provider: Any = None,
    ) -> None:
        self.config = config
        self.client = client
        self.tracer_provider = tracer_provider

    def flush(self) -> bool:
        """Flush pending spans/events to Langfuse."""
        success = True
        if self.client is not None and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception as exc:
                logger.warning(f"Langfuse client flush failed: {exc}")
                success = False

        if self.tracer_provider is not None and hasattr(self.tracer_provider, "force_flush"):
            try:
                self.tracer_provider.force_flush()
            except Exception as exc:
                logger.warning(f"TracerProvider force_flush failed: {exc}")
                success = False

        return success

    def shutdown(self) -> bool:
        """Shutdown tracer provider and flush client."""
        flushed = self.flush()
        if self.tracer_provider is not None and hasattr(self.tracer_provider, "shutdown"):
            try:
                self.tracer_provider.shutdown()
            except Exception as exc:
                logger.warning(f"TracerProvider shutdown failed: {exc}")
                return False
        return flushed


def setup_langfuse(
    session_id: str | None = None,
    *,
    config: LangfuseConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    base_url: str | None = None,
    tags: list[str] | None = None,
    required: bool = False,
) -> LangfuseTracer | None:
    """Initialize Langfuse and OpenTelemetry tracing.

    Optional by default: if keys are missing and required=False, logs an info
    message and returns None. If required=True, raises ValueError on missing keys
    or RuntimeError if initialization fails.
    """
    cfg = config or LangfuseConfig.from_environment(
        session_id=session_id,
        metadata=metadata,
        tags=tags,
    )

    # Parameter overrides if provided
    if public_key is not None:
        cfg.public_key = public_key
    if secret_key is not None:
        cfg.secret_key = secret_key
    if base_url is not None:
        cfg.base_url = base_url
    if session_id is not None:
        cfg.session_id = session_id

    if not cfg.is_configured():
        missing = ", ".join(cfg.missing_keys())
        msg = f"Langfuse tracing configuration incomplete. Missing: {missing}"
        if required:
            raise ValueError(msg)
        logger.info(f"{msg}; continuing without Langfuse tracing.")
        return None

    try:
        from langfuse import Langfuse  # type: ignore
    except ImportError as exc:
        if required:
            raise RuntimeError(
                "Langfuse is configured as required, but the 'langfuse' package is not installed."
            ) from exc
        logger.warning("Langfuse keys are present but 'langfuse' package is not installed; skipping tracing.")
        return None

    try:
        # Initialize OpenTelemetry TracerProvider if available
        tracer_provider = None
        try:
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore
            from opentelemetry import trace  # type: ignore

            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)
        except ImportError:
            pass

        langfuse_kwargs: dict[str, Any] = {
            "public_key": cfg.public_key,
            "secret_key": cfg.secret_key,
            "base_url": cfg.base_url,
        }
        if tracer_provider is not None:
            langfuse_kwargs["tracer_provider"] = tracer_provider
            langfuse_kwargs["should_export_span"] = lambda span: True
        if cfg.environment:
            langfuse_kwargs["environment"] = cfg.environment
        if cfg.release:
            langfuse_kwargs["release"] = cfg.release

        client = Langfuse(**langfuse_kwargs)
        logger.info(f"Langfuse tracing initialized successfully (session_id={cfg.session_id or 'none'}).")
        return LangfuseTracer(config=cfg, client=client, tracer_provider=tracer_provider)
    except Exception as exc:
        if required:
            raise RuntimeError(f"Failed to initialize Langfuse: {exc}") from exc
        logger.warning(f"Error initializing Langfuse tracing: {exc}; continuing without tracing.")
        return None
