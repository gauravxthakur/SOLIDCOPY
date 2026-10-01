import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.langfuse import LangfuseConfig, LangfuseTracer, setup_langfuse


class LangfuseConfigTests(unittest.TestCase):
    def setUp(self):
        self.env_keys = [
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_BASE_URL",
            "LANGFUSE_HOST",
            "LANGFUSE_ENVIRONMENT",
            "LANGFUSE_RELEASE",
            "PIPECAT_LANGFUSE_PUBLIC_KEY",
            "PIPECAT_LANGFUSE_SECRET_KEY",
            "PIPECAT_LANGFUSE_BASE_URL",
            "PIPECAT_LANGFUSE_HOST",
            "ENVIRONMENT",
            "RELEASE",
        ]
        for k in self.env_keys:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in self.env_keys:
            os.environ.pop(k, None)

    def test_default_config_is_not_configured(self):
        cfg = LangfuseConfig()
        self.assertFalse(cfg.is_configured())
        missing = cfg.missing_keys()
        self.assertIn("LANGFUSE_PUBLIC_KEY", missing)
        self.assertIn("LANGFUSE_SECRET_KEY", missing)
        self.assertIn("LANGFUSE_BASE_URL (or LANGFUSE_HOST)", missing)

    def test_partial_config_missing_keys(self):
        cfg = LangfuseConfig(public_key="pk-123", base_url="http://localhost:3000")
        self.assertFalse(cfg.is_configured())
        self.assertEqual(cfg.missing_keys(), ["LANGFUSE_SECRET_KEY"])

    def test_full_config_is_configured(self):
        cfg = LangfuseConfig(
            public_key="pk-123",
            secret_key="sk-456",
            base_url="https://cloud.langfuse.com",
            session_id="sess-789",
        )
        self.assertTrue(cfg.is_configured())
        self.assertEqual(len(cfg.missing_keys()), 0)

    def test_from_environment_standard_vars(self):
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-env"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-env"
        os.environ["LANGFUSE_BASE_URL"] = "https://cloud.langfuse.com"
        os.environ["LANGFUSE_ENVIRONMENT"] = "staging"
        os.environ["LANGFUSE_RELEASE"] = "v1.2.3"

        cfg = LangfuseConfig.from_environment(session_id="s1")
        self.assertTrue(cfg.is_configured())
        self.assertEqual(cfg.public_key, "pk-env")
        self.assertEqual(cfg.secret_key, "sk-env")
        self.assertEqual(cfg.base_url, "https://cloud.langfuse.com")
        self.assertEqual(cfg.environment, "staging")
        self.assertEqual(cfg.release, "v1.2.3")
        self.assertEqual(cfg.session_id, "s1")

    def test_from_environment_pipecat_prefixed_vars(self):
        os.environ["PIPECAT_LANGFUSE_PUBLIC_KEY"] = "pk-pipe"
        os.environ["PIPECAT_LANGFUSE_SECRET_KEY"] = "sk-pipe"
        os.environ["PIPECAT_LANGFUSE_HOST"] = "http://localhost:3000"

        cfg = LangfuseConfig.from_environment()
        self.assertTrue(cfg.is_configured())
        self.assertEqual(cfg.public_key, "pk-pipe")
        self.assertEqual(cfg.secret_key, "sk-pipe")
        self.assertEqual(cfg.base_url, "http://localhost:3000")


class SetupLangfuseTests(unittest.TestCase):
    def setUp(self):
        self.env_keys = [
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_BASE_URL",
            "LANGFUSE_HOST",
        ]
        for k in self.env_keys:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in self.env_keys:
            os.environ.pop(k, None)

    def test_optional_by_default_returns_none_when_keys_missing(self):
        tracer = setup_langfuse(session_id="test-session")
        self.assertIsNone(tracer)

    def test_raises_when_required_and_keys_missing(self):
        with self.assertRaises(ValueError) as ctx:
            setup_langfuse(session_id="test-session", required=True)
        self.assertIn("Langfuse tracing configuration incomplete", str(ctx.exception))

    def test_explicit_params_override_env(self):
        # Even with empty env, passing explicit keys configures it
        cfg = LangfuseConfig(
            public_key="pk-explicit",
            secret_key="sk-explicit",
            base_url="https://cloud.langfuse.com",
        )
        self.assertTrue(cfg.is_configured())

    def test_setup_with_mocked_langfuse(self):
        mock_langfuse_cls = MagicMock()
        mock_instance = MagicMock()
        mock_langfuse_cls.return_value = mock_instance

        mock_module = MagicMock()
        mock_module.Langfuse = mock_langfuse_cls

        with patch.dict("sys.modules", {"langfuse": mock_module}):
            tracer = setup_langfuse(
                session_id="session-123",
                public_key="pk-test",
                secret_key="sk-test",
                base_url="https://cloud.langfuse.com",
            )
            self.assertIsNotNone(tracer)
            self.assertIsInstance(tracer, LangfuseTracer)
            self.assertEqual(tracer.config.session_id, "session-123")
            mock_langfuse_cls.assert_called_once()
            call_kwargs = mock_langfuse_cls.call_args[1]
            self.assertEqual(call_kwargs["public_key"], "pk-test")
            self.assertEqual(call_kwargs["secret_key"], "sk-test")
            self.assertEqual(call_kwargs["base_url"], "https://cloud.langfuse.com")

    def test_tracer_flush_and_shutdown(self):
        mock_client = MagicMock()
        mock_provider = MagicMock()

        tracer = LangfuseTracer(
            config=LangfuseConfig(public_key="pk", secret_key="sk", base_url="url"),
            client=mock_client,
            tracer_provider=mock_provider,
        )

        res = tracer.flush()
        self.assertTrue(res)
        mock_client.flush.assert_called_once()
        mock_provider.force_flush.assert_called_once()

        res_sd = tracer.shutdown()
        self.assertTrue(res_sd)
        mock_provider.shutdown.assert_called_once()

    def test_tracer_flush_failure_handles_gracefully(self):
        mock_client = MagicMock()
        mock_client.flush.side_effect = RuntimeError("network down")

        tracer = LangfuseTracer(
            config=LangfuseConfig(public_key="pk", secret_key="sk", base_url="url"),
            client=mock_client,
        )
        res = tracer.flush()
        self.assertFalse(res)

    def test_initialization_failure_raises_if_required(self):
        mock_langfuse_cls = MagicMock(side_effect=Exception("Auth failed"))
        mock_module = MagicMock()
        mock_module.Langfuse = mock_langfuse_cls

        with patch.dict("sys.modules", {"langfuse": mock_module}):
            with self.assertRaises(RuntimeError) as ctx:
                setup_langfuse(
                    public_key="pk-test",
                    secret_key="sk-test",
                    base_url="https://cloud.langfuse.com",
                    required=True,
                )
            self.assertIn("Failed to initialize Langfuse", str(ctx.exception))

    def test_initialization_failure_returns_none_if_optional(self):
        mock_langfuse_cls = MagicMock(side_effect=Exception("Auth failed"))
        mock_module = MagicMock()
        mock_module.Langfuse = mock_langfuse_cls

        with patch.dict("sys.modules", {"langfuse": mock_module}):
            tracer = setup_langfuse(
                public_key="pk-test",
                secret_key="sk-test",
                base_url="https://cloud.langfuse.com",
                required=False,
            )
            self.assertIsNone(tracer)

    def test_missing_langfuse_package_raises_if_required(self):
        # When langfuse module cannot be imported and required=True
        with patch.dict("sys.modules", {"langfuse": None}):
            with self.assertRaises(RuntimeError) as ctx:
                setup_langfuse(
                    public_key="pk-test",
                    secret_key="sk-test",
                    base_url="https://cloud.langfuse.com",
                    required=True,
                )
            self.assertIn("langfuse", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

