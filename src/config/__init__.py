"""Configuration module — single source of truth for all platform settings."""

from src.config.settings import get_settings, Settings

__all__ = ["get_settings", "Settings"]
