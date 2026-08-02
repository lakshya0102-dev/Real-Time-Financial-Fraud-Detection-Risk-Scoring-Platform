"""Configuration module — single source of truth for all platform settings."""

from src.config.settings import Settings, get_settings

__all__ = ["get_settings", "Settings"]
