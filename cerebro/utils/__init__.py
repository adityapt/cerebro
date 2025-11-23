"""Utility functions"""

from cerebro.utils.logging import setup_logger, get_logger, logger
from cerebro.utils.process_manager import (
    kill_existing_mlx_processes,
    ensure_single_instance,
    cleanup_on_exit,
    check_system_resources
)

__all__ = [
    "setup_logger",
    "get_logger",
    "logger",
    "kill_existing_mlx_processes",
    "ensure_single_instance",
    "cleanup_on_exit",
    "check_system_resources"
]

