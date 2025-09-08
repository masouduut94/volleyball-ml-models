"""
Simple colorful logger for ML Manager.
"""

import logging
import sys


class ColorFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[91m',  # Bright Red
        'SUCCESS': '\033[92m',  # Bright Green
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        # Add color to both the level name and the message
        levelname = record.levelname
        if levelname in self.COLORS:
            color = self.COLORS[levelname]
            reset = self.COLORS['RESET']
            # Color the entire log message
            record.levelname = f"{color}{levelname}"
            record.msg = f"{record.msg}{reset}"

        return super().format(record)


def get_logger(name: str = "ml_manager", level: str = "INFO") -> logging.Logger:
    """
    Get a colorful logger instance.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Don't add handlers if they already exist
    if logger.handlers:
        return logger

    # Set level
    logger.setLevel(getattr(logging, level.upper()))

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = ColorFormatter(
        fmt='%(levelname)s: %(message)s'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


# Add custom log level for SUCCESS
logging.addLevelName(25, 'SUCCESS')


def success(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)


logging.Logger.success = success

# Global logger instance
logger = get_logger()
