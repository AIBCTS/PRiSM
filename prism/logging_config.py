import logging
import sys
from pathlib import Path


def setup_logging(
    file_log_level=logging.INFO,
    console_log_level=logging.INFO,
    log_file="prism.log",
    console_output=False,
    root_log_level=logging.INFO,
):
    """Configure root logger with file output and optional console output.

    Args:
        file_log_level: The minimum logging level for file output (default: logging.INFO)
            Supported levels in ascending order:
            - logging.DEBUG (10): Detailed information for debugging
            - logging.INFO (20): Confirmation that things are working
            - logging.WARNING (30): Indication that something unexpected happened
            - logging.ERROR (40): A more serious problem
            - logging.CRITICAL (50): Program may not be able to continue
        console_log_level: The minimum logging level for console output (default: logging.INFO)
            Uses same levels as file_log_level
        log_file: Path to the log file (default: "prism.log")
        console_output: Whether to output logs to console (default: False)
        root_log_level: The minimum logging level for the root logger (default: logging.INFO)
            Uses same levels as file_log_level
    """
    # Create logs directory if it doesn't exist
    # Use project root directory (one level up from prism package)
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / log_file

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(file_log_level)

    # Create formatter
    format_str = '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Get the root logger and set its level
    # IMPORTANT: Root logger level must be set to the LOWEST of all handler levels
    # or messages might be filtered before reaching the handlers
    root_logger = logging.getLogger()
    min_level = min(
        root_log_level, file_log_level, console_log_level if console_output else 100
    )  # 100 is higher than any log level
    root_logger.setLevel(min_level)

    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add handlers to the root logger
    root_logger.addHandler(file_handler)

    if console_output:
        # Create and add console handler if console output is enabled
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    log_msg = f"Logging setup complete. Logs will be saved to: {log_file_path}"
    if console_output:
        log_msg += " (console output enabled)"
    logging.info(log_msg)
