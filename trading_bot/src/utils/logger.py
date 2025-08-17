"""
Système de logging centralisé pour le trading bot.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class TradingBotLogger:
    """Logger personnalisé pour le trading bot."""

    def __init__(self, name: str = "trading_bot", 
                 log_file: Optional[str] = None,
                 level: int = logging.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Éviter la duplication des handlers
        if not self.logger.handlers:
            self._setup_handlers(log_file, level)

    def _setup_handlers(self, log_file: Optional[str], level: int):
        """Configure les handlers pour console et fichier."""

        # Format des logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Handler pour la console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Handler pour fichier si spécifié
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log info."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug."""
        self.logger.debug(message)

# Instance globale
_main_logger = None

def get_logger(name: str = "trading_bot") -> TradingBotLogger:
    """Retourne une instance du logger."""
    global _main_logger

    if _main_logger is None:
        # Initialiser avec le fichier de log par défaut
        log_file = Path.cwd() / "logs" / "trading_bot.log"
        _main_logger = TradingBotLogger(name, str(log_file))

    return _main_logger

def setup_logging():
    """Configure le logging pour l'interface utilisateur."""
    global _main_logger

    log_file = Path.cwd() / "logs" / "ui.log"
    _main_logger = TradingBotLogger("trading_bot.ui", str(log_file))
