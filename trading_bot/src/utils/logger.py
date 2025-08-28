"""
Systeme de logging centralisé pour le trading bot.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

def setup_logging(config_logging) -> logging.Logger:
    """
    Configure le syst�me de logging.
    
    Args:
        config_logging: Configuration du logging depuis settings.py
        
    Returns:
        Logger principal configur�.
    """
    # Cr�er le dossier logs s'il n'existe pas
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configuration du logger principal
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config_logging.level, logging.INFO))
    
    # �viter la duplication des handlers
    if logger.handlers:
        return logger
    
    # Handler pour fichier avec rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / config_logging.file_name,
        maxBytes=config_logging.max_bytes,
        backupCount=config_logging.backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Handler pour console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(config_logging.format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Ajouter les handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger nomm�.
    
    Args:
        name: Nom du logger (g�n�ralement __name__).
        
    Returns:
        Instance du logger.
    """
    return logging.getLogger(name)

# Instance globale pour compatibilit�
_main_logger: Optional[logging.Logger] = None

def get_main_logger() -> logging.Logger:
    """Retourne le logger principal."""
    global _main_logger
    if _main_logger is None:
        _main_logger = logging.getLogger("trading_bot")
    return _main_logger
