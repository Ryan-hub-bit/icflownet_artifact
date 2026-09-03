"""Simple file-and-console logger used by the GT extraction pipelines."""

import logging
import os
import time
from datetime import datetime

class Logger:
    """Custom logger class to handle both console and file logging."""
    
    def __init__(self, base_dir, log_dir="logs", log_level=logging.INFO):
        """Initialize logger with base directory and optional log directory name.
        
        Args:
            base_dir (str): Base directory for logs
            log_dir (str): Name of the logs directory inside base_dir
            log_level (int): Logging level (use constants from logging module:
                            logging.DEBUG, logging.INFO, logging.WARNING,
                            logging.ERROR, logging.CRITICAL)
        """
        # Create logs directory if it doesn't exist
        self.log_dir = os.path.join(base_dir, log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.log_dir, f"neujump_{timestamp}.log")
        
        # Configure logger
        self.logger = logging.getLogger("NeuJump")
        self.logger.setLevel(log_level)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.start_time = time.time()
        self.binary_start_time = None
        
        # Log initial information
        self.logger.info(f"Logger initialized. Log file: {self.log_filename}")
        self.logger.info(f"Base directory: {base_dir}")
    
    def info(self, message):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log a critical message."""
        self.logger.critical(message)
    
    def start_binary_timer(self):
        """Start timing for binary processing."""
        self.binary_start_time = time.time()
    
    def end_binary_timer(self, binary_name):
        """End binary timer and log processing time."""
        if self.binary_start_time is not None:
            binary_time = time.time() - self.binary_start_time
            self.logger.info(f"Binary processing time for {binary_name}: {binary_time:.2f} seconds")
            self.binary_start_time = None
    
    def log_statistics(self, total, skipped, processed):
        """Log final statistics."""
        total_time = time.time() - self.start_time
        self.logger.info("=" * 50)
        self.logger.info("Final Statistics:")
        self.logger.info(f"Total binaries: {total}")
        self.logger.info(f"Skipped binaries: {skipped}")
        self.logger.info(f"Processed binaries: {processed}")
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")
        self.logger.info("=" * 50)
        
    def set_level(self, level):
        """Set the logging level for both the logger and all handlers.
        
        Args:
            level (int): Logging level (use constants from logging module:
                        logging.DEBUG, logging.INFO, logging.WARNING,
                        logging.ERROR, logging.CRITICAL)
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
        self.logger.info(f"Log level changed to: {self._level_name(level)}")
    
    def _level_name(self, level):
        """Convert logging level number to name."""
        if level == logging.DEBUG:
            return "DEBUG"
        elif level == logging.INFO:
            return "INFO"
        elif level == logging.WARNING:
            return "WARNING"
        elif level == logging.ERROR:
            return "ERROR"
        elif level == logging.CRITICAL:
            return "CRITICAL"
        else:
            return str(level)
    
    def set_debug(self):
        """Set logging level to DEBUG."""
        self.set_level(logging.DEBUG)
        
    def set_info(self):
        """Set logging level to INFO."""
        self.set_level(logging.INFO)
        
    def set_warning(self):
        """Set logging level to WARNING."""
        self.set_level(logging.WARNING)
        
    def set_error(self):
        """Set logging level to ERROR."""
        self.set_level(logging.ERROR)
        
    def set_critical(self):
        """Set logging level to CRITICAL."""
        self.set_level(logging.CRITICAL)
