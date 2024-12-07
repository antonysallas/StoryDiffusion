# main.py
import logging
import sys
from typing import Dict, Optional

from models.model_loader import ModelLoader

from config.settings import configure_logging


class SDXLApplication:
    """Main SDXL application class."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_loader: Optional[ModelLoader] = None
        self.models: Dict = {}

    def initialize(self) -> bool:
        """Initialize application components."""
        try:
            self.model_loader = ModelLoader()
            self.models = self.model_loader.load_models()
            self.logger.info(f"Running on device: {self.model_loader.device}")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False

    def shutdown(self) -> None:
        """Cleanup resources."""
        if self.model_loader:
            self.model_loader.clear_models()
            self.logger.info("Resources cleaned up")


def main():
    """Application entry point."""
    # Configure logging
    configure_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize application
        app = SDXLApplication()
        if not app.initialize():
            sys.exit(1)

        # Application logic here

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        if "app" in locals():
            app.shutdown()


if __name__ == "__main__":
    main()
