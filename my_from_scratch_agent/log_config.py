import logging

from rich.logging import RichHandler

# Configure the root logger using RichHandler
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=True, tracebacks_show_locals=False)
    ],
)
