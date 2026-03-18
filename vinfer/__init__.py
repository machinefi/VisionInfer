from .main import main
from .inference_core import infer_frame
from .frame_producer import start_frame_producer, stop_frame_producer
from .backend.ollama_manager import start_ollama_serve, stop_ollama_serve

__version__ = "0.2.0"
__all__ = [
    "main", "infer_frame", "start_frame_producer",
    "stop_frame_producer", "start_ollama_serve", "stop_ollama_serve"
]