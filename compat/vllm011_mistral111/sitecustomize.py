"""Compatibility aliases for vLLM 0.11 with mistral-common 1.11.

mistral-common 1.11 moved multimodal chunk classes from ``messages`` to
``chunk``.  vLLM 0.11 still imports ``ImageChunk`` from the old location while
inspecting LLaVA models.  Re-exporting the class keeps the installed packages
unchanged and can be removed after vLLM is upgraded.
"""

try:
    from mistral_common.protocol.instruct import messages
    from mistral_common.protocol.instruct.chunk import ImageChunk

    if not hasattr(messages, "ImageChunk"):
        messages.ImageChunk = ImageChunk
except ImportError:
    pass
