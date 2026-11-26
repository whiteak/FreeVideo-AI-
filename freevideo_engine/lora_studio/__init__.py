"""LoRA Studio: build a training set for a character, an object or a style, train a MiniMax H3 LoRA from it, and
link the finished LoRA to a cast card so writing @Name in a prompt loads it.

The page (``page/``) runs the picture and caption work through the engine's own image and prompt-model routes; this
package keeps the datasets on disk, prepares imported photos, writes the export layouts and runs the external trainer.
Datasets live in ComfyUI's input folder under ``lora-studio/<id>/`` so every picture can be shown with ``/view`` and
handed to the image and prompt models without another copy.
"""
