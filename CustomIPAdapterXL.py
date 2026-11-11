"""The old adapter has been superseded by material_transfer.attention.

Use MaterialTransfer(TransferConfig(method='C')) or method D. The former adapter
required unvendored ip_adapter code and instantiated a second UNet per image.
"""


def __getattr__(name):
    if name == "CustomIPAdapterXL":
        raise ImportError("Use material_transfer.pipeline.MaterialTransfer with method C or D; see README.md")
    raise AttributeError(name)
