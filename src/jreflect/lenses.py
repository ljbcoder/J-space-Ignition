"""Apply native J++ matrices without mutating a generation model."""

from pathlib import Path

from .artifacts import file_hash, read_json

JPP_COMMIT = "d7efc0de372e20589e899188c0cfe8434acc63ec"


class JppLens:
    def __init__(self, path, config, hidden_size, layer_count):
        import torch

        self.path = Path(path)
        metadata_path = self.path.with_suffix(".meta.json")
        if not metadata_path.exists():
            raise ValueError("Lens needs a .meta.json provenance sidecar from jpp-fit")
        meta = read_json(metadata_path)
        if (
            meta["model"] != config.model.name
            or meta["revision"] != config.model.revision
            or meta["lens_sha256"] != file_hash(self.path)
            or meta["method"] != "jpp"
            or meta["upstream_commit"] != JPP_COMMIT
        ):
            raise ValueError("Lens provenance does not match this experiment")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved_config = checkpoint["config"]
        if saved_config.get("hf_model_name") != config.model.name:
            raise ValueError("Lens checkpoint model mismatch")
        if saved_config.get("relative_end_transport_layer") != -1:
            raise ValueError("Lens must target the final block")
        self.maps = checkpoint.get("parameters", {}).get("jacobians", checkpoint.get("J"))
        if self.maps is None or not set(config.readout.layers) <= set(self.maps):
            raise ValueError("Lens does not contain all requested source layers")
        for layer in config.readout.layers:
            matrix = self.maps[layer]
            if not 0 <= layer < layer_count - 1 or matrix.shape != (hidden_size, hidden_size):
                raise ValueError(f"Invalid Jacobian geometry at layer {layer}")
            if not torch.isfinite(matrix).all():
                raise ValueError("Lens contains non-finite matrices")
        self.sha256 = meta["lens_sha256"]

    def transport(self, residual, layer):
        # Keep maps on CPU, transferring one layer at a time to bound GPU memory.
        # Upstream fits/stores fp32 maps. Preserve that precision through transport,
        # then let the model's unembedding cast back to its inference dtype.
        matrix = self.maps[layer].to(device=residual.device)
        return residual.to(matrix.dtype) @ matrix.T
