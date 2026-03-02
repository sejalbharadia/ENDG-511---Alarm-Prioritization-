import torch
import copy


def quantize_fx_model(model, example_inputs, backend='fbgemm'):
    """Attempt FX graph-mode static quantization (preferred). If unavailable fall back to dynamic quantization.

    Args:
        model: nn.Module
        example_inputs: a tuple (x,) or tensor
    Returns quantized model.
    """
    # Try FX prepare/convert if available
    try:
        import torch.quantization.quantize_fx as qfx
        model_eval = copy.deepcopy(model).eval()

        if not isinstance(example_inputs, tuple):
            example = (example_inputs,)
        else:
            example = example_inputs

        qconfig = {"": torch.quantization.get_default_qconfig(backend)}
        prepared = qfx.prepare_fx(model_eval, qconfig, example)
        # calibration - run a few batches; here we run a single example for simplicity
        with torch.no_grad():
            for inp in example:
                prepared(inp)
        converted = qfx.convert_fx(prepared)
        return converted
    except Exception:
        # fallback: dynamic quantization (works mainly for linear layers)
        try:
            qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            return qmodel
        except Exception as e:
            raise RuntimeError("Quantization failed: " + str(e))
