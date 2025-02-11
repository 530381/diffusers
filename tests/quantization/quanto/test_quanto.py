import tempfile
import unittest

import torch

from diffusers import (
    QuantoConfig,
)
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.utils import is_optimum_quanto_available
from diffusers.utils.testing_utils import (
    nightly,
    require_accelerate,
    require_big_gpu_with_torch_cuda,
    torch_device,
)


if is_optimum_quanto_available():
    from optimum.quanto import QLinear


@nightly
@require_big_gpu_with_torch_cuda
@require_accelerate
class QuantoBaseTesterMixin:
    model_id = None
    model_cls = None
    torch_dtype = torch.bfloat16
    expected_memory_use_in_gb = 5
    keep_in_fp32_module = ""
    modules_to_not_convert = ""

    def get_dummy_init_kwargs(self):
        return {"weights": "float8"}

    def get_dummy_model_init_kwargs(self):
        return {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": self.torch_dtype,
            "quantization_config": QuantoConfig(**self.get_dummy_init_kwargs()),
        }

    def test_quanto_layers(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert isinstance(module, QLinear)

    def test_quanto_memory_usage(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        assert (model.get_memory_footprint() / 1024**3) < self.expected_memory_use_in_gb
        inputs = self.get_dummy_inputs()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            model(**inputs)
        max_memory = torch.cuda.max_memory_allocated()
        assert (max_memory / 1024**3) < self.expected_memory_use_in_gb

    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32.
        Also ensures if inference works.
        """
        _keep_in_fp32_modules = self.model_cls._keep_in_fp32_modules
        self.model_cls._keep_in_fp32_modules = self.keep_in_fp32_module

        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        model.to("cuda")

        assert (model.get_memory_footprint() / 1024**3) < self.expected_memory_use_in_gb
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in model._keep_in_fp32_modules:
                    assert module.weight.dtype == torch.float32
        self.model_cls._keep_in_fp32_modules = _keep_in_fp32_modules

    def test_modules_to_not_convert(self):
        init_kwargs = self.get_dummy_model_init_kwargs()

        quantization_config_kwargs = self.get_dummy_init_kwargs()
        quantization_config_kwargs.update({"modules_to_not_convert": self.modules_to_not_convert})
        quantization_config = QuantoConfig(**quantization_config_kwargs)

        init_kwargs.update({"quantization_config": quantization_config})

        model = self.model_cls.from_pretrained(**init_kwargs)
        model.to("cuda")

        for name, module in model.named_modules():
            if name in self.modules_to_not_convert:
                assert not isinstance(module, QLinear)

    def test_dtype_assignment(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        assert (model.get_memory_footprint() / 1024**3) < self.expected_memory_use_in_gb

        with self.assertRaises(ValueError):
            # Tries with a `dtype`
            model.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device` and `dtype`
            model.to(device="cuda:0", dtype=torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a cast
            model.float()

        with self.assertRaises(ValueError):
            # Tries with a cast
            model.half()

        # This should work
        model.to("cuda")

    def test_serialization(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        inputs = self.get_dummy_inputs()

        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**inputs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            saved_model = self.model_cls.from_pretrained(
                tmp_dir,
                torch_dtype=torch.bfloat16,
            )

        saved_model.to(torch_device)
        with torch.no_grad():
            saved_model_output = saved_model(**inputs)

        max_diff = torch.abs(model_output - saved_model_output).max()
        assert max_diff < 1e-5


class FluxTransformerQuantoMixin(QuantoBaseTesterMixin):
    model_id = "hf-internal-testing/tiny-flux-transformer"
    model_cls = FluxTransformer2DModel
    torch_dtype = torch.bfloat16
    keep_in_fp32_module = "proj_out"
    modules_to_not_convert = ["proj_out"]

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 4096, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 768),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
            "img_ids": torch.randn((4096, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "txt_ids": torch.randn((512, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "guidance": torch.tensor([3.5]).to(torch_device, self.torch_dtype),
        }


class FluxTransformerFloat8WeightsTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_use_in_gb = 10

    def get_dummy_init_kwargs(self):
        return {"weights": "float8"}


class FluxTransformerFloat8WeightsAndActivationTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_use_in_gb = 10

    def get_dummy_init_kwargs(self):
        return {"weights": "float8", "activations": "float8"}


class FluxTransformerInt8WeightsTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_use_in_gb = 10

    def get_dummy_init_kwargs(self):
        return {"weights": "int8"}

    def test_torch_compile(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        compiled_model = torch.compile(model, mode="max-autotune", fullgraph=True)
        inputs = self.get_dummy_inputs()

        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**inputs).sample
        model.to("cpu")

        compiled_model.to(torch_device)
        with torch.no_grad():
            compiled_model_output = compiled_model(**inputs).sample

        max_diff = torch.abs(model_output - compiled_model_output).max()
        assert max_diff < 1e-4


class FluxTransformerInt8WeightsAndActivationTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_use_in_gb = 10

    def get_dummy_init_kwargs(self):
        return {"weights": "int8", "activations": "int8"}

    def test_torch_compile(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        compiled_model = torch.compile(model, mode="max-autotune", fullgraph=True)
        inputs = self.get_dummy_inputs()

        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**inputs).sample
        model.to("cpu")

        compiled_model.to(torch_device)
        with torch.no_grad():
            compiled_model_output = compiled_model(**inputs).sample

        max_diff = torch.abs(model_output - compiled_model_output).max()
        assert max_diff < 1e-4


class FluxTransformerInt4WeightsTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_use_in_gb = 6

    def get_dummy_init_kwargs(self):
        return {"weights": "int4"}


class FluxTransformerInt2WeightsTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_use_in_gb = 6

    def get_dummy_init_kwargs(self):
        return {"weights": "int2"}
