import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from analysis.scripts.check_search_mistral_native import check, REVISION


class NativePreflightTests(unittest.TestCase):
    def test_missing_native_files_fail_before_loader(self):
        with TemporaryDirectory() as d:
            def forbidden(*args, **kwargs):
                self.fail("loader called without native files")
            with self.assertRaisesRegex(ValueError, "params.json missing"):
                check(d, config_loader=forbidden, supported_architectures=[])
            Path(d, "params.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "safetensors missing"):
                check(d, config_loader=forbidden, supported_architectures=[])

    def test_native_resolution_and_dimension_checks(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            dimensions = dict(hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
                              n_routed_experts=128, num_experts_per_tok=4, kv_lora_rank=256, q_lora_rank=1024)
            root.joinpath('params.json').write_text('{}')
            root.joinpath('config.json').write_text(json.dumps({'text_config': dimensions}))
            root.joinpath('consolidated.safetensors').write_bytes(b'fixture-not-real-weights')
            text = SimpleNamespace(**dimensions, architectures=['NativeBackbone'])
            config = SimpleNamespace(text_config=text, architectures=['NativeWrapper'], to_dict=lambda: {})
            def loader(path, **kwargs):
                self.assertEqual(kwargs, dict(trust_remote_code=False, revision=REVISION, config_format='mistral'))
                return config
            supported = ['NativeWrapper', 'NativeBackbone']
            result = check(root, config_loader=loader, supported_architectures=supported)
            self.assertFalse(result['weights_loaded'])
            root.joinpath('consolidated.safetensors').unlink()
            result = check(root, config_loader=loader, supported_architectures=supported, config_only=True)
            self.assertFalse(result['native_weights_present'])
            self.assertTrue(result['config_only'])
            with self.assertRaisesRegex(ValueError, 'safetensors missing'):
                check(root, config_loader=loader, supported_architectures=supported)
            root.joinpath('consolidated.safetensors').write_bytes(b'fixture-not-real-weights')
            text.architectures = None
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                check(root, config_loader=loader, supported_architectures=supported)
            text.architectures = ['NativeBackbone']
            text.num_hidden_layers = 35
            with self.assertRaisesRegex(ValueError, 'disagreement'):
                check(root, config_loader=loader, supported_architectures=supported)
