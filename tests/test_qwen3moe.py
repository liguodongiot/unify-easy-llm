import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from component.train_lora_utils import target_modules_dict


class TestQwen3MoETargetModules:
    """Test Qwen3-30B-A3B MoE target modules configuration"""

    def test_qwen3moe_target_modules_exists(self):
        """Verify qwen3MoE target_modules is defined"""
        assert 'qwen3MoE' in target_modules_dict, "qwen3MoE should be in target_modules_dict"
        
    def test_qwen3moe_target_modules_complete(self):
        """Verify qwen3MoE target_modules contains all required modules for MoE"""
        modules = target_modules_dict['qwen3MoE']
        expected_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        for mod in expected_modules:
            assert mod in modules, f"{mod} should be in qwen3MoE target_modules"
            
    def test_qwen3moe_target_modules_type(self):
        """Verify qwen3MoE target_modules is a list"""
        modules = target_modules_dict['qwen3MoE']
        assert isinstance(modules, list), "target_modules should be a list"
        
    def test_qwen3moe_has_moe_modules(self):
        """Verify qwen3MoE includes MoE-specific gate/up/down projections"""
        modules = target_modules_dict['qwen3MoE']
        assert 'gate_proj' in modules, "MoE should have gate_proj"
        assert 'up_proj' in modules, "MoE should have up_proj"
        assert 'down_proj' in modules, "MoE should have down_proj"


class TestQwen3MoEConfigFiles:
    """Test Qwen3-30B-A3B configuration files"""

    def test_qwen3moe_qlora_config_exists(self):
        """Verify qwen3MoE-qlora-config.json exists"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3MoE-qlora-config.json'
        )
        assert os.path.exists(config_path), "qwen3MoE-qlora-config.json should exist"
        
    def test_qwen3moe_lora_config_exists(self):
        """Verify qwen3MoE-lora-config.json exists"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3MoE-lora-config.json'
        )
        assert os.path.exists(config_path), "qwen3MoE-lora-config.json should exist"
        
    def test_qwen3moe_sft_config_exists(self):
        """Verify qwen3MoE-sft-config.json exists"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3MoE-sft-config.json'
        )
        assert os.path.exists(config_path), "qwen3MoE-sft-config.json should exist"
        
    def test_qwen3moe_qlora_config_sft_type(self):
        """Verify qwen3MoE QLoRA config has correct sft_type"""
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3MoE-qlora-config.json'
        )
        with open(config_path) as f:
            config = json.load(f)
        assert config.get('sft_type') == 'qlora', "sft_type should be qlora"
        
    def test_qwen3moe_lora_config_sft_type(self):
        """Verify qwen3MoE LoRA config has correct sft_type"""
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3MoE-lora-config.json'
        )
        with open(config_path) as f:
            config = json.load(f)
        assert config.get('sft_type') == 'lora', "sft_type should be lora"
        
    def test_qwen3moe_sft_config_sft_type(self):
        """Verify qwen3MoE SFT config has correct sft_type"""
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3MoE-sft-config.json'
        )
        with open(config_path) as f:
            config = json.load(f)
        assert config.get('sft_type') == 'full', "sft_type should be full"


class TestQwen3MoEReuseQwen3Template:
    """Test that qwen3MoE reuses qwen3 template"""

    def test_qwen3moe_uses_qwen3_template(self):
        """Verify qwen3MoE configs use qwen3 template"""
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3MoE-qlora-config.json'
        )
        with open(config_path) as f:
            config = json.load(f)
        assert config.get('prompt_template_name') == 'qwen3', "qwen3MoE should use qwen3 template"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
