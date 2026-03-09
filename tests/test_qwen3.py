import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from component.template import template_dict, Template
from component.train_lora_utils import target_modules_dict


class TestQwen3Template:
    """Test Qwen3 template registration"""

    def test_qwen3_template_registered(self):
        """Verify qwen3 template is registered"""
        assert 'qwen3' in template_dict, "qwen3 template should be registered"
        
    def test_qwen3_template_structure(self):
        """Verify qwen3 template has correct structure"""
        template = template_dict.get('qwen3')
        assert template is not None, "qwen3 template should exist"
        assert isinstance(template, Template), "qwen3 should be a Template instance"
        
    def test_qwen3_template_format(self):
        """Verify qwen3 template format matches expected pattern"""
        template = template_dict['qwen3']
        assert template.template_name == 'qwen3'
        assert '<|im_start|>system' in template.system_format
        assert '<|im_start|>user' in template.user_format
        assert '<|im_start|>assistant' in template.assistant_prompt_prefix
        assert template.system == "你是一个有用的助手。"
        
    def test_qwen3_start_word(self):
        """Verify qwen3 start_word is None"""
        template = template_dict['qwen3']
        assert template.start_word is None


class TestQwen3TargetModules:
    """Test Qwen3 LoRA target modules configuration"""

    def test_qwen3_target_modules_exists(self):
        """Verify qwen3 target_modules is defined"""
        assert 'qwen3' in target_modules_dict, "qwen3 should be in target_modules_dict"
        
    def test_qwen3_target_modules_values(self):
        """Verify qwen3 target_modules contains expected values"""
        modules = target_modules_dict['qwen3']
        assert 'q_proj' in modules, "q_proj should be in target_modules"
        assert 'v_proj' in modules, "v_proj should be in target_modules"
        
    def test_qwen3_target_modules_type(self):
        """Verify qwen3 target_modules is a list"""
        modules = target_modules_dict['qwen3']
        assert isinstance(modules, list), "target_modules should be a list"


class TestQwen3ConfigFiles:
    """Test Qwen3 configuration files"""

    def test_qwen3_lora_config_exists(self):
        """Verify qwen3-lora-config.json exists"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3-lora-config.json'
        )
        assert os.path.exists(config_path), "qwen3-lora-config.json should exist"
        
    def test_qwen3_sft_config_exists(self):
        """Verify qwen3-sft-config.json exists"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'train_args', 'qwen3-sft-config.json'
        )
        assert os.path.exists(config_path), "qwen3-sft-config.json should exist"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
