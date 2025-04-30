model_cls_mapping = {
    'qwen2': ('.qwen2', 'Qwen2Assistant'),
    'diva': ('.diva', 'DiVAAssistant'),
    'naive': ('.naive', 'NaiveAssistant'),
    'naive2': ('.naive2', 'Naive2Assistant'),
    'naive3': ('.naive3', 'Naive3Assistant'),
    'naive4': ('.naive4', 'Naive4Assistant'),
    'mini_omni': ('.mini_omni', 'MiniOmniAssistant'),
    'mini_omni2': ('.mini_omni2', 'MiniOmni2Assistant'),
    'gpt4o': ('.gpt4o', 'GPT4oAssistant'),
    'gpt4o_mini': ('.gpt4o', 'GPT4oMiniAssistant'),
    'moshi': ('.moshi', 'MoshiAssistant'),
    'glm': ('.glm', 'GLMAssistant'),
    'ultravox': ('.ultravox', 'UltravoxAssistant'),
    'ultravox0_5': ('.ultravox', 'Ultravox0d5Assistant'),
    'ichigo': ('.ichigo', 'IchigoeAssistant'),
    'megrez': ('.megrez', 'MegrezAssistant'),
    'meralion': ('.meralion', 'MERaLiONAssistant'),
    'lyra_mini': ('.lyra', 'LyraMiniAssistant'),
    'lyra_base': ('.lyra', 'LyraBaseAssistant'),
    'freeze_omni': ('.freeze_omni', 'FreezeOmniAssistant'),
    'minicpm': ('.minicpm', 'MiniCPMAssistant'),
    'baichuan_omni': ('.baichuan', 'BaichuanOmniAssistant'),
    'baichuan_audio': ('.baichuan', 'BaichuanAudioAssistant'),
    'step': ('.step_audio', 'StepAssistant'),
    'phi': ('.phi', 'PhiAssistant'),
    'qwen_omni_turbo': ('.qwen_omni_turbo', 'QwenOmniAssistant'),
    'speechgpt2': ('.speechgpt2', 'SpeechGPT2'),
    'localhost': ('.localhost', 'LocalAssistant'),
}

def load_model(model_name):
    """
    Load a model by its name.
    
    Args:
        model_name (str): The name of the model to load.
    
    Returns:
        model: An instance of the specified model class.
    """
    if model_name not in model_cls_mapping:
        raise ValueError(f"Model '{model_name}' is not available. Available models: {list_models()}")
    
    import importlib
    module_path, class_name = model_cls_mapping[model_name]
    module = importlib.import_module(module_path, package="src.models")
    model_class = getattr(module, class_name)
    
    return model_class()

def list_models():
    """
    List all available models.
    """
    return list(model_cls_mapping.keys())