import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from qai_hub_models.models.moshi.demo import main as demo_main
from qai_hub_models.models.moshi.model import Moshi

def test_components() -> None:
    model = Moshi.from_pretrained()
    for name in model.component_names:
        component = model.components[name]
        inputs = component.sample_inputs(use_channel_last_format=False)
        outputs = component(*[value for value in inputs.values()])
        assert outputs is not None


def test_demo() -> None:
    demo_main(is_test=True)
