import torch

from qai_hub_models.models.moshi.model import Moshi
from qai_hub_models.models.templates.moshi.app import MoshiApp


def main(is_test: bool = False) -> None:
    model = Moshi.from_pretrained()
    app = MoshiApp(model.encoder, model.temporal, model.depformer, model.decoder)
    audio = torch.zeros(1, 1, 1920)
    waveform, logits = app(audio)
    if is_test:
        assert waveform.shape == audio.shape
        assert logits.shape[1] == 8
    else:
        print(f"generated waveform={tuple(waveform.shape)} logits={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
