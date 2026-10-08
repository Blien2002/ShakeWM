import torch
import pytest
from shakewm.encoder import MockTeacher, OfficialTeacher, clean_state


def test_mock_frame_batch_consistency():
    teacher = MockTeacher()
    x = torch.rand(3, 3, 32, 32)
    torch.testing.assert_close(teacher(x), torch.cat([teacher(i[None]) for i in x]), atol=0, rtol=0)
    teacher.train()
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())


def test_official_requires_weights_and_checks_hash(tmp_path):
    with pytest.raises(ValueError, match="explicit local"):
        OfficialTeacher(None)
    path = tmp_path / "bad.pt"; torch.save({}, path)
    with pytest.raises(ValueError, match="checksum"):
        OfficialTeacher(path, expected_sha256="0" * 64)


def test_strict_ema_loading(monkeypatch, tmp_path):
    # Small fake architecture isolates strict key handling, not official numerical quality.
    from shakewm import encoder
    monkeypatch.setattr(encoder, "official_architecture", lambda: torch.nn.Linear(2, 2))
    path = tmp_path / "bad.pt"
    torch.save({"ema_encoder": {"module.backbone.weight": torch.ones(2, 2)}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        OfficialTeacher(path)
    torch.save({"ema_encoder": {"module.backbone.weight": torch.ones(2, 2),
                                "module.backbone.bias": torch.zeros(2)}}, path)
    model = OfficialTeacher(path)
    assert all(not p.requires_grad for p in model.parameters())
    model.train()
    assert not model.training and not model.encoder.training
    with pytest.raises(ValueError, match="duplicate"):
        clean_state({"module.weight": torch.ones(1), "weight": torch.ones(1)})
