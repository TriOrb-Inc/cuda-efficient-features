import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_feature_matching_binary():
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "samples" / "sample_feature_matching",
        repo_root / "build" / "samples" / "sample_feature_matching",
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    pytest.skip("sample_feature_matching バイナリが見つかりません。先にビルドしてください。")


def _descriptor_params():
    for desc_type in range(8):
        for desc_bits in (256, 512):
            yield desc_type, desc_bits


def _find_image_pair():
    candidates = [
        Path(__file__).resolve().parent / "data" / "images",
        Path(__file__).resolve().parent.parent / "images" / "input",
    ]

    for base in candidates:
        image1 = base / "sample01/left.jpg"
        image2 = base / "sample01/right.jpg"
        if image1.exists() and image2.exists():
            return image1, image2

    return None, None


@pytest.mark.parametrize("desc_type,desc_bits", list(_descriptor_params()))
def test_sample_feature_matching_runs(sample_feature_matching_binary, tmp_path, desc_type, desc_bits):
    image1, image2 = _find_image_pair()

    if not image1 or not image2:
        pytest.skip("必要な入力画像が見つかりません。submodule もしくは images/input を確認してください。")

    cmd = [
        str(sample_feature_matching_binary),
        "--no-gui",
        str(image1),
        str(image2),
        f"--descriptor-type={desc_type}",
        f"--descriptor-bits={desc_bits}",
    ]

    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0, (
        "sample_feature_matching 実行に失敗しました\n"
        f"cmd: {' '.join(cmd)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )
