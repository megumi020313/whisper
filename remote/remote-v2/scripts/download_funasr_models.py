"""下载 FunASR VAD 与标点模型到指定目录"""
from pathlib import Path
from modelscope import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "models" / "paraformer"

MODELS = {
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch": "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
}


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for model_id, subdir in MODELS.items():
        dest = TARGET_DIR / subdir
        print(f"下载 {model_id} -> {dest}")
        snapshot_download(
            model_id,
            cache_dir=str(TARGET_DIR),
            local_dir=str(dest),
        )

    print("全部下载完成")


if __name__ == "__main__":
    main()
