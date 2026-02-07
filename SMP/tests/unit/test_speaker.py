"""Validate the multi-speaker model against hold-out samples."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any
import soundfile as sf
import numpy as np
import json
from datetime import datetime
import logging

# 过滤警告信息，保持输出清晰
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

# 确保项目根目录在 Python 路径中（用于直接运行测试文件）
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 导入项目模块（使用绝对路径）
from backend.modules.speaker_id.speaker_id_wrapper import SpeakerIDWrapper


def _get_project_paths() -> tuple[Path, Path, Path]:
    """获取项目路径（项目根目录、backend目录、training目录）"""
    project_root = Path(__file__).resolve().parent.parent.parent
    backend_dir = project_root / "backend"
    training_dir = backend_dir / "training"
    return project_root, backend_dir, training_dir


def _default_paths() -> tuple[Path, Path, Path]:
    """获取默认路径（模型路径、测试数据目录、编码器目录）"""
    project_root, backend_dir, training_dir = _get_project_paths()
    tests_dir = project_root / "tests"
    
    # 尝试查找最新的模型文件
    models_dir = training_dir / "models"
    model_path = None
    if models_dir.exists():
        # 查找最新的speaker_embeddings.json文件
        pattern = "speechbrain_model_*/speaker_embeddings.json"
        matches = sorted(models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            model_path = matches[0]
    
    # 如果找不到，使用默认路径
    if model_path is None:
        test_base = Path(__file__).resolve().parent
        model_path = test_base / "model" / "speaker-identification" / "speaker_model.bin"
    
    # 测试数据目录
    test_dir = tests_dir / "utils" / "test_data" / "samples"
    # 预训练模型目录
    encoder_dir = training_dir / "model" / "speechbrain"
    return model_path, test_dir, encoder_dir


def parse_args() -> argparse.Namespace:
    # 从模型配置读取默认值
    from backend.config.model_config import ModelConfig
    
    model_path, test_dir, encoder_dir = _default_paths()
    parser = argparse.ArgumentParser(description="Test Speaker-ID-Lite accuracy by headcount tier.")
    
    # 命令行参数
    # 默认值从模型配置读取
    default_model = str(model_path) if model_path else None
    parser.add_argument("--model", default=default_model, help="Path to speaker_embeddings.json or speaker_model.bin.")
    parser.add_argument("--test-dir", default=str(test_dir), help="Directory with test WAVs (1 per speaker).")
    parser.add_argument("--test-file", help="Single test WAV file path.")
    parser.add_argument("--sample-rate", type=int, default=ModelConfig.get_sample_rate())
    parser.add_argument("--threshold", type=float, default=ModelConfig.get_similarity_threshold(), help="Accept prediction when similarity exceeds this value.")
    parser.add_argument(
        "--encoder-root",
        default=str(ModelConfig.get_encoder_root()),
        help="Directory containing SpeechBrain encoder checkpoints (default: %(default)s).",
    )
    parser.add_argument("--output-dir", default="test/processed_data/speaker", help="Output directory for test results.")
    parser.add_argument("--log-file", default="test/logs/speaker_test.log", help="Log file path.")
    parser.add_argument("--report-file", help="Test report JSON file path (auto-generated if not specified).")
    parser.add_argument(
        "--max-audio-duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds (None for no limit, default: None)."
    )
    
    args = parser.parse_args()
    
    # 从模型配置读取默认值（如果命令行未指定）
    if not args.model or args.model == default_model:
        model_path_from_config = ModelConfig.get_latest_speaker_embeddings()
        if model_path_from_config:
            args.model = str(model_path_from_config)
    
    return args


def _load_test_samples(folder: Path) -> List[Tuple[str, str]]:
    if not folder.exists():
        raise FileNotFoundError(f"Test directory missing: {folder}")
    samples: List[Tuple[str, str]] = []
    for wav in sorted(folder.glob("*.wav")):
        name = wav.name
        if not name.startswith("emp_"):
            continue
        parts = name.split("_")
        if len(parts) < 2:
            continue
        # 提取为 emp_XXX 格式，与模型键名一致
        speaker_id = f"emp_{parts[1]}"
        samples.append((str(wav), speaker_id))
    if not samples:
        raise FileNotFoundError(
            f"No valid test samples found in {folder}. Expect files like emp_001_test.wav."
        )
    return samples


def _target_accuracy(headcount: int) -> int:
    if headcount <= 20:
        return 98
    if headcount <= 100:
        return 96
    return 95


def _load_single_test_sample(file_path: Path) -> List[Tuple[str, str]]:
    """加载单个测试样本"""
    # 确保路径是绝对路径
    if not file_path.is_absolute():
        file_path = file_path.resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Test file missing: {file_path}")
    
    name = file_path.name
    # 尝试从文件名中提取说话人ID
    if name.startswith("emp_"):
        parts = name.split("_")
        if len(parts) >= 2:
            # 提取为 emp_XXX 格式，与模型键名一致
            speaker_id = f"emp_{parts[1]}"
        else:
            speaker_id = "unknown"
    else:
        speaker_id = "unknown"
    
    return [(str(file_path), speaker_id)]


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 提取模型名称（从路径中）
    model_name = model_path.parent.name if model_path.parent.name.startswith("speechbrain_model_") else "未知模型"
    
    # 打印清晰的模型信息
    print("=" * 70)
    print("🔍 声纹识别测试")
    print("=" * 70)
    print(f"📦 加载模型: {model_name}")
    print(f"📁 模型路径: {model_path}")
    print(f"🎯 相似度阈值: {args.threshold:.2f}")
    print(f"🎤 采样率: {args.sample_rate} Hz")
    print("=" * 70)
    print()

    # 根据参数决定加载单个文件还是目录
    if args.test_file:
        # 测试单个文件 - 处理相对路径和绝对路径
        test_file_path = Path(args.test_file)
        # 如果是绝对路径，直接使用
        if test_file_path.is_absolute():
            test_file_path = test_file_path.resolve()
        else:
            # 相对路径：先尝试相对于当前工作目录
            cwd_path = Path.cwd() / test_file_path
            if cwd_path.exists():
                test_file_path = cwd_path.resolve()
            else:
                # 再尝试相对于脚本目录
                script_dir = Path(__file__).resolve().parent
                script_path = script_dir / test_file_path
                if script_path.exists():
                    test_file_path = script_path.resolve()
                else:
                    # 最后尝试相对于项目根目录
                    project_root, backend_dir, training_dir = _get_project_paths()
                    project_path = project_root / test_file_path
                    if project_path.exists():
                        test_file_path = project_path.resolve()
                    else:
                        # 如果都不存在，使用原始路径（让后续代码报错）
                        test_file_path = test_file_path.resolve()
        samples = _load_single_test_sample(test_file_path)
        print(f"测试单个文件: {test_file_path}")
    else:
        # 测试整个目录
        test_dir_path = Path(args.test_dir)
        if not test_dir_path.is_absolute():
            script_dir = Path(__file__).resolve().parent
            test_dir_path = (script_dir / test_dir_path).resolve()
        samples = _load_test_samples(test_dir_path)
        print(f"测试目录: {test_dir_path}，共 {len(samples)} 个文件")

    headcount = len({sid for _, sid in samples})

    sid = SpeakerIDWrapper(
        model_path=str(model_path),
        sample_rate=args.sample_rate,
        multi_speaker=True,
        encoder_root=args.encoder_root,
    )

    correct = 0
    wrong: List[str] = []

    # 生成测试标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_audio_name = Path(samples[0][0]).stem if samples else "unknown"
    
    # 设置日志
    log_file_path = Path(args.log_file)
    if not log_file_path.is_absolute():
        script_dir = Path(__file__).resolve().parent
        # 如果路径以test/开头，相对于test目录；否则相对于test目录
        if str(log_file_path).startswith("test/"):
            log_file_path = (script_dir / str(log_file_path)[5:]).resolve()  # 去掉"test/"前缀
        else:
            log_file_path = (script_dir / log_file_path).resolve()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式：包含时间戳和音频文件名
    log_filename = f"{log_file_path.stem}_{test_audio_name}_{timestamp}{log_file_path.suffix}"
    log_file_path = log_file_path.parent / log_filename
    
    # 配置日志：文件日志详细，控制台日志简洁
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # 控制台只显示警告和错误
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录详细信息到文件
    logger.info(f"=== Speaker ID Test Report ===")
    logger.info(f"Test Audio: {test_audio_name}")
    logger.info(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Threshold: {args.threshold:.2f}")
    
    print("🔄 开始测试...")
    print()
    max_audio_duration = args.max_audio_duration  # 从命令行参数获取，默认为None（不限制）
    sample_results = []  # 存储所有样本的测试结果
    for wav_path, true_id in samples:
        # 读取音频文件
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 转换为单声道
        
        # 如果设置了最大时长限制，且音频超过限制，则截断
        if max_audio_duration is not None and max_audio_duration > 0:
            max_samples = int(sr * max_audio_duration)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                print(f"[警告] 音频文件 {wav_path} 超过 {max_audio_duration} 秒，已截断到前 {max_audio_duration} 秒")
        # 如果 max_audio_duration 为 None，则使用整个音频文件（不截断）
        
        # 将音频转换为字节格式（int16）
        audio_int16 = (audio * 32768.0).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        predicted_id, similarity = sid.match_multi(audio_bytes)
        is_correct = (predicted_id == true_id) and (similarity >= args.threshold)
        sample_accuracy = 100.0 if is_correct else 0.0
        
        # 提取文件名（简化显示）
        file_name = Path(wav_path).name
        
        if is_correct:
            correct += 1
            print(f"  ✅ {file_name:30s} → {predicted_id:10s} (相似度: {similarity:.2f})")
            logger.info(f"✅ {wav_path} → {predicted_id} (similarity: {similarity:.2f}, accuracy: {sample_accuracy:.2f}%)")
        else:
            wrong.append(f"{wav_path} (truth {true_id}, predicted {predicted_id}, sim {similarity:.2f})")
            status = "❌" if predicted_id != true_id else "⚠️"
            reason = "识别错误" if predicted_id != true_id else "相似度不足"
            print(f"  {status} {file_name:30s} → {predicted_id:10s} (相似度: {similarity:.2f}, {reason})")
            logger.warning(f"❌ {wav_path}: predicted {predicted_id}, similarity {similarity:.2f}, accuracy: {sample_accuracy:.2f}%")
        
        # 记录详细结果用于报告
        sample_results.append({
            "file": str(wav_path),
            "true_id": true_id,
            "predicted_id": predicted_id,
            "similarity": round(similarity, 4),
            "correct": is_correct
        })

    accuracy = (correct / len(samples)) * 100
    target = _target_accuracy(headcount)
    
    logger.info(f"Total speakers: {headcount}")
    logger.info(f"Samples evaluated: {len(samples)}")
    logger.info(f"Accuracy: {accuracy:.2f}% (target {target}%)")
    logger.info(f"Correct: {correct}/{len(samples)}")
    logger.info(f"Errors: {len(wrong)}")
    
    if wrong:
        logger.warning("Failed samples:")
        for item in wrong:
            logger.warning(f"  - {item}")
    
    # 生成JSON测试报告
    # 确保所有值都是JSON可序列化的（将numpy类型转换为Python原生类型）
    passed = bool(accuracy >= target)
    
    report_data = {
        "test_info": {
            "test_time": datetime.now().isoformat(),
            "test_audio": test_audio_name,
            "test_audio_path": str(samples[0][0]) if samples else None,
            "model_path": str(args.model),
            "threshold": float(args.threshold),
            "sample_rate": int(args.sample_rate),
        },
        "results": {
            "total_speakers": int(headcount),
            "samples_evaluated": int(len(samples)),
            "correct": int(correct),
            "wrong": int(len(wrong)),
            "accuracy": float(round(accuracy, 2)),
            "target_accuracy": int(target),
            "passed": passed,
        },
        "details": {
            "all_samples": [
                {
                    "file": str(r["file"]),
                    "true_id": str(r["true_id"]),
                    "predicted_id": str(r["predicted_id"]),
                    "similarity": float(r["similarity"]),
                    "correct": bool(r["correct"])
                }
                for r in sample_results
            ],
            "correct_samples": [
                {
                    "file": str(r["file"]),
                    "true_id": str(r["true_id"]),
                    "predicted_id": str(r["predicted_id"]),
                    "similarity": float(r["similarity"]),
                    "correct": bool(r["correct"])
                }
                for r in sample_results if r["correct"]
            ],
            "wrong_samples": [
                {
                    "file": str(r["file"]),
                    "true_id": str(r["true_id"]),
                    "predicted_id": str(r["predicted_id"]),
                    "similarity": float(r["similarity"]),
                    "correct": bool(r["correct"])
                }
                for r in sample_results if not r["correct"]
            ]
        }
    }
    
    # 保存报告
    if args.report_file:
        report_path = Path(args.report_file)
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            script_dir = Path(__file__).resolve().parent
            # 如果路径以test/开头，相对于test目录；否则相对于test目录
            if str(output_dir).startswith("test/"):
                output_dir = (script_dir / str(output_dir)[5:]).resolve()  # 去掉"test/"前缀
            else:
                output_dir = (script_dir / output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report_filename = f"speaker_test_report_{test_audio_name}_{timestamp}.json"
        report_path = output_dir / report_filename
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Test report saved to: {report_path}")
    logger.info(f"Log file saved to: {log_file_path}")

    print()
    print("=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    print(f"👥 说话人数量: {headcount}")
    print(f"📝 测试样本数: {len(samples)}")
    print(f"✅ 正确识别: {correct}/{len(samples)}")
    print(f"❌ 错误识别: {len(wrong)}/{len(samples)}")
    print(f"📈 准确率: {accuracy:.2f}%")
    print(f"🎯 目标准确率: {target}%")
    print(f"⚙️  相似度阈值: {args.threshold:.2f}")
    print("=" * 70)
    
    if wrong:
        print()
        print("❌ 失败样本详情:")
        for item in wrong:
            file_name = Path(item.split()[0]).name
            details = " ".join(item.split()[1:])
            print(f"   • {file_name:30s} - {details}")
    
    print()
    print("=" * 70)
    if accuracy >= target:
        print("✅ 模型通过验收标准，可以部署")
        logger.info("✅ Model passes acceptance criteria. Ready for deployment")
    else:
        print(f"⚠️  模型未达到目标（当前: {accuracy:.2f}%, 目标: {target}%）")
        print("   建议：收集更多数据或调整超参数")
        logger.warning("❌ Model below target. Consider collecting more data or tweaking hyperparameters.")
    print("=" * 70)
    print()
    print(f"📄 详细报告: {report_path}")
    print(f"📋 日志文件: {log_file_path}")
    print()


if __name__ == "__main__":
    main()
