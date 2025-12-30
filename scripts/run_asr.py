#!/usr/bin/env python3
"""
Faster-Whisper ASR 启动脚本

支持单个音频文件和批量文件夹处理。

使用方法：
    python scripts/run_asr.py  # 使用配置文件中的默认路径
    python scripts/run_asr.py --input /path/to/audio.wav
    python scripts/run_asr.py --input /path/to/audio_folder/
    python scripts/run_asr.py --config config/model_config.yaml --input audio.wav

参数说明：
    --input, -i: 输入音频文件或文件夹路径（必需）
    --output, -o: 输出目录（可选，默认使用配置文件中的设置）
    --config, -c: 配置文件路径（可选，默认 config/model_config.yaml）
    --language, -l: 语言代码（可选，默认使用配置文件）
    --beam-size, -b: Beam search 大小（可选，默认使用配置文件）
    --timestamps: 是否输出时间戳（可选）
    --words: 是否输出词级别时间戳（可选）
    --verbose, -v: 详细输出
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.modules.audio_analysis import WhisperService
from backend.core.config import get_config
from backend.utils.logger import get_logger

logger = get_logger()


class ASRProcessor:
    """ASR 处理类"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化 ASR 处理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or "config/model_config.yaml"
        self.config = None
        self.service = None
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            from backend.core.config import Config

            # 创建配置实例
            self.config = Config()
            self.config._load_config()

            logger.info("✅ 配置加载成功")
            logger.info(f"   模型路径: {self.config.whisper_path}")
            logger.info(f"   设备: {self.config.asr_model_device}")
            logger.info(f"   语言: {self.config.asr_language}")

        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            raise

    def _init_service(self):
        """初始化 ASR 服务"""
        try:
            self.service = WhisperService()
            logger.info("✅ ASR 服务初始化成功")
        except Exception as e:
            logger.error(f"❌ ASR 服务初始化失败: {e}")
            raise

    def _get_audio_files(self, input_path: str) -> List[Path]:
        """获取所有音频文件

        Args:
            input_path: 输入路径（文件或文件夹）

        Returns:
            音频文件路径列表
        """
        input_path = Path(input_path)

        if input_path.is_file():
            # 单个文件
            if input_path.suffix.lower() in self.config.supported_formats:
                return [input_path]
            else:
                logger.error(f"❌ 不支持的文件格式: {input_path.suffix}")
                return []
        elif input_path.is_dir():
            # 文件夹
            audio_files = []
            pattern = "**/*" if self.config.recursive else "*"

            for ext in self.config.supported_formats:
                audio_files.extend(input_path.glob(f"{pattern}{ext}"))
                audio_files.extend(input_path.glob(f"{pattern}{ext.upper()}"))

            # 去重并排序
            audio_files = sorted(list(set(audio_files)))
            logger.info(f"📁 发现 {len(audio_files)} 个音频文件")
            return audio_files
        else:
            logger.error(f"❌ 输入路径不存在: {input_path}")
            return []

    def _save_result(self, audio_path: Path, result: Any, output_dir: Path,
                    with_timestamps: bool = False, with_words: bool = False) -> Path:
        """保存识别结果

        Args:
            audio_path: 音频文件路径
            result: 识别结果
            output_dir: 输出目录
            with_timestamps: 是否包含时间戳
            with_words: 是否为词级别时间戳

        Returns:
            输出文件路径
        """
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成输出文件名
        stem = audio_path.stem
        if with_words:
            output_file = output_dir / f"{stem}_words.json"
        elif with_timestamps:
            output_file = output_dir / f"{stem}_timestamps.json"
        else:
            output_file = output_dir / f"{stem}.txt"

        # 保存结果
        if output_file.suffix == '.json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)

        return output_file

    def process_file(self, audio_path: Path, output_dir: Path,
                    language: Optional[str] = None,
                    beam_size: Optional[int] = None,
                    with_timestamps: bool = False,
                    with_words: bool = False,
                    verbose: bool = False) -> bool:
        """处理单个音频文件

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录
            language: 语言代码
            beam_size: Beam search 大小
            with_timestamps: 是否输出时间戳
            with_words: 是否输出词级别时间戳
            verbose: 是否详细输出

        Returns:
            处理是否成功
        """
        try:
            if verbose:
                logger.info(f"🎵 处理音频文件: {audio_path}")

            start_time = time.time()

            # 确定参数
            lang = language or self.config.asr_language
            beam = beam_size or self.config.asr_beam_size

            if verbose:
                logger.info(f"   语言: {lang}, Beam size: {beam}")

            # 执行基本转录
            basic_result = self.service.transcribe(
                audio=str(audio_path),
                language=lang,
                beam_size=beam
            )

            # 保存基本转录结果
            basic_output_file = self._save_result(audio_path, basic_result, output_dir,
                                                with_timestamps=False, with_words=False)

            # 执行带时间戳转录
            if with_words:
                timestamp_result = self.service.transcribe_with_word_timestamps(
                    audio=str(audio_path),
                    language=lang,
                    beam_size=beam
                )
            else:
                timestamp_result = self.service.transcribe_with_timestamps(
                    audio=str(audio_path),
                    language=lang,
                    beam_size=beam
                )

            # 保存时间戳结果
            timestamp_output_file = self._save_result(audio_path, timestamp_result, output_dir,
                                                    with_timestamps=True, with_words=with_words)

            elapsed_time = time.time() - start_time

            if verbose:
                logger.info(f"   识别结果: {basic_result[:50]}{'...' if len(basic_result) > 50 else ''}")
                logger.info(f"   处理时间: {elapsed_time:.2f}秒")
                logger.info(f"   输出文件:")
                logger.info(f"     - 基本转录: {basic_output_file}")
                logger.info(f"     - 时间戳: {timestamp_output_file}")

            return True

        except Exception as e:
            logger.error(f"❌ 处理失败 {audio_path}: {e}")
            return False

    def process(self, input_path: str, output_dir: Optional[str] = None,
               language: Optional[str] = None, beam_size: Optional[int] = None,
               with_timestamps: bool = False, with_words: bool = False,
               verbose: bool = False) -> bool:
        """处理音频文件或文件夹

        Args:
            input_path: 输入路径（文件或文件夹）
            output_dir: 输出目录
            language: 语言代码
            beam_size: Beam search 大小
            with_timestamps: 是否输出时间戳
            with_words: 是否输出词级别时间戳
            verbose: 是否详细输出

        Returns:
            处理是否成功
        """
        # 初始化服务
        self._init_service()

        # 确定基础输出目录
        if output_dir:
            base_output_path = Path(output_dir)
        else:
            base_output_path = Path(self.config.output_dir)

        # 创建以时间命名的子文件夹
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = base_output_path / timestamp
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取音频文件列表
        audio_files = self._get_audio_files(input_path)
        if not audio_files:
            logger.error("❌ 未找到可处理的音频文件")
            return False

        logger.info(f"🚀 开始处理 {len(audio_files)} 个音频文件")
        logger.info(f"   输出目录: {output_path.absolute()}")

        # 处理每个文件
        success_count = 0
        for i, audio_file in enumerate(audio_files, 1):
            if verbose:
                logger.info(f"\n[{i}/{len(audio_files)}]")

            # 默认同时输出基本转录和带时间戳
            if self.process_file(audio_file, output_path, language, beam_size,
                              with_timestamps=True, with_words=with_words, verbose=verbose):
                success_count += 1

        # 输出统计信息
        logger.info(f"\n📊 处理完成: {success_count}/{len(audio_files)} 成功")

        if success_count == len(audio_files):
            logger.info("🎉 所有文件处理成功！")
            return True
        else:
            logger.warning(f"⚠️  {len(audio_files) - success_count} 个文件处理失败")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Faster-Whisper ASR 音频转文字工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用配置文件中的默认路径
  python scripts/run_asr.py

  # 处理单个音频文件
  python scripts/run_asr.py --input audio.wav

  # 处理整个文件夹
  python scripts/run_asr.py --input audio_folder/

  # 指定输出目录和语言
  python scripts/run_asr.py --input audio.wav --output results --language zh

  # 输出带时间戳的结果
  python scripts/run_asr.py --input audio.wav --timestamps

  # 输出词级别时间戳
  python scripts/run_asr.py --input audio.wav --words
        """
    )

    parser.add_argument(
        "--input", "-i",
        help="输入音频文件或文件夹路径（可选，默认使用配置文件中的 audio.input_path）"
    )

    parser.add_argument(
        "--output", "-o",
        help="输出目录（可选，默认使用配置文件）"
    )

    parser.add_argument(
        "--config", "-c",
        default="config/model_config.yaml",
        help="配置文件路径（默认: config/model_config.yaml）"
    )

    parser.add_argument(
        "--language", "-l",
        choices=["zh", "en", "auto"],
        help="语言代码：zh（中文）、en（英文）、auto（自动检测）"
    )

    parser.add_argument(
        "--beam-size", "-b",
        type=int,
        choices=[1, 3, 5, 10],
        help="Beam search 大小：1（最快）、3（平衡）、5（高质量）、10（最高精度）"
    )

    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="输出带时间戳的片段结果（JSON格式）"
    )

    parser.add_argument(
        "--words",
        action="store_true",
        help="输出词级别时间戳结果（JSON格式）"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )

    args = parser.parse_args()

    # 创建临时处理器用于读取配置
    try:
        temp_processor = ASRProcessor(args.config)

        # 确定输入路径
        input_path = args.input
        if not input_path:
            # 使用配置文件中的默认路径
            if temp_processor.config.input_path:
                input_path = temp_processor.config.input_path
                logger.info(f"📁 使用配置文件中的默认输入路径: {input_path}")
            else:
                logger.error("❌ 未指定输入路径，且配置文件中没有默认路径")
                return 1

        # 检查输入路径
        if not os.path.exists(input_path):
            logger.error(f"❌ 输入路径不存在: {input_path}")
            return 1

        # 检查参数冲突
        if args.timestamps and args.words:
            logger.error("❌ 不能同时指定 --timestamps 和 --words")
            return 1

        # 执行处理
        success = temp_processor.process(
            input_path=input_path,
            output_dir=args.output,
            language=args.language,
            beam_size=args.beam_size,
            with_timestamps=args.timestamps,
            with_words=args.words,
            verbose=args.verbose
        )

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\n⚠️  用户中断处理")
        return 1
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
