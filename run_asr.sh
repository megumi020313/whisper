#!/bin/bash

# Faster-Whisper ASR 启动脚本
# 支持单个音频文件和批量文件夹处理

set -e  # 遇到错误立即退出

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置
CONFIG_FILE="config/model_config.yaml"
PYTHON_SCRIPT="scripts/run_asr.py"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "Faster-Whisper ASR 音频转文字工具"
    echo ""
    echo "用法:"
    echo "  $0 [选项] [输入路径]"
    echo ""
    echo "参数:"
    echo "  [输入路径]          音频文件或文件夹路径（可选，默认使用配置文件）"
    echo ""
    echo "选项:"
    echo "  -o, --output DIR    输出目录（默认: output）"
    echo "  -l, --language LANG 语言代码: zh/en/auto（默认: zh）"
    echo "  -b, --beam-size N   Beam search 大小: 1/3/5/10（默认: 5）"
echo "  -t, --timestamps    输出带时间戳的结果（默认启用）"
echo "  -w, --words         输出词级别时间戳结果"
echo "  -v, --verbose       详细输出"
echo "  -h, --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 使用配置文件中的默认路径处理"
    echo "  $0"
    echo ""
    echo "  # 处理单个音频文件"
    echo "  $0 audio.wav"
    echo ""
    echo "  # 处理文件夹中的所有音频"
    echo "  $0 audio_folder/"
    echo ""
    echo "  # 指定输出目录和语言"
    echo "  $0 -o results -l zh audio.wav"
    echo ""
echo "  # 输出词级别时间戳结果（可选）"
echo "  $0 --words audio.wav"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."

    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3，请安装 Python 3.8+"
        exit 1
    fi

    # 检查配置文件
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi

    # 检查Python脚本
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_error "Python脚本不存在: $PYTHON_SCRIPT"
        exit 1
    fi

    log_success "依赖检查通过"
}

# 主函数
main() {
    # 解析命令行参数
    local output_dir=""
    local language=""
    local beam_size=""
    local timestamps=false
    local words=false
    local verbose=false
    local input_path=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -o|--output)
                output_dir="$2"
                shift 2
                ;;
            -l|--language)
                language="$2"
                shift 2
                ;;
            -b|--beam-size)
                beam_size="$2"
                shift 2
                ;;
            -t|--timestamps)
                timestamps=true
                shift
                ;;
            -w|--words)
                words=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                log_error "未知选项: $1"
                echo ""
                show_help
                exit 1
                ;;
            *)
                if [ -z "$input_path" ]; then
                    input_path="$1"
                else
                    log_error "只能指定一个输入路径"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # 检查输入路径（如果指定了的话）
    if [ -n "$input_path" ] && [ ! -e "$input_path" ]; then
        log_error "输入路径不存在: $input_path"
        exit 1
    fi

    # 检查参数冲突
    if [ "$timestamps" = true ] && [ "$words" = true ]; then
        log_error "不能同时指定 --timestamps 和 --words"
        exit 1
    fi

# ============================================================
# 配置动态库路径（支持 CUDA/cuDNN）
# ============================================================
echo "🔧 配置动态库路径..."

# 1. 自动获取 nvidia 相关库的路径
CUDNN_LIB_PATH=$(python3 -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null)
CUBLAS_LIB_PATH=$(python3 -c "import nvidia.cublas; import os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), 'lib'))" 2>/dev/null)

# 2. 将路径加入 LD_LIBRARY_PATH
if [ -n "$CUDNN_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDNN_LIB_PATH:$LD_LIBRARY_PATH
    echo "   ✅ cuDNN 库路径: $CUDNN_LIB_PATH"
else
    echo "   ⚠️  未找到 cuDNN 库（如需 GPU 加速 ASR，请安装 nvidia-cudnn-cu12）"
fi

if [ -n "$CUBLAS_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$CUBLAS_LIB_PATH:$LD_LIBRARY_PATH
    echo "   ✅ cuBLAS 库路径: $CUBLAS_LIB_PATH"
fi

# 3. 将 Conda 自身的库路径也加上（双重保险）
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    echo "   ✅ Conda 库路径: $CONDA_PREFIX/lib"
fi

echo ""

# 依赖检查
check_dependencies

    # 构建命令
    local cmd="python3 $PYTHON_SCRIPT"
    if [ -n "$input_path" ]; then
        cmd="$cmd --input \"$input_path\""
    fi

    if [ -n "$output_dir" ]; then
        cmd="$cmd --output \"$output_dir\""
    fi

    if [ -n "$language" ]; then
        cmd="$cmd --language \"$language\""
    fi

    if [ -n "$beam_size" ]; then
        cmd="$cmd --beam-size \"$beam_size\""
    fi

    if [ "$timestamps" = true ]; then
        cmd="$cmd --timestamps"
    fi

    if [ "$words" = true ]; then
        cmd="$cmd --words"
    fi

    if [ "$verbose" = true ]; then
        cmd="$cmd --verbose"
    fi

    # 显示执行信息
    log_info "启动 Faster-Whisper ASR 处理..."
    if [ -n "$input_path" ]; then
        log_info "输入路径: $input_path"
    else
        log_info "输入路径: 使用配置文件默认路径"
    fi
    if [ -n "$output_dir" ]; then
        log_info "输出目录: $output_dir"
    fi

    # 执行命令
    log_info "执行命令: $cmd"
    echo ""

    eval "$cmd"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_success "处理完成！"
    else
        log_error "处理失败 (退出码: $exit_code)"
        exit $exit_code
    fi
}

# 执行主函数
main "$@"
