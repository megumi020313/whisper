# Windows / macOS 部署指南（remote + SMP）

本指南只覆盖本仓库中 `remote/` 与 `SMP/` 的 Python 部分，并以“源码可运行”为目标。

> 依赖安装以根目录 [requirements.txt](requirements.txt) 为准（该文件由源码 import 反推生成）。

---

## 0. 通用前置条件

- Python：建议 3.10–3.12（当前已验证可运行环境为 3.12.12）
- Git
- （可选）NVIDIA GPU：Windows 仅在 NVIDIA 驱动 + 对应 PyTorch CUDA wheel 配齐后可用

建议统一使用虚拟环境：

```bash
python -m venv .venv
```

激活虚拟环境：

- Windows PowerShell：
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- macOS Terminal：
  ```bash
  source .venv/bin/activate
  ```

升级 pip：

```bash
python -m pip install -U pip setuptools wheel
```

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

> 说明：根目录 requirements.txt 已锁版本到 1.txt 的可运行环境；如你需要不同 CUDA/平台轮子（尤其 Windows/macOS 的 PyTorch），请以 PyTorch 官方安装命令为准。

---

## 1. 系统级依赖（非常关键）

### 1.1 FFmpeg（SMP 音频格式转换会用到）

- Windows（任选其一）
  - `winget`：
    ```powershell
    winget install Gyan.FFmpeg
    ```
  - `choco`：
    ```powershell
    choco install ffmpeg
    ```
  安装后确保 `ffmpeg -version` 可用。

- macOS（Homebrew）：
  ```bash
  brew install ffmpeg
  ```

### 1.2 libsndfile（soundfile 依赖）

- Windows：一般通过 pip wheel 自带，无需额外安装；若 `import soundfile` 报错，建议用 Conda：
  - Miniconda/Anaconda 安装后：`conda install -c conda-forge libsndfile`

- macOS：通常 pip wheel 可用；如遇到动态库问题：
  ```bash
  brew install libsndfile
  ```

### 1.3 PortAudio（仅当你需要 pyaudio 麦克风示例）

- Windows：建议直接跳过 `pyaudio`（本仓库仅示例代码会用到）。
- macOS：
  ```bash
  brew install portaudio
  ```

---

## 2. GPU 加速说明（可选）

### 2.1 Windows + NVIDIA

本仓库的深度学习主要依赖 PyTorch。Windows 上建议按 PyTorch 官方命令安装 CUDA 版本 wheel（而不是依赖 `nvidia-cudnn-cu12` 这一类 Linux 生态包）。

- CPU 版（最省事）：保持 `requirements.txt` 安装即可
- GPU 版：到 https://pytorch.org/get-started/locally/ 选择对应 CUDA 版本，执行给出的 `pip install torch ... --index-url ...` 命令

### 2.2 macOS（Apple Silicon / Intel）

- Apple Silicon（M1/M2/M3）通常可走 MPS：安装默认 `torch` 后即可（是否启用取决于代码与模型）。
- 若只需要 CPU：保持默认安装即可。

---

## 3. 启动服务

### 3.1 Remote-V2（FastAPI）

启动目录建议在 `remote/remote-v2/`：

```bash
cd remote/remote-v2
python backend/api/app.py
```

或者使用 uvicorn：

```bash
cd remote/remote-v2
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```

访问：
- Swagger: `http://localhost:8000/docs`

> 该服务会尝试加载 `config/server_config.yaml`，以及（若存在）`config/certs/` 下证书；找不到证书会自动降级为 HTTP。

### 3.2 SMP（Flask Web + API）

为保证 `from backend...` 这类绝对导入能正常解析，请在 `SMP/` 目录启动：

```bash
cd SMP
python -m backend.api.app
```

如需生产方式（类 Linux 部署）可参考 `SMP/scripts/web/start_gunicorn.sh`，但注意：
- Windows 上一般不建议使用 `gunicorn`，可改用 `waitress` 或直接 `python -m backend.api.app`

---

## 4. 常见问题排查

- `ModuleNotFoundError: backend ...`
  - 确认你在子项目根目录启动（例如 `cd SMP` 后再运行）。

- `RuntimeError: Couldn't find ffmpeg ...` 或音频转换失败
  - 确认 `ffmpeg` 已安装且在 PATH 中（`ffmpeg -version`）。

- `OSError: libsndfile ...` / `import soundfile` 失败
  - macOS：`brew install libsndfile`
  - Windows：考虑 Conda 安装 `libsndfile`

- GPU 不可用
  - Windows：确认 NVIDIA 驱动、CUDA 版本、以及 PyTorch CUDA wheel 对齐；优先按 PyTorch 官网安装命令。
