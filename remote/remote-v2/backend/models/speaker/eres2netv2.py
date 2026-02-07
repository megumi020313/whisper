import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Union

from backend.core.config import get_config
from backend.utils.logger import get_logger
from backend.models.speaker.eres2net_official.ERes2NetV2 import ERes2NetV2 as OfficialERes2NetV2

logger = get_logger()

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MEL_BINS = 80
DEFAULT_BATCH_SIZE = 64


class ERes2NetV2:
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        self.config = get_config()
        self.device = device or self.config.device
        base_dir = Path(self.config.eres2netv2_path)
        default_path = base_dir / "pretrained_eres2netv2.ckpt"
        self.model_path = Path(model_path) if model_path else default_path

        if not self.model_path.exists():
            candidates = (
                list(base_dir.glob("*.ckpt"))
                + list(base_dir.glob("*.bin"))
                + list(base_dir.glob("*.pth"))
                + list(base_dir.glob("*.pt"))
            )
            if candidates:
                self.model_path = candidates[0]

        logger.info(f"Loading ERes2NetV2 model from {self.model_path}")
        logger.info(f"Target device: {self.device}")

        self.model = OfficialERes2NetV2(
            m_channels=64,
            feat_dim=DEFAULT_MEL_BINS,
            embedding_size=192,
            baseWidth=26,
            scale=2,
            expansion=2,
            pooling_func="TSTP",
            two_emb_layer=False,
        )

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        model_keys = set(self.model.state_dict().keys())
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name.replace("module.", "", 1)
            if name not in model_keys:
                continue
            new_state_dict[name] = v

        missing_keys = model_keys.difference(new_state_dict)
        if missing_keys:
            raise RuntimeError(f"Missing keys in checkpoint: {sorted(missing_keys)[:8]} ... total {len(missing_keys)}")

        self.model.load_state_dict(new_state_dict, strict=True)
        logger.info("✅ Model weights loaded perfectly (Strict Mode, official ERes2NetV2).")

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,}")

        self.model = self.model.to(self.device)
        self.model.eval()

        import torchaudio.compliance.kaldi as kaldi

        self.feature_extractor = kaldi.fbank

    def _extract_fbank(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        # torchaudio.compliance.kaldi 运行在 CPU；保持与官方流程一致
        audio_tensor = audio_tensor.cpu()
        feat = self.feature_extractor(
            audio_tensor,
            num_mel_bins=DEFAULT_MEL_BINS,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=DEFAULT_SAMPLE_RATE,
        )
        # 全局 CMVN（与官方 FBank 可选 mean_nor 一致）
        feat = feat - feat.mean(dim=0, keepdim=True)
        return feat.to(self.device)

    def extract_embedding(self, audio: Union[np.ndarray, torch.Tensor], is_batch: bool = False):
        if is_batch:
            return self.extract_batch_embeddings(audio)

        with torch.no_grad():
            if isinstance(audio, np.ndarray):
                wav_tensor = torch.from_numpy(audio).float().to(self.device)
            else:
                wav_tensor = audio.to(self.device)

            feat = self._extract_fbank(wav_tensor)
            input_tensor = feat.unsqueeze(0)
            embedding = self.model(input_tensor)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy()

    def extract_batch_embeddings(
        self,
        audio_batch: Union[List[np.ndarray], np.ndarray, torch.Tensor],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """批量提取声纹特征（分组批处理优化版）
        
        策略：
        1. 提取所有FBank特征
        2. 按特征长度分组（相同长度可以stack）
        3. 每组内进行批量推理
        4. 按原始顺序重组结果
        
        Args:
            audio_batch: 音频列表
            batch_size: 批处理大小
        
        Returns:
            embeddings: numpy数组，形状为(N, 192)
        """
        if audio_batch is None:
            return np.array([])
        if isinstance(audio_batch, list) and len(audio_batch) == 0:
            return np.array([])
        if isinstance(audio_batch, np.ndarray) and audio_batch.size == 0:
            return np.array([])
        if isinstance(audio_batch, torch.Tensor) and audio_batch.numel() == 0:
            return np.array([])

        # ========== 步骤1：提取所有FBank特征 ==========
        fbank_list = []
        valid_indices = []  # 记录有效样本的原始索引
        
        for idx, wav in enumerate(audio_batch):
            try:
                # 转换为tensor
                if isinstance(wav, torch.Tensor):
                    wav_tensor = wav.float()
                else:
                    wav_tensor = torch.from_numpy(np.asarray(wav)).float()
                
                # 提取FBank特征
                feat = self._extract_fbank(wav_tensor)
                fbank_list.append(feat)
                valid_indices.append(idx)
                
            except Exception as e:
                logger.warning(f"样本 {idx} FBank提取失败: {e}")
                continue
        
        if not fbank_list:
            return np.array([])
        
        # ========== 步骤2：按特征长度分组 ==========
        length_groups = {}  # {length: [(feat, original_idx), ...]}
        
        for feat, orig_idx in zip(fbank_list, valid_indices):
            length = feat.shape[0]  # 时间维度长度
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append((feat, orig_idx))
        
        logger.debug(f"批量推理分组: {len(length_groups)} 个长度组，总样本 {len(fbank_list)} 个")
        
        # ========== 步骤3：每组内进行批量推理 ==========
        embeddings_dict = {}  # {original_idx: embedding}
        
        self.model.eval()
        with torch.no_grad():
            for length, group in length_groups.items():
                # 提取该组的所有特征和索引
                group_feats = [item[0] for item in group]
                group_indices = [item[1] for item in group]
                
                # 分批处理（每批最多batch_size个）
                for i in range(0, len(group_feats), batch_size):
                    batch_feats = group_feats[i:i + batch_size]
                    batch_indices = group_indices[i:i + batch_size]
                    
                    # Stack相同长度的特征（现在可以安全stack了）
                    batch_tensor = torch.stack(batch_feats).to(self.device)
                    
                    # 批量推理
                    emb = self.model(batch_tensor)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    
                    # 保存结果（按原始索引）
                    for j, orig_idx in enumerate(batch_indices):
                        embeddings_dict[orig_idx] = emb[j].cpu()
        
        # ========== 步骤4：按原始顺序重组结果 ==========
        if embeddings_dict:
            # 按原始索引排序
            sorted_indices = sorted(embeddings_dict.keys())
            final_embeddings = torch.stack([embeddings_dict[idx] for idx in sorted_indices], dim=0)
            return final_embeddings.numpy()
        
        return np.array([])


FullERes2NetWrapper = ERes2NetV2

