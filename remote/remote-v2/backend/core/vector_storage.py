import json
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import redis
import yaml

from backend.utils.logger import get_logger
from backend.utils.redis_manager import RedisManager

logger = get_logger()


class VectorStorage:
    """向量存储，支持注册/查询/删除和 AS-Norm 计算。
    
    根据配置文件决定使用 Redis 或本地文件存储。
    如果启用 Redis 但不可用，会尝试自动启动（如果配置允许）。
    """

    def __init__(self, config_path: str = "config/model_config.yaml") -> None:
        # 加载配置
        self.config = self._load_config(config_path)
        redis_config = self.config.get("storage", {}).get("redis", {})
        
        self.redis_enabled = redis_config.get("enabled", False)
        self.redis_host = redis_config.get("host", "localhost")
        self.redis_port = redis_config.get("port", 6379)
        self.redis_db = redis_config.get("db", 0)
        self.redis_auto_start = redis_config.get("auto_start", True)
        
        # 初始化存储
        self.use_redis = False
        self.r = None
        
        if self.redis_enabled:
            self._init_redis_storage()
        else:
            logger.info("📁 Redis disabled in config, using local file storage")
            self._init_local_storage()
        
        # 加载 Cohort 矩阵
        self.cohort_matrix = None
        self.cohort_path = self.config.get("storage", {}).get("cohort_path", "data/vector_db/cohort.npy")
        self.top_k = 200
        self._load_cohort()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
            return {}
    
    def _init_redis_storage(self) -> None:
        """初始化 Redis 存储"""
        logger.info(f"🔄 Attempting to use Redis storage (host={self.redis_host}, port={self.redis_port})")
        
        # 使用 RedisManager 确保 Redis 可用
        redis_manager = RedisManager(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db
        )
        
        available, message = redis_manager.ensure_redis_available(auto_start=self.redis_auto_start)
        
        if available:
            # Redis 可用，创建连接
            try:
                pool = redis.ConnectionPool(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    decode_responses=False,
                )
                self.r = redis.Redis(connection_pool=pool)
                self.r.ping()  # 测试连接
                self.use_redis = True
                logger.info("✅ Using Redis for vector storage")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Redis: {e}")
                logger.warning("⚠️ Falling back to local file storage")
                self._init_local_storage()
        else:
            # Redis 不可用，回退到本地存储
            logger.warning(message)
            logger.warning("⚠️ Falling back to local file storage")
            self._init_local_storage()
    
    def _init_local_storage(self) -> None:
        """初始化本地文件存储"""
        local_path = self.config.get("storage", {}).get("local_path", "data/vector_db")
        self.local_storage_path = Path(local_path) / "vectors"
        self.local_storage_path.mkdir(parents=True, exist_ok=True)
        self.users_file = Path(local_path) / "users.pkl"
        
        # ========== 诊断日志：Vector DB 加载 ==========
        print("\n" + "="*80)
        print("🔍 Vector DB 诊断")
        print("="*80)
        print(f"📂 Vector DB 路径: {self.users_file}")
        print(f"📂 文件存在: {self.users_file.exists()}")
        
        if self.users_file.exists():
            try:
                users = self._load_local_users()
                print(f"✅ Vector DB 加载成功")
                print(f"📊 注册用户数: {len(users)}")
                print(f"📊 用户列表: {list(users.keys())}")
                for uid, data in users.items():
                    emb = data.get('embedding')
                    if emb is not None:
                        print(f"   - {uid}: embedding shape = {emb.shape}")
                    else:
                        print(f"   - {uid}: ❌ embedding 为空!")
            except Exception as e:
                print(f"❌ Vector DB 加载失败: {e}")
        else:
            print(f"⚠️ Vector DB 文件不存在，将创建新的数据库")
        
        print("="*80 + "\n")
        logger.info(f"📁 Using local file storage at {self.users_file}")

    def _load_local_users(self) -> Dict:
        """加载本地用户数据"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load local users: {e}")
                return {}
        return {}
    
    def _save_local_users(self, users: Dict) -> None:
        """保存本地用户数据"""
        try:
            with open(self.users_file, 'wb') as f:
                pickle.dump(users, f)
        except Exception as e:
            logger.error(f"Failed to save local users: {e}")
    
    def _load_cohort(self) -> None:
        # ========== 诊断日志：AS-Norm Cohort 加载 ==========
        print("\n" + "="*80)
        print("🔍 AS-Norm Cohort 诊断")
        print("="*80)
        print(f"📂 尝试加载 Cohort 文件: {self.cohort_path}")
        print(f"📂 文件存在: {os.path.exists(self.cohort_path)}")
        
        if os.path.exists(self.cohort_path):
            try:
                self.cohort_matrix = np.load(self.cohort_path)
                print(f"✅ Cohort 文件加载成功")
                print(f"📊 原始形状: {self.cohort_matrix.shape}")
                
                if self.cohort_matrix.ndim == 1:
                    self.cohort_matrix = self.cohort_matrix.reshape(1, -1)
                    print(f"📊 重塑后形状: {self.cohort_matrix.shape}")
                
                norms = np.linalg.norm(self.cohort_matrix, axis=1, keepdims=True)
                self.cohort_matrix = self.cohort_matrix / (norms + 1e-9)
                
                print(f"✅ Cohort 归一化完成")
                print(f"📊 最终形状: {self.cohort_matrix.shape}")
                print(f"📊 Cohort 样本数: {self.cohort_matrix.shape[0]}")
                print(f"📊 特征维度: {self.cohort_matrix.shape[1]}")
                print("="*80 + "\n")
                
                logger.info(
                    f"✅ AS-Norm cohort loaded: {self.cohort_matrix.shape} from {self.cohort_path}"
                )
            except Exception as e:  # pragma: no cover
                print(f"❌ Cohort 加载失败: {e}")
                print("="*80 + "\n")
                logger.error(f"❌ Failed to load cohort: {e}")
                self.cohort_matrix = None
        else:
            print(f"❌ FATAL ERROR: Cohort 文件不存在!")
            print(f"   期望路径: {self.cohort_path}")
            print(f"   这将导致 Z-score 计算失败，所有识别结果置信度极低")
            print("="*80 + "\n")
            logger.warning("⚠️ Cohort file not found. AS-Norm will be disabled.")
            self.cohort_matrix = None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        return float(np.dot(a, b))

    def compute_as_norm_score(self, target_emb: np.ndarray, probe_emb: np.ndarray) -> float:
        """Z-Norm评分（单向归一化，仅针对probe）
        
        Args:
            target_emb: 注册向量
            probe_emb: 测试向量
            
        Returns:
            Z-score
        """
        raw_score = self._cosine_similarity(target_emb, probe_emb)
        if self.cohort_matrix is None:
            return raw_score

        probe_flat = probe_emb.flatten()
        score_p = np.dot(self.cohort_matrix, probe_flat)
        k = min(self.top_k, len(score_p))
        top_k_idx = np.argpartition(score_p, -k)[-k:]
        top_scores = score_p[top_k_idx]
        mean = np.mean(top_scores)
        std = np.std(top_scores)
        return float((raw_score - mean) / (std + 1e-6))
    
    def compute_snorm_score(self, target_emb: np.ndarray, probe_emb: np.ndarray, 
                           target_metadata: Optional[Dict] = None) -> float:
        """S-Norm评分（对称归一化，Z-Norm + T-Norm的平均）
        
        Args:
            target_emb: 注册向量
            probe_emb: 测试向量
            target_metadata: 注册用户的元数据（包含预计算的T-Norm参数）
            
        Returns:
            S-score（对称归一化分数）
        """
        raw_score = self._cosine_similarity(target_emb, probe_emb)
        
        if self.cohort_matrix is None:
            return raw_score
        
        # 1. Z-Norm（针对probe）
        probe_flat = probe_emb.flatten()
        score_p = np.dot(self.cohort_matrix, probe_flat)
        k = min(self.top_k, len(score_p))
        top_k_idx = np.argpartition(score_p, -k)[-k:]
        top_scores_p = score_p[top_k_idx]
        mu_probe = np.mean(top_scores_p)
        sigma_probe = np.std(top_scores_p)
        z_score = (raw_score - mu_probe) / (sigma_probe + 1e-6)
        
        # 2. T-Norm（针对enrollment，从metadata读取预计算值）
        if target_metadata and 'tnorm_mu' in target_metadata and 'tnorm_sigma' in target_metadata:
            mu_enroll = target_metadata['tnorm_mu']
            sigma_enroll = target_metadata['tnorm_sigma']
            t_score = (raw_score - mu_enroll) / (sigma_enroll + 1e-6)
            
            # 3. S-Norm（对称归一化）
            s_score = (z_score + t_score) / 2.0
            return float(s_score)
        else:
            # 降级到Z-Norm
            logger.debug(f"T-Norm params not found in metadata, fallback to Z-Norm")
            return float(z_score)
    
    def check_voiceprint_distinctiveness(self, embedding: np.ndarray, 
                                        avg_threshold: float = 0.4,
                                        max_threshold: float = 0.7) -> Dict[str, Any]:
        """检查声纹显著性（防止"大众脸"声纹）
        
        Args:
            embedding: 声纹特征向量
            avg_threshold: 平均相似度阈值（超过此值认为辨识度低）
            max_threshold: 最大相似度阈值（超过此值认为与某人过于相似）
            
        Returns:
            检查结果字典
        """
        if self.cohort_matrix is None:
            return {
                "passed": True, 
                "reason": "cohort_unavailable",
                "avg_similarity": 0.0,
                "max_similarity": 0.0
            }
        
        # 计算与Cohort的相似度
        emb_flat = embedding.flatten()
        similarities = np.dot(self.cohort_matrix, emb_flat)
        avg_similarity = float(np.mean(similarities))
        max_similarity = float(np.max(similarities))
        
        # 显著性判断
        if avg_similarity > avg_threshold or max_similarity > max_threshold:
            return {
                "passed": False,
                "reason": "low_distinctiveness",
                "avg_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "quality_level": "poor",
                "suggestion": "声音辨识度偏低，建议在安静环境重新录制以提升识别准确率"
            }
        
        # 分级评估
        if avg_similarity < 0.3:
            quality_level = "excellent"
        elif avg_similarity < 0.35:
            quality_level = "good"
        elif avg_similarity < 0.4:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "passed": True,
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "quality_level": quality_level
        }

    def register(self, user_id: str, embedding: np.ndarray, metadata: Optional[Dict] = None, overwrite: bool = False) -> None:
        """注册用户声纹（预计算T-Norm参数用于S-Norm）
        
        Args:
            user_id: 用户ID
            embedding: 声纹特征向量
            metadata: 用户元数据
            overwrite: 是否覆盖已存在的用户
        """
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        
        # 预计算T-Norm参数（用于S-Norm）
        if metadata is None:
            metadata = {}
        
        if self.cohort_matrix is not None:
            try:
                enroll_scores = np.dot(self.cohort_matrix, emb)
                k = min(self.top_k, len(enroll_scores))
                top_k_idx = np.argpartition(enroll_scores, -k)[-k:]
                top_scores = enroll_scores[top_k_idx]
                
                mu_enroll = float(np.mean(top_scores))
                sigma_enroll = float(np.std(top_scores))
                
                # 存储T-Norm参数到metadata
                metadata['tnorm_mu'] = mu_enroll
                metadata['tnorm_sigma'] = sigma_enroll
                
                logger.debug(f"T-Norm params computed: mu={mu_enroll:.4f}, sigma={sigma_enroll:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute T-Norm params: {e}")
        
        if self.use_redis:
            # Redis存储
            key = f"user:{user_id}"
            if (not overwrite) and self.r.exists(key):
                raise ValueError(f"User {user_id} already exists. Set overwrite=True to replace.")
            meta = json.dumps(metadata, ensure_ascii=False)
            self.r.hset(key, mapping={"embedding": emb.tobytes(), "metadata": meta})
        else:
            # 本地文件存储
            users = self._load_local_users()
            if (not overwrite) and user_id in users:
                raise ValueError(f"User {user_id} already exists. Set overwrite=True to replace.")
            users[user_id] = {"embedding": emb, "metadata": metadata}
            self._save_local_users(users)
        
        logger.info(f"User {user_id} registered (overwrite={overwrite}, S-Norm enabled).")

    def identify(self, embedding: np.ndarray, threshold: float, use_as_norm: bool = False, 
                use_snorm: bool = True) -> Tuple[Optional[str], float]:
        """识别说话人
        
        Args:
            embedding: 声纹特征向量
            threshold: 识别阈值
            use_as_norm: 是否使用 AS-Norm 计算 Z-score（默认False）
            use_snorm: 是否使用 S-Norm 计算对称归一化分数（默认True，优先级高于use_as_norm）
            
        Returns:
            (user_id, score): 用户ID和分数（余弦相似度/Z-score/S-score）
        """
        users = self.load_all_vectors_dict()
        if not users:
            return None, float(0.0)

        probe = np.asarray(embedding, dtype=np.float32).flatten()
        best_id: Optional[str] = None
        best_score = -999.0  # 改为负数，因为Z-score/S-score可以是负数
        
        for uid, emb in users.items():
            if use_snorm and self.cohort_matrix is not None:
                # 使用 S-Norm 计算对称归一化分数（优先级最高）
                user_data = self.get_user(uid)
                metadata = user_data.get('metadata', {}) if user_data else {}
                score = self.compute_snorm_score(emb, probe, metadata)
            elif use_as_norm and self.cohort_matrix is not None:
                # 使用 AS-Norm 计算 Z-score
                score = self.compute_as_norm_score(emb, probe)
            else:
                # 使用余弦相似度
                score = self._cosine_similarity(emb, probe)
            
            if score > best_score:
                best_id = uid
                best_score = score

        if best_id is not None and best_score >= threshold:
            return best_id, float(best_score)
        return None, float(best_score)

    def get_user(self, user_id: str) -> Optional[Dict]:
        if self.use_redis:
            key = f"user:{user_id}"
            data = self.r.hgetall(key)
            if not data:
                return None
            emb = None
            meta = {}
            if b"embedding" in data:
                emb = np.frombuffer(data[b"embedding"], dtype=np.float32)
            if b"metadata" in data:
                try:
                    meta = json.loads(data[b"metadata"].decode())
                except Exception:  # pragma: no cover
                    meta = {}
            return {"user_id": user_id, "embedding": emb, "metadata": meta}
        else:
            users = self._load_local_users()
            if user_id in users:
                return {"user_id": user_id, **users[user_id]}
            return None

    def list_users(self) -> List[str]:
        if self.use_redis:
            keys = self.r.keys("user:*")
            return [k.decode().split(":", 1)[1] for k in keys]
        else:
            users = self._load_local_users()
            return list(users.keys())

    def delete(self, user_id: str) -> bool:
        if self.use_redis:
            key = f"user:{user_id}"
            return self.r.delete(key) > 0
        else:
            users = self._load_local_users()
            if user_id in users:
                del users[user_id]
                self._save_local_users(users)
                return True
            return False

    def load_all_vectors_dict(self) -> Dict[str, np.ndarray]:
        if self.use_redis:
            keys = self.r.keys("user:*")
            users: Dict[str, np.ndarray] = {}
            if not keys:
                return users

            pipeline = self.r.pipeline()
            for k in keys:
                pipeline.hget(k, "embedding")
            results = pipeline.execute()

            for k, data in zip(keys, results):
                if data:
                    uid = k.decode().split(":")[-1]
                    emb = np.frombuffer(data, dtype=np.float32)
                    users[uid] = emb
            return users
        else:
            users_data = self._load_local_users()
            return {uid: data["embedding"] for uid, data in users_data.items()}


_VECTOR_STORAGE: Optional[VectorStorage] = None


def get_vector_storage() -> VectorStorage:
    global _VECTOR_STORAGE
    if _VECTOR_STORAGE is None:
        _VECTOR_STORAGE = VectorStorage()
    return _VECTOR_STORAGE