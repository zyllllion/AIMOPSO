"""
算法结果缓存管理器
用于保存和加载算法运行结果，避免重复计算
"""

import pickle
import os
import hashlib
import json
from pathlib import Path


class AlgorithmCacheManager:
    """算法结果缓存管理器"""
    
    def __init__(self, cache_dir="algorithm_cache"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self):
        """加载缓存索引"""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        with open(self.cache_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
    
    def _generate_cache_key(self, algorithm_name, scene_id, params, experiment_group=None):
        """
        生成缓存键
        
        Args:
            algorithm_name: 算法名称
            scene_id: 场景ID
            params: 算法参数字典
            experiment_group: 实验组编号 (1=PSO变体组, 2=经典算法组)
        
        Returns:
            缓存键字符串
        """
        # 清理算法名称中的非法文件名字符（Windows: < > : " / \ | ? *）
        safe_algorithm_name = algorithm_name.replace('*', 'star').replace('/', '_').replace('\\', '_')
        safe_algorithm_name = safe_algorithm_name.replace(':', '_').replace('?', '_').replace('"', '_')
        safe_algorithm_name = safe_algorithm_name.replace('<', '_').replace('>', '_').replace('|', '_')
        
        # 将参数转换为可哈希的字符串
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # 添加实验组标识
        if experiment_group is not None:
            group_suffix = f"_group{experiment_group}"
        else:
            group_suffix = ""
        
        return f"{safe_algorithm_name}_scene{scene_id}{group_suffix}_{param_hash}"
    
    def save_result(self, algorithm_name, scene_id, params, result_data, experiment_group=None):
        """
        保存算法结果
        
        Args:
            algorithm_name: 算法名称
            scene_id: 场景ID
            params: 算法参数字典 (如 {'pop_size': 100, 'n_gen': 500, 'seed': 1})
            result_data: 结果数据字典 (如 {'path': ..., 'costs': ..., 'time': ...})
            experiment_group: 实验组编号 (1=PSO变体组, 2=经典算法组)
        """
        cache_key = self._generate_cache_key(algorithm_name, scene_id, params, experiment_group)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # 保存结果数据
        with open(cache_file, 'wb') as f:
            pickle.dump(result_data, f)
        
        # 更新索引
        self.cache_index[cache_key] = {
            'algorithm': algorithm_name,
            'scene_id': scene_id,
            'params': params,
            'file': str(cache_file),
            'timestamp': str(Path(cache_file).stat().st_mtime)
        }
        self._save_cache_index()
        
        print(f"  ✅ 已缓存 {algorithm_name} 的结果: {cache_key}")
    
    def load_result(self, algorithm_name, scene_id, params, experiment_group=None):
        """
        加载算法结果
        
        Args:
            algorithm_name: 算法名称
            scene_id: 场景ID
            params: 算法参数字典
            experiment_group: 实验组编号 (1=PSO变体组, 2=经典算法组)
        
        Returns:
            结果数据字典，如果不存在则返回None
        """
        cache_key = self._generate_cache_key(algorithm_name, scene_id, params, experiment_group)
        
        if cache_key not in self.cache_index:
            return None
        
        cache_file = Path(self.cache_index[cache_key]['file'])
        
        if not cache_file.exists():
            print(f"  ⚠️  缓存文件不存在: {cache_file}")
            return None
        
        try:
            import numpy as np
            import sys
            if not hasattr(np, '_core'):
                np._core = np.core
                sys.modules['numpy._core'] = np.core
                sys.modules['numpy._core.multiarray'] = np.core.multiarray
            with open(cache_file, 'rb') as f:
                result_data = pickle.load(f)
            print(f"  ✅ 从缓存加载 {algorithm_name} 的结果")
            return result_data
        except Exception as e:
            print(f"  ❌ 加载缓存失败: {e}")
            return None
    
    def has_cache(self, algorithm_name, scene_id, params, experiment_group=None):
        """
        检查是否存在缓存
        
        Args:
            algorithm_name: 算法名称
            scene_id: 场景ID
            params: 算法参数字典
            experiment_group: 实验组编号 (1=PSO变体组, 2=经典算法组)
        
        Returns:
            True如果存在缓存，否则False
        """
        cache_key = self._generate_cache_key(algorithm_name, scene_id, params, experiment_group)
        return cache_key in self.cache_index
    
    def clear_cache(self, algorithm_name=None, scene_id=None):
        """
        清除缓存
        
        Args:
            algorithm_name: 算法名称，如果为None则清除所有
            scene_id: 场景ID，如果为None则清除所有场景
        """
        keys_to_remove = []
        
        for cache_key, info in self.cache_index.items():
            should_remove = True
            
            if algorithm_name is not None and info['algorithm'] != algorithm_name:
                should_remove = False
            
            if scene_id is not None and info['scene_id'] != scene_id:
                should_remove = False
            
            if should_remove:
                keys_to_remove.append(cache_key)
                cache_file = Path(info['file'])
                if cache_file.exists():
                    cache_file.unlink()
        
        for key in keys_to_remove:
            del self.cache_index[key]
        
        self._save_cache_index()
        print(f"  🗑️  已清除 {len(keys_to_remove)} 个缓存")
    
    def list_cache(self):
        """列出所有缓存"""
        print("\n" + "="*80)
        print(" 缓存列表")
        print("="*80)
        
        if not self.cache_index:
            print("  (空)")
            return
        
        for cache_key, info in self.cache_index.items():
            print(f"\n📦 {cache_key}")
            print(f"   算法: {info['algorithm']}")
            print(f"   场景: Scene {info['scene_id']}")
            print(f"   参数: {info['params']}")
            print(f"   文件: {info['file']}")


# 使用示例
if __name__ == "__main__":
    # 创建缓存管理器
    cache_mgr = AlgorithmCacheManager()
    
    # 示例：保存结果
    params = {'pop_size': 100, 'n_gen': 500, 'seed': 1}
    result = {'path': [[1, 2, 3]], 'costs': [0.1, 0.2, 0.3, 0.4], 'time': 120.5}
    cache_mgr.save_result('PSO', 2, params, result)
    
    # 示例：加载结果
    loaded = cache_mgr.load_result('PSO', 2, params)
    print(f"加载的结果: {loaded}")
    
    # 示例：列出缓存
    cache_mgr.list_cache()
    
    # 示例：清除缓存
    # cache_mgr.clear_cache('PSO')  # 清除PSO的所有缓存
    # cache_mgr.clear_cache()  # 清除所有缓存
