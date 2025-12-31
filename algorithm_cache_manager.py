"""
Algorithm result cache manager
Saves and loads algorithm results to avoid redundant computation
"""

import pickle
import os
import hashlib
import json
from pathlib import Path


class AlgorithmCacheManager:
    """Algorithm result cache manager"""
    
    def __init__(self, cache_dir="algorithm_cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Cache directory path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self):
        """Load cache index"""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index"""
        with open(self.cache_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
    
    def _generate_cache_key(self, algorithm_name, scene_id, params, experiment_group=None):
        """
        Generate cache key
        
        Args:
            algorithm_name: Algorithm name
            scene_id: Scene ID
            params: Algorithm parameters dict
            experiment_group: Experiment group number
        
        Returns:
            Cache key string
        """
        safe_algorithm_name = algorithm_name.replace('*', 'star').replace('/', '_').replace('\\', '_')
        safe_algorithm_name = safe_algorithm_name.replace(':', '_').replace('?', '_').replace('"', '_')
        safe_algorithm_name = safe_algorithm_name.replace('<', '_').replace('>', '_').replace('|', '_')
        
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        if experiment_group is not None:
            group_suffix = f"_group{experiment_group}"
        else:
            group_suffix = ""
        
        return f"{safe_algorithm_name}_scene{scene_id}{group_suffix}_{param_hash}"
    
    def save_result(self, algorithm_name, scene_id, params, result_data, experiment_group=None):
        """
        Save algorithm result
        
        Args:
            algorithm_name: Algorithm name
            scene_id: Scene ID
            params: Algorithm parameters dict
            result_data: Result data dict
            experiment_group: Experiment group number
        """
        cache_key = self._generate_cache_key(algorithm_name, scene_id, params, experiment_group)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result_data, f)
        self.cache_index[cache_key] = {
            'algorithm': algorithm_name,
            'scene_id': scene_id,
            'params': params,
            'file': str(cache_file),
            'timestamp': str(Path(cache_file).stat().st_mtime)
        }
        self._save_cache_index()
        
        print(f"  ✅ Cached {algorithm_name} result: {cache_key}")
    
    def load_result(self, algorithm_name, scene_id, params, experiment_group=None):
        """
        Load algorithm result
        
        Args:
            algorithm_name: Algorithm name
            scene_id: Scene ID
            params: Algorithm parameters dict
            experiment_group: Experiment group number
        
        Returns:
            Result data dict, or None if not found
        """
        cache_key = self._generate_cache_key(algorithm_name, scene_id, params, experiment_group)
        
        if cache_key not in self.cache_index:
            return None
        
        cache_file = Path(self.cache_index[cache_key]['file'])
        
        if not cache_file.exists():
            print(f"  ⚠️  Cache file not found: {cache_file}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                result_data = pickle.load(f)
            print(f"  ✅ Loaded {algorithm_name} from cache")
            return result_data
        except Exception as e:
            print(f"  ❌ Failed to load cache: {e}")
            return None
    
    def has_cache(self, algorithm_name, scene_id, params, experiment_group=None):
        """
        Check if cache exists
        
        Args:
            algorithm_name: Algorithm name
            scene_id: Scene ID
            params: Algorithm parameters dict
            experiment_group: Experiment group number
        
        Returns:
            True if cache exists, False otherwise
        """
        cache_key = self._generate_cache_key(algorithm_name, scene_id, params, experiment_group)
        return cache_key in self.cache_index
    
    def clear_cache(self, algorithm_name=None, scene_id=None):
        """
        Clear cache
        
        Args:
            algorithm_name: Algorithm name, None to clear all
            scene_id: Scene ID, None to clear all scenes
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
        print(f"  🗑️  Cleared {len(keys_to_remove)} cache entries")
    
    def list_cache(self):
        """List all cache entries"""
        print("\n" + "="*80)
        print(" Cache List")
        print("="*80)
        
        if not self.cache_index:
            print("  (empty)")
            return
        
        for cache_key, info in self.cache_index.items():
            print(f"\n📦 {cache_key}")
            print(f"   Algorithm: {info['algorithm']}")
            print(f"   Scene: Scene {info['scene_id']}")
            print(f"   Params: {info['params']}")
            print(f"   File: {info['file']}")


if __name__ == "__main__":
    cache_mgr = AlgorithmCacheManager()
    
    params = {'pop_size': 100, 'n_gen': 500, 'seed': 1}
    result = {'path': [[1, 2, 3]], 'costs': [0.1, 0.2, 0.3, 0.4], 'time': 120.5}
    cache_mgr.save_result('PSO', 2, params, result)
    
    loaded = cache_mgr.load_result('PSO', 2, params)
    print(f"Loaded result: {loaded}")
    
    cache_mgr.list_cache()
