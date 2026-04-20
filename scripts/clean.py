"""
清理项目临时文件和缓存

用法: python scripts/clean.py
"""

import shutil
from pathlib import Path


def clean_directory(patterns):
    """清理匹配的文件和目录"""
    project_root = Path(".")
    
    for pattern in patterns:
        for path in project_root.rglob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"🗑️  删除目录: {path}")
                else:
                    path.unlink()
                    print(f"🗑️  删除文件: {path}")
            except Exception as e:
                print(f"⚠️  无法删除 {path}: {e}")


def main():
    """主函数"""
    print("🧹 开始清理项目...")
    print("="*60)
    
    patterns_to_clean = [
        # Python
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "*.egg-info",
        "dist",
        "build",
        
        # Jupyter
        ".ipynb_checkpoints",
        "*.ipynb_checkpoints",
        
        # IDE
        ".idea/workspace.xml",
        ".idea/tasks.xml",
        ".idea/shelf",
        
        # Model files
        "*.pth",
        "*.pt",
        "*.pkl",
        "*.h5",
        "*.hdf5",
        "*.onnx",
        
        # Logs
        "*.log",
        "logs",
        
        # Temporary
        "tmp",
        "temp",
        "*.tmp",
    ]
    
    clean_directory(patterns_to_clean)
    
    print("="*60)
    print("✅ 清理完成！")


if __name__ == "__main__":
    main()
