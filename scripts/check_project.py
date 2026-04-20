"""
项目健康检查脚本

检查项目结构、依赖、代码质量等
"""

import subprocess
import sys
from pathlib import Path


def check_file_exists(files, description):
    """检查文件是否存在"""
    print(f"\n[文件检查] {description}")
    all_exist = True
    for file in files:
        path = Path(file)
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {file}")
        if not path.exists():
            all_exist = False
    return all_exist


def check_imports():
    """检查关键模块是否能导入"""
    print("\n[模块导入检查]")
    
    imports_to_check = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]
    
    all_ok = True
    for module, name in imports_to_check:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} (not installed)")
            all_ok = False
    
    # 检查项目模块
    sys.path.insert(0, "src")
    try:
        from machine_learning import BPNN
        print("  [OK] machine_learning module")
    except ImportError as e:
        print(f"  [FAIL] machine_learning module: {e}")
        all_ok = False
    
    try:
        from utils import load_fashion_mnist
        print("  [OK] utils module")
    except ImportError as e:
        print(f"  [FAIL] utils module: {e}")
        all_ok = False
    
    return all_ok


def check_directory_structure():
    """检查目录结构"""
    print("\n[目录结构检查]")
    
    required_dirs = [
        "notebooks",
        "src/machine_learning",
        "src/utils",
        "docs",
        "assets/images",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {dir_path}/")
        if not path.exists():
            all_exist = False
    
    return all_exist


def check_code_quality():
    """检查代码质量"""
    print("\n[代码质量检查]")
    
    checks = [
        ("ruff check src/", "Ruff linting"),
    ]
    
    results = []
    for cmd, name in checks:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK] {name}")
            results.append(True)
        else:
            print(f"  [WARN] {name}")
            results.append(False)
    
    return all(results)


def main():
    """主函数"""
    print("=" * 60)
    print("Project Health Check")
    print("=" * 60)
    
    config_files = [
        "pyproject.toml",
        "README.md",
        ".gitignore",
        "justfile",
    ]
    
    checks = [
        (lambda: check_file_exists(config_files, "Configuration Files"), "Config"),
        (check_directory_structure, "Directory Structure"),
        (check_imports, "Dependencies"),
        (check_code_quality, "Code Quality"),
    ]
    
    results = []
    for check_func, name in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {name} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("All checks passed! Project is healthy.")
    else:
        print("Some checks failed. Please review the output above.")
    print("=" * 60)
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
