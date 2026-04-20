"""
项目初始化脚本

用于设置开发环境、安装依赖等
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"⚠️  命令执行失败: {cmd}")
    return result.returncode == 0


def setup_environment():
    """设置开发环境"""
    print("\n🚀 开始设置机器学习项目环境...")
    
    # 检查 Python 版本
    print(f"\n✅ Python 版本: {sys.version}")
    
    # 同步依赖
    if not run_command("uv sync", "同步项目依赖"):
        print("尝试使用 pip...")
        run_command("pip install -e .", "使用 pip 安装依赖")
    
    # 安装开发工具
    print("\n📦 安装开发工具...")
    tools = ["pre-commit", "rumdl", "ruff"]
    for tool in tools:
        run_command(f"uv tool install {tool}", f"安装 {tool}")
    
    # 设置 pre-commit hooks
    if Path(".pre-commit-config.yaml").exists():
        run_command("pre-commit install --hook-type commit-msg --hook-type pre-push", 
                   "设置 pre-commit hooks")
    
    print("\n" + "="*60)
    print("✨ 环境设置完成！")
    print("="*60)
    print("\n可用命令:")
    print("  just init      - 初始化项目")
    print("  just test      - 运行测试")
    print("  just format    - 格式化代码")
    print("  just lint      - 代码检查")
    print("  just clean     - 清理临时文件")


if __name__ == "__main__":
    setup_environment()
