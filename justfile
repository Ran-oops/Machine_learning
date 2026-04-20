# Machine Learning 项目 Justfile
# 用法: just <命令>

# 默认显示帮助
default:
    @just --list

# ========== 环境设置 ==========

# 初始化项目环境
init:
    @echo "Initializing project environment..."
    uv sync
    uv tool install pre-commit
    uv tool install rumdl
    uv tool install ruff
    pre-commit install --hook-type commit-msg --hook-type pre-push
    @echo "Environment initialized!"

# 同步依赖
sync:
    uv sync

# 更新依赖
update:
    uv lock --upgrade

# ========== 代码质量 ==========

# 格式化代码
format:
    ruff format src/ scripts/

# 检查代码
lint:
    ruff check src/ scripts/

# 自动修复代码问题
fix:
    ruff check --fix src/ scripts/

# 检查 Markdown
check-md:
    rumdl check .

# 格式化 Markdown
format-md:
    rumdl fmt .

# ========== 测试 ==========

# 运行所有测试
test:
    python test.py

# 运行特定测试
test-module MODULE:
    python -c "from test import test_{{MODULE}}; test_{{MODULE}}()"

# 检查项目健康状态
check:
    python scripts/check_project.py

# ========== 清理 ==========

# 清理临时文件（跨平台）
clean:
    python scripts/clean.py

# 深度清理（包括虚拟环境）
clean-all: clean
    @echo "Removing virtual environment..."
    python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
    @echo "Deep clean complete!"

# ========== 开发工具 ==========

# 启动 Jupyter Lab
lab:
    uv run --with jupyter jupyter lab

# 运行 Python 交互式环境
python:
    python

# 查看项目结构
tree:
    @python -c "import os; [print(f'{root}/' if dirs else f'{os.path.join(root, file)}') for root, dirs, files in os.walk('.') if '.git' not in root and '.venv' not in root and '__pycache__' not in root for file in ([''] if dirs else files)][:50]"

# 统计代码行数
stats:
    @python -c "
import os
def count_lines(path, ext):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(ext):
                try:
                    with open(os.path.join(root, f), 'r', encoding='utf-8') as file:
                        total += len(file.readlines())
                except:
                    pass
    return total

py_lines = count_lines('src', '.py')
nb_count = sum(1 for _, _, files in os.walk('notebooks') for f in files if f.endswith('.ipynb'))
print(f'Python code: {py_lines} lines')
print(f'Notebooks: {nb_count} files')
"

# ========== 构建 ==========

# 构建包
build:
    uv build

# ========== Git ==========

# 创建新功能分支
feature BRANCH:
    git checkout -b feature/{{BRANCH}}

# 创建修复分支
fix BRANCH:
    git checkout -b fix/{{BRANCH}}

# ========== 文档 ==========

# 查看更新日志
changelog:
    @head -50 CHANGELOG.md

# ========== 帮助 ==========

# 显示所有可用命令
help:
    @echo "Machine Learning Project Commands:"
    @echo ""
    @echo "Environment:"
    @echo "  just init       - Initialize project environment"
    @echo "  just sync       - Sync dependencies"
    @echo "  just update     - Update dependencies"
    @echo ""
    @echo "Code Quality:"
    @echo "  just format     - Format code"
    @echo "  just lint       - Lint code"
    @echo "  just fix        - Auto-fix code issues"
    @echo "  just check-md   - Check Markdown"
    @echo "  just format-md  - Format Markdown"
    @echo ""
    @echo "Testing:"
    @echo "  just test       - Run all tests"
    @echo "  just check      - Project health check"
    @echo ""
    @echo "Cleanup:"
    @echo "  just clean      - Clean temporary files"
    @echo "  just clean-all  - Deep clean (including .venv)"
    @echo ""
    @echo "Development:"
    @echo "  just lab        - Start Jupyter Lab"
    @echo "  just python     - Start Python REPL"
    @echo "  just tree       - Show project structure"
    @echo "  just stats      - Code statistics"
    @echo ""
    @echo "Git:"
    @echo "  just feature <name>  - Create feature branch"
    @echo "  just fix <name>      - Create fix branch"
    @echo ""
    @echo "Documentation:"
    @echo "  just changelog  - View changelog"
