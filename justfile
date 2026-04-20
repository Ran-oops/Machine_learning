# Machine Learning 项目 Justfile
# 用法: just <命令>

# 默认显示帮助
default:
    @just --list

# ========== 环境设置 ==========

# 初始化项目环境
init:
    @echo "🚀 初始化项目环境..."
    uv sync
    uv tool install pre-commit
    uv tool install rumdl
    uv tool install ruff
    pre-commit install --hook-type commit-msg --hook-type pre-push
    @echo "✅ 环境初始化完成！"

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

# 清理临时文件
clean:
    python scripts/clean.py

# 深度清理（包括虚拟环境）
clean-all: clean
    @echo "🗑️  删除虚拟环境..."
    rm -rf .venv/
    @echo "✅ 深度清理完成！"

# ========== 开发工具 ==========

# 启动 Jupyter Notebook
notebook:
    jupyter notebook

# 启动 Jupyter Lab
lab:
    jupyter lab

# 运行 Python 交互式环境
python:
    python

# 查看项目结构
tree:
    tree -L 3 -I '.venv|__pycache__|*.pyc|.git' .

# 统计代码行数
stats:
    @echo "📊 代码统计："
    @echo ""
    @echo "Python 文件:"
    find src -name "*.py" -exec wc -l {} + | tail -1
    @echo ""
    @echo "Notebook 文件:"
    find notebooks -name "*.ipynb" | wc -l
    @echo ""
    @echo "总文件数:"
    find . -type f -not -path "./.venv/*" -not -path "./.git/*" | wc -l

# ========== 构建 ==========

# 构建包
build:
    uv build

# 发布到 PyPI (需要配置)
publish:
    uv publish

# ========== Git ==========

# 创建新功能分支
feature BRANCH:
    git checkout -b feature/{{BRANCH}}

# 创建修复分支
fix BRANCH:
    git checkout -b fix/{{BRANCH}}

# 安全的合并到 main (需要先确认)
merge-to-main BRANCH:
    @echo "⚠️  即将合并 {{BRANCH}} 到 main"
    @read -p "确认? [y/N] " confirm && [[ $confirm == [yY] ]] || exit 1
    git checkout main
    git pull origin main
    git merge {{BRANCH}}
    git push origin main

# ========== 文档 ==========

# 生成文档 (如使用 Sphinx 等)
docs:
    @echo "📖 文档功能待实现"

# 查看更新日志
changelog:
    @cat CHANGELOG.md | head -50

# ========== 帮助 ==========

# 显示所有可用命令
help:
    @echo "🔧 Machine Learning 项目命令："
    @echo ""
    @echo "环境设置:"
    @echo "  just init       - 初始化项目环境"
    @echo "  just sync       - 同步依赖"
    @echo "  just update     - 更新依赖"
    @echo ""
    @echo "代码质量:"
    @echo "  just format     - 格式化代码"
    @echo "  just lint       - 检查代码"
    @echo "  just fix        - 自动修复代码"
    @echo "  just check-md   - 检查 Markdown"
    @echo "  just format-md  - 格式化 Markdown"
    @echo ""
    @echo "测试:"
    @echo "  just test       - 运行所有测试"
    @echo "  just check      - 项目健康检查"
    @echo ""
    @echo "清理:"
    @echo "  just clean      - 清理临时文件"
    @echo "  just clean-all  - 深度清理（包括虚拟环境）"
    @echo ""
    @echo "开发:"
    @echo "  just notebook   - 启动 Jupyter Notebook"
    @echo "  just lab        - 启动 Jupyter Lab"
    @echo "  just python     - 启动 Python"
    @echo "  just tree       - 查看项目结构"
    @echo "  just stats      - 代码统计"
    @echo ""
    @echo "Git:"
    @echo "  just feature <name>  - 创建功能分支"
    @echo "  just fix <name>      - 创建修复分支"
    @echo ""
    @echo "文档:"
    @echo "  just changelog  - 查看更新日志"
    @echo "  just docs       - 生成文档"
