#!/bin/bash
# ============================================================
# 实验前代码同步检查脚本
# 用途：每次运行实验前执行，确保代码是最新的
# 用法：bash scripts/check_git_update.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
WARN=0
FAIL=0

echo "========================================"
echo "  实验前 Git 同步检查"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# ---- 1. 检查是否有远程仓库配置 ----
echo -n "[1/5] 检查远程仓库配置 ... "
REMOTE=$(git remote get-url origin 2>/dev/null || true)
if [ -z "$REMOTE" ]; then
    echo -e "${YELLOW}未配置${NC}"
    echo "       远程仓库 origin 未设置。请先配置："
    echo "         git remote add origin <REPO_URL>"
    echo ""
    WARN=$((WARN + 1))
    SKIP_FETCH=1
else
    echo -e "${GREEN}已配置${NC}"
    echo "       远程地址: $REMOTE"
    SKIP_FETCH=0
fi

# ---- 2. 检查本地未提交修改 ----
echo -n "[2/5] 检查本地未提交修改 ... "
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${GREEN}干净${NC}"
else
    echo -e "${YELLOW}存在修改${NC}"
    echo ""
    git status --short
    echo ""
    echo "       ⚠ 以上文件已被修改但未提交。"
    echo "       如果是维护类修改（文档/路径/依赖），可忽略。"
    echo "       如果非维护类修改，请执行: git stash 或 git restore"
    echo ""
    WARN=$((WARN + 1))
fi

# ---- 3. 从远程拉取最新引用 ----
if [ "$SKIP_FETCH" -eq 0 ]; then
    echo -n "[3/5] 拉取远程最新引用 ... "
    if git fetch origin --quiet 2>&1; then
        echo -e "${GREEN}成功${NC}"
    else
        echo -e "${RED}失败${NC}"
        echo "       git fetch 失败，请检查网络/代理配置。"
        echo ""
        FAIL=$((FAIL + 1))
        SKIP_FETCH=1
    fi
else
    echo "[3/5] 拉取远程最新引用 ... ${YELLOW}跳过（无远程）${NC}"
fi

# ---- 4. 检查本地是否落后于远程 ----
if [ "$SKIP_FETCH" -eq 0 ]; then
    echo -n "[4/5] 比较本地与远程版本 ... "
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    LOCAL=$(git rev-parse HEAD)
    REMOTE_HASH=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "")

    if [ -z "$REMOTE_HASH" ]; then
        echo -e "${YELLOW}远程分支不存在${NC}"
        echo "       远程 origin/$BRANCH 未找到，可能需要首次 push 或设置 upstream。"
        WARN=$((WARN + 1))
    elif [ "$LOCAL" = "$REMOTE_HASH" ]; then
        echo -e "${GREEN}已是最新${NC}"
        echo "       本地与 origin/$BRANCH 一致。"
    else
        # 检查本地是否落后
        BEHIND=$(git rev-list --count HEAD..origin/$BRANCH 2>/dev/null || echo "0")
        AHEAD=$(git rev-list --count origin/$BRANCH..HEAD 2>/dev/null || echo "0")
        if [ "$BEHIND" -gt 0 ]; then
            echo -e "${RED}落后 $BEHIND 个提交！${NC}"
            echo "       ⚠ 远程有更新，请立即执行 git pull 后再运行实验！"
            FAIL=$((FAIL + 1))
        elif [ "$AHEAD" -gt 0 ]; then
            echo -e "${YELLOW}超前 $AHEAD 个提交${NC}"
            echo "       本地有未推送的提交（仅维护类操作允许）。"
            WARN=$((WARN + 1))
        else
            echo -e "${GREEN}已是最新${NC}"
        fi
    fi
else
    echo "[4/5] 比较本地与远程版本 ... ${YELLOW}跳过（无远程）${NC}"
fi

# ---- 5. 检查 Git 用户配置 ----
echo -n "[5/5] 检查 Git 用户配置 ... "
USER_NAME=$(git config user.name 2>/dev/null || echo "")
USER_EMAIL=$(git config user.email 2>/dev/null || echo "")
if [ -z "$USER_NAME" ] || [ "$USER_NAME" = "RGCF Experiment Runner" ]; then
    echo -e "${YELLOW}使用默认值${NC}"
    echo "       用户名: ${USER_NAME:-未设置}"
    echo "       邮箱:   ${USER_EMAIL:-未设置}"
    echo "       如需修改: git config user.name \"Your Name\""
    WARN=$((WARN + 1))
else
    echo -e "${GREEN}已配置${NC}"
fi

# ---- 汇总 ----
echo ""
echo "========================================"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}检查不通过: $FAIL 项失败, $WARN 项警告${NC}"
    echo "  ⛔ 请在运行实验前解决以上问题！"
    echo ""
    if [ "$SKIP_FETCH" -eq 0 ] && [ -n "$BEHIND" ] && [ "$BEHIND" -gt 0 ]; then
        echo "  → 远程有 $BEHIND 个新提交，执行:"
        echo "    git pull origin $BRANCH"
    fi
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo -e "  ${YELLOW}检查通过（$WARN 项警告）${NC}"
    echo "  可以运行实验，但建议关注以上警告。"
    exit 0
else
    echo -e "  ${GREEN}全部通过 ✅${NC}"
    echo "  代码已是最新，可以开始运行实验。"
    exit 0
fi
