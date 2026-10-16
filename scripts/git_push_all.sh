#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="/mnt/data_sda1/PR_3D_SPB/city3d-pipeline"
BRANCH="${BRANCH:-main}"
cd "$PROJECT_DIR"

COMMIT_MSG="${1:-chore: auto-commit $(date '+%Y-%m-%d %H:%M:%S')}"

echo "==================================================="
echo "=== GIT PUSH ALL ==="
echo "==================================================="
echo "Проект: $PROJECT_DIR"
echo "Ветка:  $BRANCH"
echo ""
echo "--- Remote ---"
git remote -v
echo ""
echo "--- git add -A ---"
git add -A

if git diff --cached --quiet; then
    echo "==> Нет изменений"
else
    echo "--- Файлы ($(git diff --cached --name-only | wc -l)) ---"
    git --no-pager diff --cached --stat | head -40

    if git diff --cached --name-only | grep -E '\.bak-|BAK|__pycache__|^build/|^logs/' >/dev/null; then
        echo ""
        echo "⚠ МУСОР В ИНДЕКСЕ:"
        git diff --cached --name-only | grep -E '\.bak-|BAK|__pycache__|^build/|^logs/'
        read -p "Продолжить? (y/N) " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    fi

    echo ""
    echo "==> Коммит: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG" || exit 1
fi

echo ""
echo "==================================================="
echo "=== PUSH ==="
echo "==================================================="

FAILED=()
for remote in $(git remote); do
    URL=$(git remote get-url "$remote" 2>/dev/null || echo "?")
    echo ""
    echo "--- $remote ($URL) ---"
    if git push "$remote" "$BRANCH" 2>&1; then
        echo "✓ $remote OK"
    else
        echo "✗ $remote FAILED"
        FAILED+=("$remote")
    fi
done

echo ""
echo "==================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ ВСЕ REMOTE СИНХРОНИЗИРОВАНЫ"
else
    echo "⚠ ПРОБЛЕМЫ С: ${FAILED[*]}"
fi
echo "==================================================="
git --no-pager log --oneline -1
