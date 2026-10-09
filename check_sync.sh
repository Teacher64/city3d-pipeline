cat > ~/check_sync.sh << 'SCRIPT'
#!/bin/bash
BRANCH="${1:-main}"
cd /mnt/data_sda1/PR_3D_SPB/city3d-pipeline

LOCAL=$(git rev-parse HEAD)
echo "=== Локальный HEAD ==="
git log --oneline -1
echo

ALL_OK=true
for remote in origin kitware gitlabcom; do
    echo "=== $remote ==="
    if git fetch "$remote" "$BRANCH" 2>/dev/null; then
        REMOTE=$(git rev-parse "$remote/$BRANCH" 2>/dev/null)
        if [ "$LOCAL" = "$REMOTE" ]; then
            echo "  ✅ синхронизировано ($REMOTE)"
        else
            AHEAD=$(git rev-list --count "$remote/$BRANCH"..HEAD)
            BEHIND=$(git rev-list --count HEAD.."$remote/$BRANCH")
            echo "  ⚠️  впереди: $AHEAD, позади: $BEHIND"
            ALL_OK=false
        fi
    else
        echo "  ❌ fetch failed"
        ALL_OK=false
    fi
done

echo
if $ALL_OK; then
    echo "🎉 Все три репозитория синхронизированы"
else
    echo "⚠️  Расхождения — запустите git pushall"
fi
SCRIPT

chmod +x ~/check_sync.sh