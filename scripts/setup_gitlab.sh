#!/bin/bash
# Установка GitLab CE с хранилищем на /mnt/data_sda1/gitlab
# Запускать от michel:  sudo bash setup_gitlab.sh
set -e

GITLAB_ROOT="/mnt/data_sda1/gitlab"

echo "=== 1. Установка зависимостей ==="
sudo apt update
sudo apt install -y curl openssh-server ca-certificates tzdata perl

echo "=== 2. Добавление репозитория GitLab ==="
curl "https://packages.gitlab.com/install/repositories/gitlab/gitlab-ce/script.deb.sh" | sudo bash

echo "=== 3. Установка GitLab CE (создаст пользователя git) ==="
export EXTERNAL_URL="http://localhost:8080"
sudo -E apt install -y gitlab-ce

echo "=== 4. Подготовка директорий на sda1 ==="
sudo mkdir -p $GITLAB_ROOT/{repos,lfs,uploads}
sudo chown -R git:git $GITLAB_ROOT     # git уже существует после шага 3

echo "=== 5. Доступ michel к данным GitLab (по желанию) ==="
sudo usermod -aG git michel

echo "=== 6. Настройка хранилищ на sda1 ==="
sudo tee -a /etc/gitlab/gitlab.rb > /dev/null <<EOF

# --- Хранилища на sda1 ---
gitlab_rails['lfs_storage_path'] = "$GITLAB_ROOT/lfs"
git_data_dirs({
  "default" => {
    "path" => "$GITLAB_ROOT/repos"
  }
})
EOF

sudo gitlab-ctl reconfigure
sudo gitlab-ctl restart

echo "=== 7. Готово ==="
echo "URL:      http://localhost:8080"
echo "Логин:    root"
echo "Пароль:   sudo cat /etc/gitlab/initial_root_password"
echo "Не забудьте: newgrp git  (или перелогиниться), чтобы у michel появился доступ к $GITLAB_ROOT"
