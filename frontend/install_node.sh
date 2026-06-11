#!/bin/bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

nvm install 22
nvm use 22
nvm alias default 22

cd /mnt/d/Nhi/VS_code/paper-research/backend/frontend
rm -rf node_modules package-lock.json
npm install
