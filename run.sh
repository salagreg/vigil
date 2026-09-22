#!/usr/bin/env bash
# Lance le serveur VIGIL et le redemarre automatiquement s'il plante.
# Chaque (re)demarrage est journalise avec la date/heure dans logs/server.log.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# La sortie est redirigee vers un fichier : sans ca, Python bufferise ses
# print() et les logs n'apparaissent qu'a l'arret du serveur.
export PYTHONUNBUFFERED=1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a logs/server.log
}

log "=== Lancement de VIGIL ==="

while true; do
  log "Demarrage du serveur (uvicorn)..."
  uvicorn api.main:app --host 0.0.0.0 --port 8000 >> logs/server.log 2>&1
  EXIT_CODE=$?
  log "Le serveur s'est arrete (code $EXIT_CODE). Redemarrage dans 2s..."
  sleep 2
done
