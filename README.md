# vigil

VIGIL est une caméra de surveillance intelligente, reliée à un tableau de bord, qui détecte les intrusions à bord et continue de fonctionner quand le cloud est injoignable.

Étape 1/3 : détection en direct de débris spatiaux (rochers, météorites, astéroïdes) et tableau de bord web temps réel. Les personnes et autres objets ne sont pas détectés : seul le thème "débris spatial" est reconnu. Les niveaux d'alerte, le mode autonome et le cloud/MQTT arrivent dans les étapes suivantes.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS : la première exécution demande l'autorisation d'accès à la caméra pour l'application qui lance le terminal (Terminal, VS Code, ...). Accepter la popup système, ou l'activer dans Réglages Système > Confidentialité et sécurité > Caméra.

## Lancement

```bash
./run.sh
```

`run.sh` démarre le serveur et le relance automatiquement s'il plante, en journalisant chaque (re)démarrage dans `logs/server.log`.

Vocabulaire détecté par défaut : `stone, pebble, grey rock, small rock, piece of rock` (modifiable dans `detector/detector.py`, variable `DEFAULT_CLASSES`). Volontairement limité au thème "débris spatial" : les personnes, outils ou autres objets de la scène ne déclenchent aucune détection.

Chaque objet détecté reçoit un identifiant unique (`Rocher #1`, `#2`, ...) et une zone (`GAUCHE`, `DROITE`, `HAUT`, `BAS`, `CENTRE`...) : une alerte n'est déclenchée qu'à la première apparition d'un objet, pas en boucle tant qu'il reste dans le champ. Le radar du dashboard affiche sa position en temps réel (distance approximative basée sur la taille apparente, pas une mesure physique).

Chaque détection confirmée est enregistrée en base avec une miniature de l'image annotée (`snapshots/`, affichée dans l'historique) et, une fois l'objet sorti du champ, sa durée totale de présence.

## Test

1. Ouvrir [http://localhost:8000](http://localhost:8000)
2. Poser un objet devant la caméra (un caillou par défaut) : un cadre de détection doit apparaître sur le flux vidéo, une entrée dans l'historique, et un point sur le radar.
3. `GET /health` doit répondre `{"status": "ok"}`.
4. `GET /events` renvoie l'historique des détections en JSON.

## Structure

```
detector/     boucle de capture + detection (thread dedie, reconnexion camera)
api/          serveur FastAPI, base SQLite (vigil.db)
dashboard/    tableau de bord web (HTML/CSS/JS natif, sans build)
scripts/      script de test autonome (yolo_webcam_test.py)
logs/         journaux du serveur
```
