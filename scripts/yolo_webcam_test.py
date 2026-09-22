"""
Script de test autonome (etape 0).

Ouvre la webcam, charge un modele YOLO-World avec un vocabulaire
d'objets personnalisable, et affiche le flux video avec les
detections dessinees dessus. Sert a valider que la detection
fonctionne reellement avant de construire le reste du projet.

Usage:
    python scripts/yolo_webcam_test.py
    python scripts/yolo_webcam_test.py --classes "rock,tool,helmet"
    python scripts/yolo_webcam_test.py --camera 1 --conf 0.1

Appuyer sur 'q' pour quitter la fenetre.
"""

import argparse
import sys

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Test webcam + YOLO-World")
    parser.add_argument(
        "--classes",
        default="stone,small rock,grey rock",
        help="Vocabulaire a detecter, separe par des virgules (defaut: stone,small rock,grey rock)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Index de la webcam (defaut: 0)")
    parser.add_argument(
        "--model",
        default="yolov8s-worldv2.pt",
        help="Nom du modele YOLO-World (telecharge automatiquement)",
    )
    parser.add_argument("--conf", type=float, default=0.05, help="Seuil de confiance (defaut: 0.05)")
    return parser.parse_args()


def main():
    args = parse_args()
    vocabulary = [c.strip() for c in args.classes.split(",") if c.strip()]

    # flush=True partout : la sortie est parfois redirigee vers un fichier
    # (ex: python -u script.py | tee log.txt) et sans ca, les lignes restent
    # bloquees dans le buffer tant que le programme ne se termine pas proprement.
    print(f"[INFO] Chargement du modele {args.model} ...", flush=True)
    model = YOLO(args.model)
    model.set_classes(vocabulary)
    print(f"[INFO] Vocabulaire actif : {vocabulary}", flush=True)

    print(f"[INFO] Ouverture de la camera index {args.camera} ...", flush=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam. Verifie l'index et les permissions.", flush=True)
        sys.exit(1)

    print("[INFO] Flux ouvert. Appuie sur 'q' dans la fenetre video pour quitter.", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ATTENTION] Lecture d'image echouee, on reessaie...", flush=True)
                continue

            results = model.predict(frame, conf=args.conf, verbose=False)
            annotated = results[0].plot()

            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"[DETECTION] {label} ({conf:.2f})", flush=True)

            cv2.imshow("VIGIL - test YOLO-World", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
