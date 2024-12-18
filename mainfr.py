import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import *

# Charger le modèle YOLO
model = YOLO('yolov8l.pt')

# Fonction pour afficher les coordonnées de la souris
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Charger la vidéo
cap = cv2.VideoCapture('video.mp4')

# Charger les classes COCO
with open("coco.txt", "r") as fichier:
    classes = fichier.read().split("\n")

# Initialisation des compteurs et des lignes
compteur = 0
suivi_voiture = Tracker()
suivi_camion = Tracker()
ligne_montante = 280
ligne_descendante = 300
marge = 5

voiture_montante = {}
voiture_descendante = {}
compteur_voiture_montante = []
compteur_voiture_descendante = []

camion_montant = {}
camion_descendant = {}
compteur_camion_montant = []
compteur_camion_descendant = []

# Boucle principale
while True:
    ret, image = cap.read()
    if not ret:
        cv2.putText(image, "Fin de la vidéo. Appuyez sur une touche pour quitter.",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("RGB", image)
        cv2.waitKey(0)
        break

    compteur += 1
    if compteur % 3 != 0:
        continue

    image = cv2.resize(image, (1020, 500))

    # Prédiction avec YOLO
    resultats = model.predict(image, conf=0.25)
    detections = resultats[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    liste_voitures = []
    liste_camions = []

    # Filtrer les objets détectés
    for _, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        classe = int(row[5])
        nom_classe = classes[classe]

        if 'car' in nom_classe:
            liste_voitures.append([x1, y1, x2, y2])
        elif 'truck' in nom_classe:
            liste_camions.append([x1, y1, x2, y2])

    # Mise à jour du suivi des voitures
    boites_voitures = suivi_voiture.update(liste_voitures)
    for boite in boites_voitures:
        x3, y3, x4, y4, id_voiture = boite
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Voitures montantes
        if ligne_montante - marge < cy < ligne_montante + marge:
            voiture_montante[id_voiture] = (cx, cy)
        if id_voiture in voiture_montante:
            if ligne_descendante - marge < cy < ligne_descendante + marge:
                if id_voiture not in compteur_voiture_montante:
                    compteur_voiture_montante.append(id_voiture)

        # Voitures descendantes
        if ligne_descendante - marge < cy < ligne_descendante + marge:
            voiture_descendante[id_voiture] = (cx, cy)
        if id_voiture in voiture_descendante:
            if ligne_montante - marge < cy < ligne_montante + marge:
                if id_voiture not in compteur_voiture_descendante:
                    compteur_voiture_descendante.append(id_voiture)

    # Mise à jour du suivi des camions
    boites_camions = suivi_camion.update(liste_camions)
    for boite in boites_camions:
        x7, y7, x8, y8, id_camion = boite
        cx, cy = (x7 + x8) // 2, (y7 + y8) // 2

        # Camions montants
        if ligne_montante - marge < cy < ligne_montante + marge:
            camion_montant[id_camion] = (cx, cy)
        if id_camion in camion_montant:
            if ligne_descendante - marge < cy < ligne_descendante + marge:
                if id_camion not in compteur_camion_montant:
                    compteur_camion_montant.append(id_camion)

        # Camions descendants
        if ligne_descendante - marge < cy < ligne_descendante + marge:
            camion_descendant[id_camion] = (cx, cy)
        if id_camion in camion_descendant:
            if ligne_montante - marge < cy < ligne_montante + marge:
                if id_camion not in compteur_camion_descendant:
                    compteur_camion_descendant.append(id_camion)

    # Affichage des lignes et des compteurs
    cv2.line(image, (1, ligne_montante), (1018, ligne_montante), (0, 255, 0), 2)
    cv2.line(image, (3, ligne_descendante), (1016, ligne_descendante), (0, 0, 255), 2)

    cvzone.putTextRect(image, f'V m : {len(compteur_voiture_montante)}', (50, 60), 2, 2)
    cvzone.putTextRect(image, f'V d : {len(compteur_voiture_descendante)}', (50, 160), 2, 2)
    cvzone.putTextRect(image, f'C m : {len(compteur_camion_montant)}', (792, 43), 2, 2)
    cvzone.putTextRect(image, f'C d : {len(compteur_camion_descendant)}', (792, 100), 2, 2)

    cv2.imshow("RGB", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
