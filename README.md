# Cel
Celem projektu jest zbieranie danych odnośnie pozycji zawodników na podstawie nagrań meczów
# Opis
Modele wykorzystane do realizacji zadania to
- Detekcja zawodników: ultralitics YOLO11m
- Detekcja punktów boiska: ultralytics YOLO11m-pose

Do trenowania modeli wykorzystano:
- Detekcja zawodników: https://www.soccer-net.org/tasks/game-state-reconstruction
- Detyekcja punktów boiska: https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi
youtu
Wizualizacja działania: https://youtu.be/KqoZb4bxlbA

Kolejność wywołania konkretnych elementów znajduje się w Notebooku **full_pipeline.ipynb**