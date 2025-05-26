### Polecenie
Zadanie polega na stworzeniu modelu rekurencyjnego, który przewidywał będzie kompozytora danego utworu muzyki klasycznej w oparciu o jego zapis w formie sekwencji akordów. Akordy znormalizowane zostały do klucza C-dur lub A-mol w zależności od skali utworu (durowa/molowa).

### Dane
Dane dostępne są w folderze onedrive. Dane przygotowane są w postaci Pickli w których znajduje się lista krotek z sekwencjami i odpowiadającymi im klasami - autorami odpowiednio: {0: 'bach', 1: 'beethoven', 2: 'debussy', 3: 'scarlatti', 4: 'victoria'} (train.pkl). W pliku test_no_target znajdują się testowe sekwencje, dla których predykcje mają Państwo przewidzieć.

Uwaga, utwory mogą być oczywiście różnych długości. W celu stworzenia batcha danych różnej długości, muszą je Państwo odpowiednio przygotować
