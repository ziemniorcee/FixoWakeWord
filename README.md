# Fikso wake-word detector: raport

## 1. Cel i motywacja

Celem projektu jest zbudowanie malego detektora polskiego slowa aktywujacego "Fikso", ktory dziala na strumieniu audio z mikrofonu. W tym zadaniu najwazniejsza nie jest sama dokladnosc klasyfikacji pojedynczych plikow, ale praktyczne zachowanie systemu: szybki trening, niewielki model, lokalne uruchomienie oraz akceptowalna liczba falszywych aktywacji.

Docelowy scenariusz przypomina lokalny wake-word detector dla aplikacji lub urzadzenia mobilnego. Model powinien analizowac krotkie okna audio, reagowac na wypowiedziane "Fikso" i ignorowac zwykla mowe oraz tlo.

## 2. Powiazanie z materialem kursu

Projekt wykorzystuje podejscie omawiane przy sieciach konwolucyjnych: sygnal audio jest zamieniany na reprezentacje czasowo-czestotliwosciowa, a nastepnie klasyfikowany przez CNN. Zamiast pracowac bezposrednio na probkach fali, model otrzymuje 40-pasmowy log-mel spectrogram.

Taka reprezentacja dobrze pasuje do CNN, poniewaz lokalne wzorce na spektrogramie odpowiadaja fragmentom fonetycznym slowa: zmianom energii w czasie i czestotliwosci. Rozdzialy dotyczace modeli rekurencyjnych sa naturalnym kierunkiem rozszerzenia, ale w obecnym prototypie CNN jest prostsze, szybsze i latwiejsze do wdrozenia.

## 3. Dane

Wszystkie pliki audio sa mono, 16-bit PCM, 16 kHz. Skrypt treningowy sam odkrywa pliki w katalogu `data` i wykonuje staly, stratyfikowany podzial 70/15/15 dla kazdego zrodla danych.

Aktualny zbior treningowy zawiera 5442 pliki:

| Zrodlo | Etykieta | Liczba plikow | Rola |
|---|---:|---:|---|
| `positive_real` | 1 | 1000 | Rzeczywiste nagrania slowa "Fikso" |
| `positive_ai_augmented` | 1 | 405 | Syntetyczne pozytywy osadzone w realistycznych oknach |
| `negative_real` | 0 | 3905 | Mowa, tlo i dzwieki bez slowa aktywujacego |
| `hard_negative_real` | 0 | 132 | Trudne negatywy: podobnie brzmiace slowa i frazy |

Podzial danych w ostatnim treningu:

| Czesc zbioru | Liczba plikow |
|---|---:|
| Train | 3808 |
| Validation | 817 |
| Test | 817 |

Syntetyczne pozytywy sa generowane przez `augment_positive_ai.py`. Skrypt umieszcza slowo w oknie 1,5 s, dodaje losowy offset, zmiane glosnosci, lekki poglos oraz fragmenty realnego tla z negatywow. Dzieki temu model nie uczy sie wylacznie idealnie przycietych probek.

## 4. Model

Model analizuje okna audio o dlugosci 1,5 s. Przed ekstrakcja cech stosowany jest frontend audio dopasowany do mowy:

- preemphasis: 0.97,
- pasmo mel: 120-4800 Hz,
- liczba pasm mel: 40,
- hop streamingowy: 250 ms.

Po ekstrakcji log-mel cech dane trafiaja do malej sieci CNN z trzema blokami konwolucyjnymi: 12, 24 i 32 kanaly. Na koncu znajduje sie klasyfikator binarny, ktory zwraca prawdopodobienstwo wystapienia slowa "Fikso" w danym oknie.

Checkpoint `checkpoints/fikso_cnn.pt` ma 357849 bajtow, czyli okolo 350 KiB. Jest to rozmiar odpowiedni dla lokalnego prototypu i dalszych testow na urzadzeniu mobilnym.

## 5. Logika streamingowa

W trybie streamingowym system przesuwa okno co 250 ms i liczy wynik modelu dla kolejnych fragmentow audio. Prosta bramka RMS pomija bardzo ciche fragmenty. Sama pojedyncza wysoka predykcja nie wystarcza do detekcji: system wymaga kilku kolejnych trafien powyzej progu, a po wykryciu wlacza cooldown, zeby nie zglaszac wielu aktywacji dla jednej wypowiedzi.

Aktualnie skalibrowane ustawienia streamingowe:

| Parametr | Wartosc |
|---|---:|
| Prog streamingowy | 0.815 |
| Wymagane kolejne trafienia | 2 |
| Recall streamingowy na realnych pozytywach | 0.705 |
| Realne audio negatywne | 197.754 min |
| Falszywe alarmy | 251 |
| Falszywe alarmy na godzine | 76.155 |

Te wyniki pokazuja, ze system dziala jako prototyp, ale nie jest jeszcze gotowy produkcyjnie. Najwiekszym problemem pozostaje liczba falszywych alarmow na dlugich nagraniach tla.

## 6. Trening

Ostatni trening zostal uruchomiony z ziarnem losowym 42. Model byl trenowany przy uzyciu optymalizatora Adam oraz binarnej entropii krzyzowej. W trakcie treningu stosowano lekka augmentacje: zmiane glosnosci, szum, poglos oraz mieszanie pozytywow z realnym tlem.

Trening na CPU trwal 285.94 s. Najlepszy prog klasyfikacyjny zostal dobrany na zbiorze walidacyjnym i wyniosl 0.47.

## 7. Wyniki klasyfikacji

Wyniki na zbiorze walidacyjnym:

| Metryka | Wartosc |
|---|---:|
| Prog | 0.47 |
| Accuracy | 0.9070 |
| Precision | 0.7733 |
| Recall | 0.9052 |
| F1 | 0.8341 |
| TP / FP / FN / TN | 191 / 56 / 20 / 550 |

Wyniki na zbiorze testowym:

| Metryka | Wartosc |
|---|---:|
| Prog | 0.47 |
| Accuracy | 0.9143 |
| Precision | 0.7787 |
| Recall | 0.9336 |
| F1 | 0.8491 |
| TP / FP / FN / TN | 197 / 56 / 14 / 550 |

Wysoki recall oznacza, ze model znajduje wiekszosc pozytywnych probek. Nizsza precision pokazuje jednak, ze nadal zbyt wiele negatywow otrzymuje wysoki wynik. To dobrze tlumaczy roznice miedzy metrykami klasyfikacji a zachowaniem streamingowym.

## 8. Analiza bledow

Lista bledow w `results/metrics.json` pokazuje dwa glowne wzorce.

Pierwszy problem to falszywe pozytywy. Kilka plikow z `hard_negative_real` otrzymuje bardzo wysokie prawdopodobienstwo, np. 0.9487, 0.9536 albo 0.9736. Oznacza to, ze podobnie brzmiace slowa i frazy sa nadal najtrudniejszym przypadkiem dla modelu.

Drugi problem to falszywe negatywy. Czesc realnych wypowiedzi "Fikso" dostaje bardzo niski wynik, np. `positive_real_0146.wav` ma 0.0654, `positive_real_0607.wav` ma 0.0670, a `positive_real_0885.wav` ma 0.0197. Prawdopodobne przyczyny to roznice miedzy mowcami, odleglosc od mikrofonu, szum, zbyt cicha probka albo niekorzystne ulozenie slowa w oknie.

Najwazniejszy wniosek: dobra klasyfikacja izolowanych klipow nie wystarcza. Wake-word detector trzeba oceniac na dlugich nagraniach tla, bo to one ujawniaja realna liczbe falszywych aktywacji.

## 9. Co zadzialalo

Najbardziej pomocne elementy projektu:

- dodanie realnych nagran pozytywnych i negatywnych,
- osobny zbior trudnych negatywow,
- realistyczne osadzanie syntetycznych pozytywow w oknach audio,
- log-mel spectrogram jako wejscie dla malego CNN,
- synchronizacja frontendu audio miedzy Pythonem i eksportem Android,
- kalibracja streamingowa na realnym tle, a nie tylko na pojedynczych plikach.

## 10. Ograniczenia i dalsze kroki

Aktualny system nalezy traktowac jako dzialajacy prototyp. Detektor rozpoznaje slowo "Fikso" i ma lekki model, ale false alarms per hour nadal sa zbyt wysokie dla komfortowego uzycia.

Najbardziej sensowne nastepne eksperymenty:

1. Odsluchac falszywe alarmy z dlugich nagran negatywnych i pogrupowac je wedlug przyczyny.
2. Dograc wiecej podobnie brzmiacych slow i fraz do `hard_negative_real`.
3. Zwiekszyc roznorodnosc realnych pozytywow: inni mowcy, rozne mikrofony, rozne odleglosci.
4. Przetestowac bardziej konserwatywne ustawienia streamingowe, nawet kosztem recall.
5. Porownac obecny CNN z wariantem CNN + GRU albo modelem z wiekszym kontekstem czasowym.

## 11. Podsumowanie

Projekt pokazuje pelny pipeline budowy wake-word detectora: od danych audio, przez augmentacje i trening CNN, po streamingowa kalibracje oraz pomiar falszywych alarmow. Najwieksza lekcja jest praktyczna: w systemach aktywowanych glosem metryki na zbiorze testowym sa tylko pierwszym krokiem. Ostateczna jakosc zalezy od zachowania na dlugim, realistycznym tle i od kontroli falszywych aktywacji.
