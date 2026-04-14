# Teoria Sterowania w Robotyce - Projekt
## Opis projektu
Projekt koncentruje się na implementacji i analizie algorytmu sterowania Pure Pursuit dla symulowanego bolidu wyścigowego.
## Cele
Głównym celem jest zbadanie wpływu parametru Lookahead Distance $L_d$ na stabilność i precyzję śledzenia trajektorii przy różnych prędkościach oraz krzywiznach toru. System wykorzystuje moduł Reinforcement Learning do bieżącej optymalizacji parametrów kontrolera, w tym dynamicznego dobierania $L_d$. Skuteczność sterowania na granicy przyczepności opon zapewniają dwie heurystyki: prędkościowa (LVS) oraz krzywiznowa (CBA).
## Milestones:
### Fundamenty i Środowisko
* Implementacja geometrii toru – stworzenie klasy Track wykorzystującej Shapely do reprezentacji linii centralnej i granic toru.
* Kinematyczny model pojazdu – uruchomienie uproszczonej fizyki, aby przetestować podstawowy algorytm Pure Pursuit.
* Wizualizacja – stworzenie silnika animacji w Matplotlib wyświetlającego bolid i punkt Lookahead w czasie rzeczywistym.
### Dynamika
* Implementacja Bicycle Model – zaprogramowanie nieliniowych równań stanu $\dot{\overline{x}}$ uwzględniających siły opon.
* Implementacja Modelu Opon Pacejka – uwzględnienie kątów uślizgu $\alpha_F, \alpha_R$ oraz nasycenia opon.
### Heurystyki i Adaptacyjne $L_d$
* Logika krzywiznowa – stworzenie funkcji mapującej lokalną krzywiznę $\kappa(s)$ na skrócenie dystansu patrzenia.
* Skalowanie prędkościowe – implementacja liniowej i nieliniowej zależności $L_d$ od $v_x$.
* Analiza porównawcza – zebranie danych o błędzie śledzenia dla różnych heurystyk.
### Reinforcement Learning (RL)
* Projektowanie funkcji nagrody – implementacja nagrody promującej czas okrążenia przy zachowaniu stabilności.
* Trening agenta – wykorzystanie algorytmu PPO do nauki optymalnego doboru $L_d$ w dynamicznych warunkach.
## Struktura techniczna
Językiem programowania wykorzystywanym do implementacji projektu jest python 3.1x, wraz z bibliotekami: 
* Numpy 
* Scipy 
* Matplotlib 
* Shapely
## Członkowie projektu
* Hubert Kubiak
* Kacper Łuczak
