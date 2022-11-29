# data_compression_xai\par
Data compression for improved explanation estimation

Obecny stan kodu:

  Kod Huberta:
  
  5 eksperymentów, z czego 1 na datasecie diabetes z sklearna (~500 pozycji) i reszta na datasecie o bankach (~45tys). Exp 2 i 3 są na pełnym zbiorze, 4 i 5 bez zmiennych kategorialnych. 3,4,5 używają funkcji compression z goodpoints. 5 eksperyment skupia się na tym, jakie kernel functions są najlepsze.
  
  Dla każdego z eksperymentów trenowany jest XGBoost, który następnie jest wyjaśniany z użyciem a) pełnego zbioru testowego b) zbioru powstałego przez kernel thinning zbioru testowego c)zbioru wielkości poprzedniego powstałego przez wylosowanie z równym prawdopodobieństwem punktów ze zbioru testowego.
  
  Używane metody wyjaśnień: SHAP, PDP,PVI, ALE.
  
  Mierzone: różnice czasów wyjaśnienia pomiędzy zbiorami a-c i a-b, czas kompresji, odległość wyjaśnienia na zbiorze b i c od wyjaśnienia na zbiorze a w metryce L1, odległość wassersteina między rozkładami a i c oraz a i b, dla SHAPA odległość między wyjaśnieniami w metryce wassersteina.
  
  exp5 tldr: kernel gaussian 0.05 najlepszy, czasem gaussian 1, pośrodku średnio.
  exp2-4,6 tldr: SHAP obiecujący zarówno pod względem poprawy dokładności i czasu, PVI wszędzie beznadziejne, PDP/ALE gorszy czas, lepsza dokładność (obiecujące na większym zbiorze?)
  
  6 eksperyment: kod z pierwszych dwóch puszczony na zbiorze Friedmana [1][2] z sklearna. Czas gorszy dla KT - zbiór za łatwy? Inny zbiór może?
  
  
  Stan analizy statystycznej:
  co robił Hubert: histogram i jaki procent eksperymentów wychodzi różnica czasów >0.
  Co my robimy: test t studenta one-tailed dla wyjaśnienia na zbiorach a i c oraz a i b. Dla czasów - różnicy czasów wyjaśnień na zbiorach a i b, a czasu KT. 
  
  Propozycja na checkpoint 9.12: tabelka zbiór na metodę - gdzie jest poprawa zarówno w czasie, jak i jakości względem uniform sampling, gdzie tylko w jednym, gdzie w ogóle.
    
  References
  
[1]J. Friedman, “Multivariate adaptive regression splines”, The Annals of Statistics 19 (1), pages 1-67, 1991.

[2]L. Breiman, “Bagging predictors”, Machine Learning 24, pages 123-140, 1996.
