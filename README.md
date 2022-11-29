# data_compression_xai\par
Data compression for improved explanation estimation

Obecny stan kodu:

  Kod Huberta:
  
  5 eksperymentów, z czego 1 na datasecie diabetes z sklearna (~500 pozycji) i reszta na datasecie o bankach (~45tys). Exp 2 i 3 są na pełnym zbiorze, 4 i 5 bez zmiennych kategorialnych. 3,4,5 używają funkcji compression z goodpoints.
  
  Dla każdego z eksperymentów trenowany jest XGBoost, który następnie jest wyjaśniany z użyciem a) pełnego zbioru testowego b) zbioru powstałego przez kernel thinning zbioru testowego c)zbioru wielkości poprzedniego powstałego przez wylosowanie z równym prawdopodobieństwem punktów ze zbioru testowego.
  
  Używane metody wyjaśnień: SHAP, PDP,PVI, ALE.
  
  Mierzone: różnice czasów wyjaśnienia pomiędzy zbiorami a-c i a-b, czas kompresji, odległość wyjaśnienia na zbiorze b i c od wyjaśnienia na zbiorze a w metryce L1, odległość wassersteina między rozkładami a i c oraz a i b, dla SHAPA odległość między wyjaśnieniami w metryce wassersteina.
  
  6 eksperyment: kod z pierwszych dwóch puszczony na zbiorze Friedmana [1][2] z sklearna.
  
  
  References
  
[1]J. Friedman, “Multivariate adaptive regression splines”, The Annals of Statistics 19 (1), pages 1-67, 1991.

[2]L. Breiman, “Bagging predictors”, Machine Learning 24, pages 123-140, 1996.
