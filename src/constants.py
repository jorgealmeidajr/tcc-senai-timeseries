
import arima



# tamanho dos dados que serao usados para o treinamento
# do modelo ARIMA e do modelo de redes neurais
# o conjunto de treinamento tera 75% do conjunto total
PORCENTAGEM: float = 0.75

ARIMA_PARAMS = arima.get_arima_params(
  p_values=range(0, 4), 
  d_values=range(0, 3), 
  q_values=range(0, 3)
)
