
import arima



# tamanho dos dados que serao usados para o treinamento
# do modelo ARIMA e do modelo de redes neurais
# o conjunto de treinamento tera uma porcentagem do conjunto total
PORCENTAGEM: float = 0.80

ARIMA_PARAMS = arima.get_arima_params(
  p_values=range(0, 7), 
  d_values=range(0, 5), 
  q_values=range(0, 5)
)
