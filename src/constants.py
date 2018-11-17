
import arima



# tamanho dos dados que serao usados para o treinamento
# do modelo ARIMA e do modelo de redes neurais
# o conjunto de treinamento tera uma porcentagem do conjunto total
PORCENTAGEM: float = 0.80

ARIMA_PARAMS = arima.get_arima_params(
  p_values=range(0, 7), 
  d_values=range(0, 4), 
  q_values=range(0, 4)
)

# definicao dos parametros usados em todas as redes neurais MLP
def get_mlp_model_configs():
  # define scope of configs
  n_input = [12]
  n_nodes = [5, 10] # , 25, 50, 100]
  n_epochs = [50, 100] # , 500, 1000, 2000]
  n_batch = [1, 10, 25] # , 50, 100, 150]
  n_layers = [2] # , 4, 8]

	# create configs
  configs = list()

  for i in n_input:
    for nodes in n_nodes:
      for epochs in n_epochs:
        for batch in n_batch:
          for layers in n_layers:
            cfg = [i, nodes, epochs, batch, layers]
            configs.append(cfg)

  print('Total de configuracoes a serem testadas: %d' % len(configs))
  return configs
