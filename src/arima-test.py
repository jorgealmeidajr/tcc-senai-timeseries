
import unittest
import arima


class Test_Arima(unittest.TestCase):

  def test_get_arima_params(self):
    p_values = range(0, 1)
    d_values = range(0, 1)
    q_values = range(0, 1)
    
    self.assertEqual(arima.get_arima_params(p_values, d_values, q_values), [(0, 0, 0)])
    self.assertEqual(len(arima.get_arima_params(p_values, d_values, q_values)), 1)

    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)
    
    self.assertEqual(
      arima.get_arima_params(p_values, d_values, q_values), 
      [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
    self.assertEqual(len(arima.get_arima_params(p_values, d_values, q_values)), 8)
  

  def test_split_dataset(self):
    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    train, test = arima.split_dataset(dataset, 0.5)
    self.assertEqual(train, [0, 1, 2, 3, 4])
    self.assertEqual(test, [5, 6, 7, 8, 9, 10])

    train, test = arima.split_dataset(dataset, 0.75)
    self.assertEqual(train, [0, 1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(test, [8, 9, 10])

    train, test = arima.split_dataset(dataset, 0.9)
    self.assertEqual(train, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    self.assertEqual(test, [9, 10])
    

if __name__ == '__main__':
    unittest.main()
