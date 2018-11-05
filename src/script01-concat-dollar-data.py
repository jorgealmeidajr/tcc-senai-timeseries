
import re
import os
import datetime

# script para gerar um arquivo com os dados da serie temporal
# no arquivo us-dollar-vs-brazilian-real-rate.csv
# e juntar com os dados do arquivo 
# us-dollar-vs-brazilian-real-rate-20181104.html

def main():
  missing_data = load_missing_data()
  missing_data = sorted(missing_data, key = lambda i: i['date'], reverse=False)

  missing_data_str: str = ''

  for m in missing_data:
    missing_data_str = missing_data_str + m['date'] + ',' + m['rate'] + '\n'

  original_data: str = load_original_data()

  new_data = original_data.rstrip() + '\n' + missing_data_str
  write(new_data)


def load_missing_data():
  script_path = os.path.abspath(__file__)
  path = script_path.rpartition('\\src\\')[0]

  rel_path = "data\\us-dollar-vs-brazilian-real-rate-20181104.html"
  abs_file_path = os.path.join(path, rel_path)

  file = open(abs_file_path, 'r')

  result = []
  valor: str = ''
  data: str = ''

  for line in file.readlines():
    line = line.strip()
    
    m = re.search(r'1\s+USD\s+=\s+(\d+.*)\s+BRL', line)
    if m:
      #print(line)
      #print('Match found: ', m.group(1))
      valor = m.group(1)
      continue
        
    m = re.search(r'USD\s+BRL\s+rate\s+for\s+(\d+/\d+/\d+)', line)
    if m:
      data = m.group(1)
      
      # fazer a leitura da cotacao ate o dia 22/09 
      # ignorar o resto
      if data == '21/09/2018':
        break

      data = datetime.datetime.strptime(data, '%d/%m/%Y').strftime('%Y-%m-%d')

      result.append({'date': data, 'rate': valor})
      valor = ''
      data = ''
      
      continue

  return result


def load_original_data():
  script_path = os.path.abspath(__file__)
  path = script_path.rpartition('\\src\\')[0]

  rel_path = "data\\us-dollar-vs-brazilian-real-rate.csv"
  abs_file_path = os.path.join(path, rel_path)

  file = open(abs_file_path, 'r')
  text = file.read()

  return text


def write(output):
  script_path = os.path.abspath(__file__)
  path = script_path.rpartition('\\src\\')[0]

  rel_path = "output\\timeseries01.csv"
  abs_file_path = os.path.join(path, rel_path)

  file = open(abs_file_path, 'w')
  file.write(output)
  file.close()


if __name__ == '__main__':
  main()
