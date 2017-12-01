def del_scolumns(csv_path, encoding='utf8'):
  with open(csv_path, 'r', encoding=encoding) as fr:
    lines = fr.readlines()
    with open(csv_path, 'w', encoding=encoding) as fw:
      for line in lines:
        if len(line.split(",")) < 26:
          fw.write(line.replace("'", ""))

if __name__ == '__main__':
  del_scolumns('chronic_kidney_disease.csv')
  del_scolumns('chronic_kidney_disease_full.csv')
