all:
	make clean
	python3 arff_2_csv.py
	python3 del_surplus_columns.py

clean:
	rm -f *.csv

csv:
	python3 arff_2_csv.py

fix:
	python3 del_surplus_columns.py
