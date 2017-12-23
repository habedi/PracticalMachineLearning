all:
	make clean

clean:
	rm -f processed/metadata/*.json
	rm -f processed/messages/*.txt

download:
	rm -f enron_mail_20150507.tar.gz
	wget -c https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
