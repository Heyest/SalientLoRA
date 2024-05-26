public_server="http://lmflow.org:5000"
echo "downloading alpaca dataset"
wget http://lmflow.org:5000/alpaca.tar.gz
tar zxvf ${filename}
rm ${filename}
