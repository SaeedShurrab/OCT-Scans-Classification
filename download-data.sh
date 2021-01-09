mkdir $(pwd)/data
mkdir $(pwd)/data/raw
mkdir $(pwd)/data/preprocessed

sudo apt-get install wget

wget -P $(pwd)/data/raw https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/e9a18c27-67da-40a2-8f48-1c12f03de39d

cd $(pwd)/data/raw

mv e9a18c27-67da-40a2-8f48-1c12f03de39d octdata.zip

unzip octdata.zip && rm octdata.zip

echo 'IMPORTANT MESSAGE: the dataset zip file contains a chest X-ray images, dont forget to remove them'
