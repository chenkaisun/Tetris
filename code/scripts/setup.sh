#mkdir data
#mkdir data/wikihow
#mkdir data/wikihow/debug
#mkdir data/wikihow/subset1
#mkdir data/wikihow/subset2
#mkdir data/wikihow/subset3
#mkdir data/wikihow/subset4
#mkdir data/wikihow/subset5
#mkdir data/wikihow/subset6
#mkdir data/wikihow/subset7
#mkdir data/wikihow/subset8
#mkdir data/wikihow/subset9
#mkdir data/wikihow/full_cgen
#mkdir data/wikihow/subset10
##cd ..
##mkdir BARTScore



#######Copy transformer data into data since gitignore
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m" -O BARTScore/bart.pth && rm -rf /tmp/cookies.txt

## torch
## geometric
cd ..
python -m pip install -r requirements.txt
python -m spacy download en
python -c "import nltk; nltk.download('stopwords', quiet=True)"
