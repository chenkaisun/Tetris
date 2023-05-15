
from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS

all_month_oneie_dir="data/sent_track/allmonth_oneie/"

mftc_dir="data/MFTC/"
download_dir="D:/Downloads/"
mftc_data_path="data/MFTC/MFTC_V4_text.json"
mftc_sandy_dir=f"{mftc_dir}Sandy/"
mftc_all_dir=f"{mftc_dir}All/"
mftc_allfreval_dir=f"{mftc_dir}All_fr_eval/"
mftc_All_fr_eval_dir=f"{mftc_dir}All_fr_eval/"
mftc_fr_path=f"{mftc_allfreval_dir}sampled_twitter_preview.json"
fr_raw_data_path=f"{mftc_allfreval_dir}sampled_twitter_preview.json"

user_mvp_dir="data/user_mvp/"
user_mvp_dir_all="data/user_mvp/full/"

sent_track_dir="data/sent_track/"


NS_tmp_dir="..\\data\WikiHow\\tmp\\"
NS_tmp_dir_filtered="..\\data\WikiHow\\tmp\\filtered\\"
NS_cgen_dir="data/wikihow/full_cgen/"
NS_ss9_dir="data/wikihow/subset9/"
NS_ss9_grounded_dir=f"{NS_ss9_dir}grounded/"
NS_ss9_train=f"{NS_ss9_dir}data_train.json"




abbr2category = {"food": "Food and Entertaining",
                                          "cv": "Cars & Other Vehicles", "fb":"Finance and Business"}