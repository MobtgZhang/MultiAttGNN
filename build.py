import os
import argparse
import pickle

from src.utils import build_stop_words
from src.utils import build_dictionary,build_embeddings
from src.utils import build_graph
from src.data import Dictionary
from src.utils import build_wrime_dataset
from src.utils import build_aichallenger_dataset,build_weibo4moods,build_clue2020emotions_dataset
from src.utils import build_goemotions_dataset,build_SST5_dataset
from src.utils import build_student

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=str,default="./data",help="The data directory.")
    parser.add_argument("--wrime-label",type=str,default="writer",help="The data directory.")
    parser.add_argument("--result-dir",type=str,default="./result",help="The result directory.")
    parser.add_argument("--embedding-file",type=str,default=None,help="The result directory.")
    parser.add_argument("--dataset",type=str,default="AIChallenger",help="The result directory.")
    parser.add_argument("--stop-words",action='store_false',help="The stop words of the graph.")
    parser.add_argument("--window-size",type=int,default=3,help="The truncate of the graph.")
    parser.add_argument("--wrime-name",type=str,default='ver2',help="The truncate of the graph.")
    parser.add_argument("--weighted-graph",action='store_false',help="The weighted-graph of the dataset.")
    args = parser.parse_args()
    return args
def check_args(args):
    assert os.path.exists(args.data_dir)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if args.stop_words:
        args.stop_words_file = os.path.join(args.result_dir,"stopwords.pkl")
    else:
        args.stop_words_file = None
def check_embeddings(args,lang):
    if lang == 'zh':
        args.embedding_file = "/media/mobtgzhang/Software/Vecs/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"
    elif lang =='ja':
        args.embedding_file = "/media/mobtgzhang/Software/Vecs/jawiki.all_vectors.100d.txt"
    elif lang =='en':
        args.embedding_file = "/media/mobtgzhang/Software/Vecs/glove.twitter.27B.100d.txt"
    else:
        raise ValueError("Unknown %s language embeddings file."%lang)
def main():
    args = get_args()
    check_args(args)
    print(str(args))
    dataset_dir = os.path.join(args.data_dir,args.dataset)
    result_data_dir = os.path.join(args.result_dir,args.dataset)
    if not os.path.exists(result_data_dir):
        os.makedirs(result_data_dir)
    save_train_dataset_file = os.path.join(result_data_dir,"train_dataset.pkl")
    save_validate_dataset_file = os.path.join(result_data_dir,"validate_dataset.pkl")
    save_test_dataset_file = os.path.join(result_data_dir,"test_dataset.pkl")
    exists_flag = os.path.exists(save_train_dataset_file) and \
            os.path.exists(save_validate_dataset_file) and \
                os.path.exists(save_test_dataset_file)
    # build the dataset
    if args.dataset.lower() == 'aichallenger':
        lang = 'zh'
        if not exists_flag:
            build_aichallenger_dataset(dataset_dir,result_data_dir)
    elif args.dataset.lower() == 'clue2020emotions':
        lang = 'zh'
        if not exists_flag:
            build_clue2020emotions_dataset(dataset_dir,result_data_dir)
    elif args.dataset.lower() == 'goemotions':
        lang = 'en'
        if not exists_flag:
            build_goemotions_dataset(dataset_dir,result_data_dir)
    elif args.dataset.lower() == 'sst-5':
        lang = 'en'
        if not exists_flag:
            build_SST5_dataset(dataset_dir,result_data_dir)
    elif args.dataset.lower() == 'wrime':
        lang = 'ja'
        load_dataset_file = os.path.join(dataset_dir,"wrime-%s.tsv"%args.wrime_name)
        if not exists_flag:
            build_wrime_dataset(load_dataset_file,result_data_dir,args.wrime_label)
    elif args.dataset.lower() == 'weibo4moods':
        lang = 'zh'
        load_dataset_file = os.path.join(dataset_dir,"simplifyweibo_4_moods.csv")
        if not exists_flag:
            build_weibo4moods(load_dataset_file,result_data_dir)
    elif args.dataset.lower() == 'student':
        lang = 'zh'
        load_train_dataset_file = os.path.join(dataset_dir,"student-v1.xlsx")
        if not exists_flag:
            build_student(load_train_dataset_file,result_data_dir)
    else:
        raise ValueError("Unknown dataset:%s"%str(args.dataset))
    
    if lang =='zh':
        # create stop words
        embedding_head = True
        load_stop_words_file = os.path.join(args.data_dir,"stopwords_zh.json")
        save_stop_words_file = os.path.join(args.result_dir,"stopwords_zh.pkl")
        if args.stop_words and not os.path.exists(save_stop_words_file):
            build_stop_words(load_stop_words_file,save_stop_words_file,lang)
    elif lang == 'ja':
        embedding_head = True
        # create stop words
        load_stop_words_file = os.path.join(args.data_dir,"stopwords_ja.json")
        save_stop_words_file = os.path.join(args.result_dir,"stopwords_ja.pkl")
        if args.stop_words and not os.path.exists(save_stop_words_file):
            build_stop_words(load_stop_words_file,save_stop_words_file,lang)
    elif lang == 'en':
        embedding_head = False
        # create stop words
        load_stop_words_file = os.path.join(args.data_dir,"stopwords_en.json")
        save_stop_words_file = os.path.join(args.result_dir,"stopwords_en.pkl")
        if args.stop_words and not os.path.exists(save_stop_words_file):
            build_stop_words(load_stop_words_file,save_stop_words_file,lang)
    else:
        raise ValueError("Unknown language")
    # build the dictionary 
    save_chars_dict_file = os.path.join(result_data_dir,"chars_dict.pkl")
    save_words_dict_file = os.path.join(result_data_dir,"words_dict.pkl")
    if not (os.path.exists(save_chars_dict_file) and os.path.exists(save_words_dict_file)):
        save_stopwords_file = os.path.join(args.result_dir,"stopwords_zh.pkl")
        with open(save_stopwords_file,mode="rb") as rfp:
            stop_words = pickle.load(rfp)
        build_dictionary(result_data_dir,stop_words)
        del stop_words
    # build the embedding
    save_embedding_file = os.path.join(result_data_dir,"embedding.pkl")
    check_embeddings(args,lang)
    if not os.path.exists(save_embedding_file):
        build_embeddings(result_data_dir,args.embedding_file,save_embedding_file,embedding_head)
    # build the graph
    graph_names = ['train','validate','test']
    for name in graph_names:
        save_graph_file = os.path.join(result_data_dir,"%s_graph.pkl"%name)
        save_dataset_file = os.path.join(result_data_dir,"%s_dataset.pkl"%name)
        if not os.path.exists(save_graph_file):
            words_dict = Dictionary.load(save_words_dict_file)
            chars_dict = Dictionary.load(save_chars_dict_file)
            build_graph(words_dict,chars_dict,save_dataset_file,save_graph_file,args.window_size,args.weighted_graph)
if __name__ == "__main__":
    main()


