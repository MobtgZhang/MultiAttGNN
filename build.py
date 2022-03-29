import os
import argparse
import pickle

from src.utils import build_dataset,build_stop_words
from src.utils import build_dictionary,build_embeddings
from src.utils import build_graph
from src.data import Dictionary

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=str,default="./data",help="The data directory.")
    parser.add_argument("--result-dir",type=str,default="./result",help="The result directory.")
    parser.add_argument("--embedding-file",type=str,default=None,help="The result directory.")
    parser.add_argument("--dataset",type=str,default="AIChallenger",help="The result directory.")
    parser.add_argument("--stop-words",action='store_false',help="The stop words of the graph.")
    parser.add_argument("--window-size",type=int,default=3,help="The truncate of the graph.")
    parser.add_argument("--truncate",action='store_true',help="The truncate of the graph.")
    parser.add_argument("--max-trunc-len",type=int,default=50,help="The max-trunc-len of the dataset.")
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
    args.embedding_file = "/media/mobtgzhang/Software/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"
def main():
    
    args = get_args()
    check_args(args)
    dataset_dir = os.path.join(args.data_dir,args.dataset)
    result_data_dir = os.path.join(args.result_dir,args.dataset)
    if not os.path.exists(result_data_dir):
        os.makedirs(result_data_dir)
    # build the dataset
    load_train_dataset_file = os.path.join(dataset_dir,"sentiment_analysis_trainingset.csv")
    save_train_dataset_file = os.path.join(result_data_dir,"train.pkl")
    if not os.path.exists(save_train_dataset_file):
        build_dataset(load_train_dataset_file,save_train_dataset_file,sent_name='content')
    load_validate_dataset_file = os.path.join(dataset_dir,"sentiment_analysis_validationset.csv")
    save_validate_dataset_file = os.path.join(result_data_dir,"validate.pkl")
    if not os.path.exists(save_validate_dataset_file):
        build_dataset(load_validate_dataset_file,save_validate_dataset_file,sent_name='content')
    # create stop words
    load_stop_words_file = os.path.join(args.data_dir,"stopwords.txt")
    save_stop_words_file = os.path.join(args.result_dir,"stopwords.pkl")
    if not os.path.exists(save_stop_words_file):
        build_stop_words(load_stop_words_file,save_stop_words_file)

    # build the dictionary 
    save_chars_dict_file = os.path.join(result_data_dir,"chars_dict.pkl")
    save_words_dict_file = os.path.join(result_data_dir,"words_dict.pkl")
    if not (os.path.exists(save_chars_dict_file) and os.path.exists(save_words_dict_file)):
        build_dictionary(result_data_dir)
    
    # build the embedding
    save_embedding_file = os.path.join(result_data_dir,"embedding.pkl")
    if not os.path.exists(save_embedding_file):
        build_embeddings(result_data_dir,args.embedding_file,save_embedding_file)
    # build the graph
    save_train_graph_file = os.path.join(result_data_dir,"train_graph.pkl")
    if not os.path.exists(save_train_graph_file):
        words_dict = Dictionary.load(save_words_dict_file)
        chars_dict = Dictionary.load(save_chars_dict_file)
        build_graph(words_dict,chars_dict,save_train_dataset_file,save_train_graph_file,args.stop_words_file,
                        args.window_size,args.weighted_graph)
    save_validate_graph_file = os.path.join(result_data_dir,"validate_graph.pkl")
    if not os.path.exists(save_validate_graph_file):
        words_dict = Dictionary.load(save_words_dict_file)
        chars_dict = Dictionary.load(save_chars_dict_file)
        build_graph(words_dict,chars_dict,save_validate_dataset_file,save_validate_graph_file,args.stop_words_file,
                        args.window_size,args.weighted_graph)
if __name__ == "__main__":
    main()


