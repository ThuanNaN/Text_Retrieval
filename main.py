import argparse
import numpy as np
from utils.ultis import save_result
from model import lsi, vsm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Text retrieval - CS419')
    
    parser.add_argument(
        "--path_docs",  
        type=str,
        nargs="?", 
        help="Path of documents folder", 
        default= "./Cranfield"
    )
        
    parser.add_argument(
        "--path_query",  
        type=str,
        nargs="?", 
        help="Path of query file (.txt)", 
        default= "./TEST/query.txt"
    
    )
    parser.add_argument(
        "--path_folder_save",  
        type=str,
        nargs="?",
        help="Path of folder save result (.txt)", 
        default= "./Result"
    )

    parser.add_argument(
        "--topk", 
        type=int,  
        nargs="?",
        help="Top K result will return", 
        default=1400
    )

    parser.add_argument(
        "--model", 
        type=str,  
        nargs="?",
        help="Choose model", 
        default="vsm"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    path_docs = args.path_docs
    path_query =  args.path_query

    if args.model == "lsi":
        preds = lsi(path_docs, path_query, topk =  args.topk)
    else:
        preds = vsm(path_docs, path_query, topk =  args.topk)

    save_result(args.path_folder_save, preds)
