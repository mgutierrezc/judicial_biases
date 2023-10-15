import pandas as pd
import numpy as np
import sys
import tqdm, json
from tqdm import tqdm
from cleaning_tools.corpus_cleaner_funcs import *
from cleaning_tools.ner_gender_detection import *
from create_judges_files import *

maxInt = sys.maxsize
parameters_json = "./parameters.json"

if __name__ == "__main__":
    parameters = json.load(open(parameters_json))  

    local_path = parameters["paths"]["unix_paths"]["out_directory"]
    judges_subfolder = local_path + "/judges_dfs"

    # obtaining judges names
    judges_list = []
    for year in tqdm.tqdm(range(2018, 2023)):
        judges = read_pickle(judges_subfolder, f"judges_{year}")
        judges_list += judges

    # creating unique list of judges
    judges_list = list(set(judges_list))
    create_pickle(judges_list, "judges_list", local_path)

    # stacking all cases at judge level
    case_stacker(judges_subfolder, judges_list)
