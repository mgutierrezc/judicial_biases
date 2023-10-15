import pandas as pd
import numpy as np
import sys, os, re
import csv, tqdm, pickle, json
import sqlite3
from collections import Counter
from tqdm import tqdm
from cleaning_tools.corpus_cleaner_funcs import *
from cleaning_tools.ner_sentiment_detection import *

maxInt = sys.maxsize
parameters_json = "./parameters.json"

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def folder_creator(folder_name: str, path: str) -> None:
    """
    Generates a folder in specified path
                
    input: name of root folder, path where you want 
    folder to be created
    output: None
    """
                                    
    # defining paths
    data_folder_path = path + "/" + folder_name
    data_folder_exists = os.path.exists(data_folder_path)

    # creating folders if don't exist
    if data_folder_exists:
        pass
    else:    
        # create a new directory because it does not exist 
        os.makedirs(data_folder_path)

    print(f"The new directory '{folder_name}' was created successfully! \nYou can find it on the following path: {path}")


def inner_merge(cases_df: pd.core.frame.DataFrame, case_id_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Merges data from cases with judges ids
    """
    return pd.merge(cases_df, case_id_df, how="inner")


def detect_mult_judges(judge_name):
    #print(f"judge: {judge_name}, type: {type(judge_name)}")
    if isinstance(judge_name, str):
        if judge_name.count(",") >= 2 or judge_name.count(".") >= 2:
            return "Multiple"
        else:
            return "Single"
    else:
        return "No name"


def df_filtered_judge(df, judge_name):
    return df[df['Juez:'] == judge_name]


def read_pickle(folder_path: str, file_name: str):
    """
    Reads pickle file on a specific folder
    """
    pickle_file = open(folder_path + f"/{file_name}", "rb")
    pickle_data = pickle.load(pickle_file)
    pickle_file.close()
    return pickle_data


def create_pickle(object_name, file_name: str, path: str) -> None:
    """
    Creates a pickle file for object. Note: Path should have no slash 
    at the end
    """
    with open(path + f"/{file_name}", "wb") as storing_output:
        pickle.dump(object_name, storing_output)
        storing_output.close()


def extract_judges_data(cases_df, judges_list: list, suffix: str, folder_path: str) -> None:
    """
    Creates dataframes for each judge containing all the cases analyzed by them
    and stores them as pickles
    """

    for judge in tqdm.tqdm(judges_list):
        # creating folder per judge 
        judge_folder = re.sub(r"\s+", '-', judge)
        directory = folder_path + "/" + judge_folder
        if not os.path.exists(directory): # only creating if folder not exists
            os.makedirs(directory)
        
        # saving dataframe with judge cases 
        judge_df = cases_df[cases_df['Juez:'] == judge]
        with open(directory + "/" + judge_folder + "-" + suffix, "wb") as storing_output:
            pickle.dump(judge_df, storing_output)
            storing_output.close()


def case_stacker(folder_path: str, judges_list: list) -> None:
    """
    Stacks all available dataframes for a judge and stores
    the output as a pickle file
    """
    for judge in tqdm.tqdm(judges_list):     
        # finding files of current judge
        judge_folder = re.sub(r"\s+", '-', judge)
        directory = folder_path + "/" + judge_folder        
        judge_file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # reading files 
        judge_files = []
        for file_name in judge_file_names:
            judge_pickle = open(directory + f"/{file_name}", "rb")
            judge_df = pickle.load(judge_pickle)
            judge_pickle.close()
            judge_files.append(judge_df)
        
        # creating stacked file
        if len(judge_files) > 1:
            output = pd.concat(judge_files)
        else:
            output = judge_files[0]
        
        # creating subdirectory to store stacked file
        stacked_files_subfolder = directory + "/stacked"
        if not os.path.exists(stacked_files_subfolder): # only creating if folder not exists
            os.makedirs(stacked_files_subfolder)
        
       # saving dataframe with stacked judge cases 
        with open(stacked_files_subfolder + "/" + judge_folder + "-stacked", "wb") as storing_output:
            pickle.dump(output, storing_output)
            storing_output.close()


def create_cleaned_dfs(path: str, file_name: str, con: object, nlp: object, num_chunks=1000) -> object:
    """
    Creates dataframes w cleaned judge names/data and only w
    cases that don't have multiple judges
    """
    
    # reading files
    pickle_data_df = pd.read_sql(f"SELECT * FROM {file_name}", con)

    # dividing data in batches
    pickle_data_df = np.array_split(pickle_data_df[["Juez:", "text"]], num_chunks)

    # creating subdirectory to store temp chunked files
    temp_files_subfolder = path + "/temp"
    if not os.path.exists(temp_files_subfolder): # only creating if folder not exists
        os.makedirs(temp_files_subfolder)
    
    it_counter = 0 # cleaning data in chunks
    for chunk_df in tqdm.tqdm(pickle_data_df[it_counter:]):
        # keeping single judge files
        chunk_df["multiple_judge"] = chunk_df["Juez:"].apply(lambda judge: detect_mult_judges(judge))
        chunk_df = chunk_df[chunk_df["multiple_judge"] == "Single"]

        # cleaning judge vars 
        chunk_df["Juez:"] = chunk_df["Juez:"].apply(lambda row: re.sub(r"[^a-zA-Z.\d\s]", '', row)) 
        chunk_df["Juez:"] = chunk_df["Juez:"].str.replace(".", "")

        # cleaning and tokenizing text
        chunk_df["sentences_tokenized"] = chunk_df["text"].apply(lambda text: sentence_tokenizer(text, stopwords, nlp))
        chunk_df["cleaned_text"] = chunk_df["text"].apply(lambda text: " ".join(spanish_cleaner(text, stopwords)) 
                                                                    if type(text) is str else np.NaN)
        chunk_df["tokenized_text_pos"] = chunk_df["cleaned_text"].apply(lambda cleaned_text: 
                                                                spanish_pos_tagger(cleaned_text, nlp) if type(cleaned_text) is str else np.NaN)
        chunk_df = chunk_df[["Juez:", "sentences_tokenized", "tokenized_text_pos"]]
        create_pickle(chunk_df, f"chunk_df_{it_counter}", temp_files_subfolder) # storing chunk as pkl for future use
        it_counter += 1


def stacker_temp_chunks(temp_files_subfolder, chunk_prefix="chunk_df", initial_chunk=0, final_chunk=1000):
    """
    Stacks all the temp files w cleaned data from a specific year
    specified current folder
    """

    # stacking files 
    output = read_pickle(temp_files_subfolder, f"{chunk_prefix}_{initial_chunk}")
    for file_num in tqdm.tqdm(range(initial_chunk + 1, final_chunk + 1)):
        # reading current chunk
        chunk_pickle_df = read_pickle(temp_files_subfolder, f"{chunk_prefix}_{file_num}")
        output = pd.concat([output, chunk_pickle_df])
    
    create_pickle(output, f"{chunk_prefix}_{initial_chunk}-{final_chunk}", temp_files_subfolder)


def erase_chunks(temp_files_subfolder, chunk_prefix="chunk_df", num_chunks=1000):
    """
    Erases all created chunks
    """

    for file_num in tqdm.tqdm(range(1, num_chunks)):
        os.remove(temp_files_subfolder + f"/{chunk_prefix}_{file_num}") # erasing temp pickles
    os.remove(temp_files_subfolder + f"/{chunk_prefix}_0") # erasing first temp pickle


def get_judges_list(dataframe, path: str, file_name) -> list:
    """
    Returns a pickle with the list of judges of current df
    and stores it as pkl
    """

    # obtaining judges names
    judges_list = list(dataframe["Juez:"].unique())
    try:
        judges_list.remove("NO DEFINIDO") # removing undefined judges
    except:
        pass

    # storing judges names
    with open(path + "/" + file_name, "wb") as storing_output:
        pickle.dump(judges_list, storing_output)
        storing_output.close()

    return judges_list


if __name__ == "__main__":
    parameters = json.load(open(parameters_json))  

    local_path = parameters["paths"]["unix_paths"]["out_directory"]
    year = parameters["parameters"]["year"]
    judges_subfolder = local_path + "/judges_dfs"

    folder_creator("judges_dfs", local_path)
    
    # connecting to local database
    conn = sqlite3.connect(local_path + "/judges_database")

    # reading stopwords
    stopwords = nltk.corpus.stopwords.words('spanish')
    stopwords = [word for word in stopwords if word not in male_pronouns and word not in female_pronouns]
    
    # reading required objects
    nlp = spacy.load("es_core_news_sm")
    nlp.add_pipe("sentencizer") # adding sentencizer for model
    gender_detector = gender.Detector()

    # divide df, clean data and stack cleaned chunks
    create_cleaned_dfs(local_path, f"merged_cases_{year}", conn, nlp, 1000)
    stacker_temp_chunks(local_path + "/temp", initial_chunk=0, final_chunk=999)
    
    os.rename(local_path + "/temp/chunk_df_0-999", local_path + f"/temp/chunk_df_{year}") # rename stacked chunks
    erase_chunks(local_path + "/temp") # erasing chunks
   
    # obtaining list of judges in cases
    judges_df = read_pickle(local_path + "/temp", f"chunk_df_{year}")
    judges = get_judges_list(judges_df, judges_subfolder, f"judges_{year}")
        
    # create pickle dfs of all cases a processed by a judge per year
    extract_judges_data(judges_df, judges, str(year), judges_subfolder)
