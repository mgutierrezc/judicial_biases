import pandas as pd
import numpy as np
import tqdm, json, sys
from tqdm import tqdm
from cleaning_tools.corpus_cleaner_funcs import *
from cleaning_tools.ner_sentiment_detection import *
import pingouin as pg
from create_judges_files import *
from sklearn.utils import resample
from gensim.models import Word2Vec
from statistics import median

maxInt = sys.maxsize

# loading parameters
parameters_json = "./parameters.json"
parameters = json.load(open(parameters_json))
word_dimension_tokens = parameters["word_dimension_tokens"]

# reading dfs from judges
local_path = parameters["paths"]["unix_paths"]["out_directory"]
judges_subfolder = local_path + "/judges_dfs"
judges_list = read_pickle(local_path, "judges_list")

# functions

def w2v_training(judge: str, path: str, num_bootstrap=25, vec_size=100) -> list:
    """
    Bootstraps the data from a judge and trains a w2v model for each bootstraped sample
    """
    
    str_judge, judge_path, judge_file = parse_judge_name(judge, path) # reading the file

    # read data
    judge_data = read_pickle(judge_path, judge_file)
    judge_data = judge_data[~judge_data["sentences_tokenized"].isnull()]
    judge_data["flag_no_data"] = judge_data["sentences_tokenized"].apply(lambda tokens: len(tokens) == 0)
    judge_data = judge_data[judge_data["flag_no_data"] == False]

    # keeping sentences only
    all_sentences_raw = list(judge_data["sentences_tokenized"])
    sentences = []
    for sentences_raw in all_sentences_raw:
        for sentence in sentences_raw:
            if sentence != []: # removing empty lists
                sentences.append(sentence)

    # training w sentences and storing models
    for iteration in tqdm.tqdm(range(num_bootstrap)):
        bootstrap_sentences = resample(sentences, replace=True, n_samples=len(sentences)) # boostrapping sample
        model = Word2Vec(bootstrap_sentences, vector_size=vec_size) # training model
        
        # storing models
        folder_creator("models", path + "/" + str_judge)
        create_pickle(model, f"model_{iteration}_{str_judge}", path + "/" + str_judge + "/" + "models")


def vector_single_category(category_name: str, model: object, vec_size=100) -> object:
    """
    Obtains a word vector for a single category
    """
    
    dimension = np.zeros(vec_size) # placeholder for catg dimension
    
    for word in word_dimension_tokens[category_name]: # summing vector for each word in category
        dimension += model[word]

    dimension = dimension/(len(word_dimension_tokens[category_name])) # averaging sum of vectors
    
    return dimension


def obtain_slants(judge: str, path: str, num_bootstrap=25, vec_size=100) -> list:
    """
    Obtains the vectors for male-female/career-family required to build the gender slants
    """

    str_judge, _, __ = parse_judge_name(judge, path) # reading the file

    slants = { # placeholder for slant measures
        "gender_career": [],
        "gender_moral": []
    }

    # failure counts (for logging)
    fail_count_models = 0
    fail_count_registry = {}
    fail_count_registry["model"] = []
    fail_count_registry["fail_count_male"] = []
    fail_count_registry["fail_count_female"] = []
    
    for iteration in tqdm.tqdm(range(num_bootstrap)): # obtain slant for each bootstrapped model
        # reading the models 
        model_path = path + "/" + str_judge + "/" + "models"
        model = read_pickle(model_path, f"model_{iteration}_{str_judge}")

        fail_count_male = 0
        fail_count_female = 0
        try:
            male_dimension = np.zeros(vec_size) # creating a 'male' category
            for word in word_dimension_tokens["male_names"] + word_dimension_tokens["male"]:
                male_dimension += model.wv[word]
            male_dimension = male_dimension/(len(word_dimension_tokens["male_names"]) + word_dimension_tokens["male"])
        except Exception as E:
            fail_count_male += 1

        try:
            female_dimension = np.zeros(vec_size) # creating a 'female' category
            for word in word_dimension_tokens["female_names"] + word_dimension_tokens["female"]:
                female_dimension += model.wv[word]
            female_dimension = female_dimension/(len(word_dimension_tokens["female_names"]) + word_dimension_tokens["female"])
        except Exception as E:
            fail_count_female += 1

        if fail_count_male != len(word_dimension_tokens["male_names"]) and fail_count_female != len(word_dimension_tokens["female_names"]):
            gender_dimension = male_dimension - female_dimension # gender slant

            # career-family slant
            career_dimension = vector_single_category("career", model.wv)
            family_dimension = vector_single_category("family", model.wv)
            career_fam_dimension = career_dimension - family_dimension

            # good-bad slant
            good_dimension = vector_single_category("good", model.wv)
            bad_dimension = vector_single_category("bad", model.wv)
            moral_dimension = good_dimension - bad_dimension

            slants["gender_career"].append(cosine_similarity(gender_dimension, career_fam_dimension))
            slants["gender_moral"].append(cosine_similarity(gender_dimension, moral_dimension))
        else:
            fail_count_models += 1
        
        # updating register of failure count for current model (for logging)
        fail_count_registry["model"].append(model)
        fail_count_registry["fail_count_male"].append(fail_count_male)
        fail_count_registry["fail_count_female"].append(fail_count_female)
        
    # obtain median of slants and failure count
    fail_count_registry = pd.DataFrame.from_dict(fail_count_registry) # converting failure registry into df
    return median(slants["gender_career"]), median(slants["gender_moral"]), fail_count_registry


def gender_slant_validator(judge: str, path: str,word_dimension_tokens: object,  num_bootstrap=25, vec_size=100) -> list:
    """
    Obtains the vectors for male-female/career-family required to build the gender slants
    """

    str_judge, _, __ = parse_judge_name(judge, path) # reading the file
    df_validation = pd.DataFrame()

    for iteration in tqdm.tqdm(range(num_bootstrap)): # obtain slant for each bootstrapped model        
        # reading the models 
        model_path = path + "/" + str_judge + "/" + "models"
        model = read_pickle(model_path, f"model_{iteration}_{str_judge}")

        # male-female slant
        male_dimension = np.zeros(vec_size) # male calculations
        for word in word_dimension_tokens["male_names"] + word_dimension_tokens["male"]:
            male_dimension += model.wv[word]
        male_dimension = male_dimension/(len(word_dimension_tokens["male_names"]) + word_dimension_tokens["male"])
        
        female_dimension = np.zeros(vec_size) # female calculations
        for word in word_dimension_tokens["female_names"] + word_dimension_tokens["female"]:
            female_dimension += model.wv[word]
        female_dimension = female_dimension/(len(word_dimension_tokens["female_names"]) + word_dimension_tokens["female"])

        gender_dimension = male_dimension - female_dimension # gender slant
        
        data_validation = {} # initializing dict w validation variables
        data_validation["name"] = []
        data_validation["name_is_male"] = []
        data_validation["cosine_sim"] = []
        data_validation["model_name"] = []

        for word in word_dimension_tokens["male_names"] + word_dimension_tokens["female_names"]: # populating values for the validation

            if word in word_dimension_tokens["male_names"]: # name is male var
                data_validation["name_is_male"].append(1)
            else:
                data_validation["name_is_male"].append(0)

            data_validation["cosine_sim"].append(cosine_similarity(model.wv[word], gender_dimension)) # cosine similarity
            
            data_validation["name"].append(word) # current name
            data_validation["model_name"].append(f"model_{iteration}_{str_judge}") # current model name
            
        current_data_df = pd.DataFrame.from_dict(current_data_df) # current dict to dataframe
        df_validation = pd.concat([df_validation, current_data_df], ignore_index=True) # storing values of current bootstrapped model

    # regressing dummy over median 
    linear_model = pg.linear_regression(df_validation["cosine_sim"], df_validation["name_is_male"])

    print(linear_model) # erase after debugging
    return linear_model, df_validation


if __name__ == "__main__":

    output = pd.DataFrame()
    for judge in tqdm.tqdm(judges_list[:2]): # creating models for all judges all judges
        judge_dict = {} # placeholder for current judge output
        
        # model training and slant calculations
        w2v_training(judge, judges_subfolder)
        gender_career_slant, gender_moral_slant, failure_registry = obtain_slants(judge, judges_subfolder)
        
        judge_dict["name"] = judge
        judge_dict["gender_career_slant"] = gender_career_slant
        judge_dict["gender_moral_slant"] = gender_moral_slant
        
        # exporting outputs
        judge_df = pd.DataFrame.from_dict(judge_dict)
        output = pd.concat([output, judge_df], ignore_index=True)
        output.to_excel(local_path + "/slant_judge_measures.xlsx", index=False)
        failure_registry.to_excel(local_path + "/failure_registry.xlsx", index=False)
