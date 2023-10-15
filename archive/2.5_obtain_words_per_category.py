import pandas as pd
import numpy as np
import sys, os, re
import csv, tqdm, pickle, json
from collections import Counter
from tqdm import tqdm
from cleaning_tools.corpus_cleaner_funcs import *
from cleaning_tools.ner_gender_detection import *
from create_judges_files import *

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


def create_json(dict_name: dict, file_name: str) -> None:
    """
    Creates a json file from current dictionary
    """
    with open(file_name, "w") as file_json:
        json.dump(dict_name, file_json)


def obtain_words_judge(judge: str, path: str) -> list:
    """
    Returns the male names, female names, pronouns and nouns found in
    the data of a judge
    """
    # reading the file
    str_judge = judge.replace(" ", "-")
    judge_path = path + "/" + str_judge + "/stacked"
    judge_file = str_judge + "-stacked"
    
    # read data
    judge_data = read_pickle(judge_path, judge_file)
    judge_data = judge_data[~judge_data["tokenized_text_pos"].isnull()]
    judge_data["flag_no_data"] = judge_data["tokenized_text_pos"].apply(lambda tokens: len(tokens) == 0)
    judge_data = judge_data[judge_data["flag_no_data"] == False]

    # obtaining people names
    judge_data["names"] = judge_data["tokenized_text_pos"].apply(lambda tokenized_text_pos: element_detector(tokenized_text_pos, "PROPN")
                                                                 if type(tokenized_text_pos) is list else np.NaN)

    judge_data["male names"] = judge_data["names"].apply(lambda names_text: male_classifier(names_text, gender_detector)
                                                         if type(names_text) is list else np.NaN)

    judge_data["female names"] = judge_data["names"].apply(lambda names_text: female_classifier(names_text, gender_detector)
                                                           if type(names_text) is list else np.NaN)

    # obtain most common pronouns
    judge_data["pronouns"] = judge_data["tokenized_text_pos"].apply(lambda tokenized_text_pos: pron_det_detector(tokenized_text_pos)[0] if type(tokenized_text_pos) is list else np.NaN)

    # obtain most common adjectives
    judge_data["adjectives"] = judge_data["tokenized_text_pos"].apply(lambda tokenized_text_pos:
                                                                element_detector(tokenized_text_pos, "ADJ")
                                                                if type(tokenized_text_pos) is list else np.NaN)
    # obtain most common nouns
    judge_data["nouns"] = judge_data["tokenized_text_pos"].apply(lambda tokenized_text_pos: 
                                                            element_detector(tokenized_text_pos, "NOUN")
                                                            if type(tokenized_text_pos) is list else np.NaN)

    # obtaining words for vectors: male/female/pronoun detection
    male_names = [item for item in judge_data["male names"].values.tolist() if str(item) != "nan"] 
    male_names = [item for sublist in male_names for item in sublist]
    female_names = [item for item in judge_data["female names"].values.tolist() if str(item) != "nan"]
    female_names = [item for sublist in female_names for item in sublist]
    pronouns = [item for item in judge_data["pronouns"].values.tolist() if str(item) != "nan"]
    pronouns = [item for sublist in pronouns for item in sublist]
    nouns = [item for item in judge_data["nouns"].values.tolist() if str(item) != "nan"]
    nouns = [item for sublist in nouns for item in sublist]
    adjectives = [item for item in judge_data["adjectives"].values.tolist() if str(item) != "nan"]
    adjectives = [item for sublist in adjectives for item in sublist]

    male_count = dict(Counter(male_names))
    female_count = dict(Counter(female_names))
    pronouns_count = dict(Counter(pronouns))
    nouns_count = dict(Counter(nouns))
    adjectives_count = dict(Counter(adjectives))

    return male_count, female_count, pronouns_count, nouns_count, adjectives_count


if __name__ == "__main__":
    parameters = json.load(open(parameters_json))  
    
    local_path = parameters["paths"]["unix_paths"]["out_directory"]
    judges_subfolder = local_path + "/judges_dfs"

    # reading required objects
    stopwords = nltk.corpus.stopwords.words('spanish')
    nlp = spacy.load("es_core_news_sm")
    gender_detector = gender.Detector()
    judges_list = read_pickle(local_path, "judges_list")

    # loading models
    w2vec_file = parameters["parameters"]["embeddings_file"]
    model = create_w2v_dict(w2vec_file)

    # creating placeholders for count variables
    male_names_total = []
    female_names_total = []
    pronouns_total = []
    nouns_total = []
    adjectives_total = []

    male_names_total_count = []
    female_names_total_count = []
    pronouns_total_count = []
    nouns_total_count = []
    adjectives_total_count = []

    # obtaining words required for word vectors
    success_count = 0
    for judge in tqdm.tqdm(judges_list):
        try:
            male_count, female_count, pronouns_count, nouns_count, adjectives_count = obtain_words_judge(judge, judges_subfolder) 
            
            # storing count results for current set of cases
            male_names_total_count.append(male_count)
            female_names_total_count.append(female_count)
            pronouns_total_count.append(pronouns_count)
            nouns_total_count.append(nouns_count)
            adjectives_total_count.append(adjectives_count)

            male_names_total += list(male_count.keys())
            female_names_total += list(female_count.keys())
            pronouns_total += list(pronouns_count.keys())
            nouns_total += list(nouns_count.keys())
            adjectives_total += list(adjectives_count.keys())

            success_count += 1 # increasing success counter
        
        except Exception as e:
            #pass
            print(e)

    print("success rate: ", (success_count/len(judges_list))*100)
        
    # storing unique names and pronouns found on set of cases
    unique_male_names = list(set(male_names_total))
    unique_female_names = list(set(female_names_total))
    unique_pronouns = list(set(pronouns_total))
    unique_nouns = list(set(nouns_total))
    unique_adjectives = list(set(adjectives_total))

    # obtaining count of male/female names
    final_count_male = obtain_final_count(unique_male_names, male_names_total_count)
    pd.DataFrame({"word": list(final_count_male.keys()), "count": list(final_count_male.values())}).to_excel(local_path + "/final_count_male.xlsx")
    final_count_female = obtain_final_count(unique_female_names, female_names_total_count)
    pd.DataFrame({"word": list(final_count_female.keys()), "count": list(final_count_female.values())}).to_excel(local_path + "/final_count_female.xlsx")
    
    # pronouns count
    final_count_pronouns = obtain_final_count(unique_pronouns, pronouns_total_count)
    prons_df = pd.DataFrame({"word": list(final_count_pronouns.keys()), "count": list(final_count_pronouns.values())})
    prons_df = prons_df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') 
                                if isinstance(x, str) else x)
    prons_df.to_excel(local_path + "/final_count_pronouns.xlsx")

    # nouns count
    final_count_nouns = obtain_final_count(unique_nouns, nouns_total_count)
    nouns_df = pd.DataFrame({"word": list(final_count_nouns.keys()), "count": list(final_count_nouns.values())})
    nouns_df = nouns_df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8')
                                            if isinstance(x, str) else x)
    nouns_df.to_excel(local_path + "/final_count_nouns.xlsx")

    # adjectives count
    final_count_adjectives = obtain_final_count(unique_adjectives, adjectives_total_count)
    adjectives_df = pd.DataFrame({"word": list(final_count_adjectives.keys()), "count": list(final_count_adjectives.values())})
    adjectives_df = adjectives_df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    adjectives_df.to_excel(local_path + "/final_count_adjectives.xlsx")


    # classifying words
    nouns_distances = categories_classifier(unique_nouns, ["carrera", "familia"], model) # get distance to "bueno" and "malo"
    adjectives_distances = categories_classifier(unique_nouns, ["bien", "mal"], model) # get distance to "bueno" and "malo"
    
    # storing 'good' distances
    adjectives_good_distances = adjectives_distances["bien"]
    adjectives_good_df = pd.DataFrame({"word": list(adjectives_good_distances.keys()), "count": list(adjectives_good_distances.values())})
    adjectives_good_df = adjectives_good_df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    adjectives_good_df.to_excel(local_path + "/adjectives_good_distances.xlsx")
    
    # storing 'bad' distances
    adjectives_bad_distances = adjectives_distances["mal"]
    adjectives_bad_df = pd.DataFrame({"word": list(adjectives_bad_distances.keys()), "count": list(adjectives_bad_distances.values())})
    adjectives_bad_df = adjectives_bad_df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    adjectives_bad_df.to_excel(local_path + "/adjectives_bad_distances.xlsx")

    # storing 'career' distances
    nouns_career_distances = nouns_distances["carrera"]
    nouns_career_df = pd.DataFrame({"word": list(nouns_career_distances.keys()), "count": list(nouns_career_distances.values())})
    nouns_career_df = nouns_career_df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    nouns_career_df.to_excel(local_path + "/nouns_career_distances.xlsx")

    # storing 'family' distances
    nouns_family_distances = nouns_distances["familia"]
    nouns_family_df = pd.DataFrame({"word": list(nouns_family_distances.keys()), "count": list(nouns_family_distances.values())})
    nouns_family_df = nouns_family_df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    nouns_family_df.to_excel(local_path + "/nouns_family_distances.xlsx")

    create_pickle(nouns_distances, "nouns_distances", local_path)
