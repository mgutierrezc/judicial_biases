import pandas as pd
import tqdm, spacy, nltk, pickle, re, logging
from sklearn.utils import resample

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO)
                  
                    
def spanish_cleaner(txt_file: str) -> list:
    """
    Cleans the inputted text following spanish grammatical rules
    """

    text = txt_file
    text = re.sub(r"(&[a-zA-Z]*;)", " ", text)  # the txt files had some unwanted text like &rsquo; this line removes such text
    text = text.lower()
    
    # remove punctuation and numbers from the string
    punctuations = '''!()[]{};:'"\,<>./¿?@#$%^&*_–~=+¨`“”’|0123456789'''  # all but hyphens
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, "")
    
    # replacing encoding characters
    enc_characters = [" st ", " nd ", " rd ", " th ", "srl", "lpfvf", "pctc", "jmxcff", "ayrq", "axu", "oadk", "jcxj", "nplt", "eef", "fcfc", "qyoc", "gobpe", "pfg", "vqrx", "csjppj", "xas", "feeback", "hafceqc", "xqj", "hellip", "rsquo", "ldquo", "rdquo", "ndash", "-", "n°", "nº", "º", "°", "dprgdonpdl", "«", "»", "…", "derjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderj", "ii", "iii", "vii", "viii"]
    
    for item in enc_characters:
        text = text.replace(item, " ")
    
    # cleaning for spanish stop words
    stopword_es = nltk.corpus.stopwords.words('spanish') # loading spanish stop words
    custom_substrs = ["http", "hangouts", "meet", "gmailcom"] # html related
    length_custom_stopwords = len(custom_substrs)
    words = text.split() # tokenizing sentence
    cleaned_words = [word for word in words if word not in stopword_es and len(word) > 1] 
    
    sentence_no_custom = [] # omitting words that contain 
    for cleaned_word in cleaned_words:
        counter_stopwords = 0
        for word in custom_substrs: # evaluating if word contains substr
            if word not in cleaned_word: # if passes, +1 for counter
                counter_stopwords += 1
            if counter_stopwords == length_custom_stopwords: # append if passes all custom substrs tests
                sentence_no_custom.append(cleaned_word)

    return sentence_no_custom


def csv_file_reader(file_path: str) -> object:
    """
    Reads a CSV dataframe as panda dataframe
    """

    # reading data
    df = pd.read_csv(file_path)
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    return df


def parse_judge_name(judge: str, path: str) -> list:
    """
    Parses the raw name of a judge
    """

    # reading the file
    str_judge = judge.replace(" ", "-")
    judge_path = path + "/" + str_judge + "/stacked"
    judge_file = str_judge + "-stacked"

    return str_judge, judge_path, judge_file


def spanish_sentence_cleaner(df, column_data: str) -> list:
    """
    Reads data from several documents and creates a cleaned list 
    with all the sentences per document

    Cleaning involves the removal of punctuation, stop words and 
    encoding characters
    """

    logging.in1("--Sencence cleaner--")
    text_data = list(df[column_data]) # working with text data column

    nlp = spacy.load('es_core_news_sm') # loading spanish model for data cleaning
    nlp.add_pipe("sentencizer")

    sentences_list_doc_wise = [] # cleaning sentences in data
    for i in tqdm.trange(len(text_data)):
        clean_sentences_list = []
        temp = str(text_data[i]).lower()
        sentences = [sent.text for sent in nlp(temp).sents] # obtaining sentences for text    
        for sentence in sentences: # erasing punctuation and stopwords
            cleaned_sentence = spanish_cleaner(sentence)
            if cleaned_sentence != []: # omitting empty lists
                clean_sentences_list.append(cleaned_sentence)

        sentences_list_doc_wise.append(clean_sentences_list) # storing cleaned sentences

    logging.info("Sentence cleaning completed")
    return sentences_list_doc_wise


def top_words(cleaned_sentences: list, num_common_words=50000) -> list:
    """
    Obtaining a list with the N most common words in
    current text
    """

    logging.info("--Top words in text--")

    word_list = [] # list with all words
    for sublist in tqdm.tqdm(cleaned_sentences):
        for sentence in sublist:
            for word in sentence:
                word_list.append(word)
    freq_list = nltk.FreqDist(word_list).most_common(num_common_words) #change this to 50,000 on full corpus
    
    logging.info("Top words obtained")
    return [ele[0] for ele in freq_list]


def creating_train_sample(cleaned_sentences: list, top_n_words: list) -> list:
    """
    Creates a bootstraping sample with sentences that contain at least
    one word from the most frequent ones
    """

    logging.info("--Creating training sample--")
    bootstrap_docs = resample(cleaned_sentences, replace=True, n_samples=len(cleaned_sentences))
    output = []
    for doc in tqdm.tqdm(bootstrap_docs):
        top_sentences = []
        for sentence in doc:
            for word in top_n_words:
                if word in sentence:
                    top_sentences.append(sentence)
                    break
        output.append(top_sentences)
    logging.info("Train sample created")
    return output


if __name__ == "__main__":
    #### testing corpus creation functions

    path = "D:\Accesos directos\Trabajo\World Bank\WB Repos\peru_glove\data_raw\DF_DOWNLOADS_SAMPLE_500.csv"
    output_path = "../aux_data"
    column_name = "text"

    cleaned_sentences = spanish_sentence_cleaner(path, column_name)
    top_50_words = top_words(cleaned_sentences)
    train_sample = creating_train_sample(cleaned_sentences, top_50_words)

    storing_output = open(output_path+"/train_sample", "wb")
    pickle.dump(train_sample, storing_output)
    storing_output.close()
    
    test_pickle = open(output_path+"/train_sample", "rb")
    train_sample_p = pickle.load(test_pickle)
    test_pickle.close()

    print(train_sample_p)
