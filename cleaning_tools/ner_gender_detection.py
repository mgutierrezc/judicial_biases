import nltk, spacy, scipy, json
import regex as re
import numpy as np
import gender_guesser.detector as gender
from collections import Counter

# parameters
parameters_json = "./parameters.json"

# functions
def distance_words(word: str, comparison_word: str, model: object) -> float:
    """
    obtains difference between the word and the comparison word vector 
    """

    comparison_vector = embedding_for_word(comparison_word, model)
    word_vector = embedding_for_word(word, model)

    try:
        output = np.linalg.norm(word_vector - comparison_vector)
    except:
        output = np.NaN

    return output


def cosine_similarity(vector_1, vector_2, verbose=False) -> float:
    """
    Obtains cosine similarity/proximity between two vectors

    input: vectors 1 and 2
    output: cosine similarity
    """

    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    if verbose:
        print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
    return (1-cosine)


def create_w2v_dict(embeddings_file_path: str) -> dict:
    """
    returns w2v dict based on embeddings txt file
    """

    output = {}
    w2v_file = open(embeddings_file_path)

    for line in w2v_file:
        records = line.split() # splits line in words/numbers
        word = records[0]
        vector_w2v = np.array(records[1:], dtype='float32') # taking the numeric elements to create vector
        output[word] = vector_w2v # storing the vector on dict

    return output


def embedding_for_word(word: str, model: dict) -> list:
    """
    obtains word embedding for a given word
    """

    try:
        output = model[word]
    except:
        output = np.NaN

    return output


def spanish_pos_tagger(text: str, nlp: object) -> list:
    """
    POS tagger for spanish text
    """
    
    doc = nlp(text) # tokenizing using nlp model
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags


def element_detector(text_tokenized_pos: list, element: str) -> list:
    """
    Detects and keeps all elements in text 
    """

    output = []
    for item_w_pos in text_tokenized_pos: # iterating through all tokenized items
        token = str(item_w_pos[0]).strip()
        token_pos = str(item_w_pos[1]).strip()
        if token_pos == element: # keeping only names
            output.append(token)
    return output


def pron_det_detector(text_tokenized_pos: list) -> list:
    """
    Detects and keeps all pronouns and determinants in text 
    """

    pronouns = []
    determinants = []
    for item_w_pos in text_tokenized_pos: # iterating through all tokenized items
        token = str(item_w_pos[0]).strip()
        token_pos = str(item_w_pos[1]).strip()
        if token_pos == "PRON": # keeping only pronouns
            pronouns.append(token)
        elif token_pos == "DET": # keeping only determinants
            determinants.append(token)
    return pronouns, determinants


def categories_classifier(tokens: list, categories: list, model: object) -> tuple:
    """
    Classifies the inputted tokens to each category
    """

    # initializing dict of dicts by evaluated category
    distance_categories_token = {}
    for category in categories:
        distance_categories_token[category] = {}

    # obtaining distance to each category
    for token in tokens:
        current_distances = {} 
        for category in categories: # distance to each word
            distance = distance_words(token, category, model)
            if distance != np.NaN:
                current_distances[category] = distance

        if current_distances != {}:
            closer_categ = min(current_distances, key = current_distances.get) # getting closer category/min distance
            distance_categories_token[closer_categ][token] = current_distances[closer_categ]

    for category in categories: # sorting current dicts by proximity
        distance_categories_token[category] = {key: value for key, value in sorted(distance_categories_token[category].items(), key=lambda key_val_tuple: key_val_tuple[1])}

    return distance_categories_token


def male_classifier(tokenized_names: list, gender_guesser: object) -> tuple:
    """
    Classifies the inputted names according to their gender
    """

    male_names = []
    for name in tokenized_names: # evaluating each name
        gender = gender_guesser.get_gender(name.capitalize()) # obtaining gender
        if gender == "male" or gender == "mostly_male": # obtaining male names
            male_names.append(name)
    return male_names


def female_classifier(tokenized_names: list, gender_guesser: object) -> tuple:
    """
    Classifies the inputted names according to their gender
    """

    female_names = []
    for name in tokenized_names: # evaluating each name
        gender = gender_guesser.get_gender(name.capitalize()) # obtaining gender        
        if gender == "female" or gender == "mostly_female": # obtaining female names
            female_names.append(name)
    return female_names


def sentence_tokenizer(txt_file: str, stopword_es: object, nlp: object) -> list:
    """
    Cleans and tokenizes all the sentences in the current text
    """

    output = []
    try:
        splitted_sentences = [sent.text for sent in nlp(txt_file).sents] # obtaining sentences for text    
        for sentence in splitted_sentences: # cleaning each sentence
            if sentence != []:
                cleaned_sentence = spanish_cleaner(sentence, stopword_es)
                output.append(cleaned_sentence)
        
        return output
    except:
        return output


def spanish_cleaner(txt_file: str, stopword_es: object) -> list:
    """
    Removes unnecessary characters from text
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
    enc_characters = [" st ", " nd ", " rd ", " th ", "srl", "lpfvf", "pctc", "jmxcff", "ayrq", "axu", "oadk", "jcxj", "nplt", "eef", "fcfc", "qyoc", "gobpe", "pfg", "vqrx", "csjppj", "xas", "feeback", "hafceqc", "xqj", "hellip", "rsquo", "ldquo", "rdquo", "ndash", "-", "n°", "nº", "º", "°", "dprgdonpdl", "«", "»", "…", "derjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderjudicialpoderj", "ii", "iii", "vii", "viii", "jr"]
    
    for item in enc_characters:
        text = text.replace(item, " ")
    
    # cleaning for spanish stop words
    custom_substrs = ["http", "hangouts", "meet", "gmailcom", "created", "with", "aspose"] # html and aspose related
    custom_gender_words = ["él", "ella", "la", "ese", "esa", "esos", "esas", "este", "esta", "aquel", "aquella", "aquellos", "aquellas", "lo", "la", "los", "las", "aquel", "aquella", "mío", "mía", "míos", "mías", "suyo", "suya", "suyos", "suyas"] # list with pronouns associated to a specific gender
    length_custom_stopwords = len(custom_substrs)
    words = text.split() # tokenizing sentence
    cleaned_words = [word for word in words if (word not in stopword_es and len(word) > 1) or word in custom_gender_words]
     
    sentence_no_custom = [] # omitting words that contain 
    for cleaned_word in cleaned_words:
        counter_stopwords = 0
        for word in custom_substrs: # evaluating if word contains substr
            if word not in cleaned_word: # if passes, +1 for counter
                counter_stopwords += 1
            if counter_stopwords == length_custom_stopwords: # append if passes all custom substrs tests
                sentence_no_custom.append(cleaned_word)

    return sentence_no_custom


def obtain_final_count(unique_names: list, total_counts: list) -> dict:
    """
    Obtains the final count of words for the evaluated dimensions
    """

    output = {} 
    for name in unique_names:
        if name not in output.keys(): # initialize the entry on the dictionary
            output[name] = 0
        for dict_count in total_counts: # add previous counts
            try:
                output[name] += dict_count[name]
            except:
                pass
    return  dict(sorted(output.items(), key=lambda item: item[1], reverse=True))


if __name__ == "__main__":
    # creating testing data
    text = """
    En contraste con la armonía entre la carrera y la familia, también debemos enfrentar la presencia insidiosa del mal en nuestras vidas. La maldad se manifiesta a través de la envidia, el engaño, la codicia y la crueldad. Hay individuos despiadados como Sebastián, un manipulador maestro del engaño, y Valeria, una mujer astuta que no duda en pisotear a los demás para lograr sus objetivos egoístas. Estos seres perversos están dispuestos a cualquier cosa para obtener poder y riqueza, incluso si eso significa sembrar dolor y destrucción en su camino.

    El mal se arraiga en las sombras más oscuras de nuestra sociedad, manifestándose en la corrupción, la violencia y la opresión. Existen figuras siniestras como Ramiro, un hombre sin escrúpulos que aprovecha su posición de poder para explotar y oprimir a los más vulnerables, y Amanda, una mujer fría y despiadada que se deleita en la miseria ajena. Estos individuos representan la oscuridad que puede envenenar la armonía familiar y el desarrollo profesional, alimentando un ciclo destructivo que socava los valores y la felicidad.

    En medio de esta dualidad entre el bien y el mal, cada uno de nosotros se enfrenta a decisiones trascendentales. Debemos mantenernos alerta y rechazar los impulsos maliciosos que puedan amenazar nuestra integridad y la de aquellos que amamos. Al tomar conciencia de las consecuencias de nuestras acciones y cultivar la empatía y la compasión, podemos contrarrestar la influencia del mal y trabajar en pro de un mundo mejor, donde la bondad prevalezca sobre la maldad."""
    
    # loading objects required for processing
    stopwords = nltk.corpus.stopwords.words('spanish')
    gender_detector = gender.Detector()
    nlp = spacy.load("es_core_news_sm")
    parameters = json.load(open(parameters_json))  
    
    # processing text
    cleaned_text = " ".join(spanish_cleaner(text, stopwords)) # cleaning for stopwords
    tokenized_text_pos = spanish_pos_tagger(cleaned_text, nlp) # tokenizing and pos tagging
    
    # obtaining words for vectors - step 1: name detection
    names_text = element_detector(tokenized_text_pos, "PROPN") # obtaining names
    male_names = male_classifier(names_text, gender_detector)
    female_names = female_classifier(names_text, gender_detector)
    print("male_names", male_names)
    print("female_names", female_names)

    # obtaining words for vectors - step 2: pronoun/determinant detection
    pronouns, determinants = pron_det_detector(tokenized_text_pos) # obtaining names
    print("pronouns", pronouns)
    print("determinants", determinants)
    
    w2vec_file = parameters["parameters"]["embeddings_file"]
    model = create_w2v_dict(w2vec_file)

    # obtaining words for vectors - step 3: good/bad nound detection
    nouns_text = element_detector(tokenized_text_pos, "NOUN") # obtaining names
    nouns_count = dict(Counter(nouns_text)) # counting nouns
    unique_nouns_text = list(set(nouns_text)) # unique nouns in text

    # classifying nouns
    nouns_distances = categories_classifier(unique_nouns_text, ["bien", "mal", "carrera", "familia"], model) # get distance to "bueno" and "malo"
    nouns_good_distances = nouns_distances["bien"]
    nouns_bad_distances = nouns_distances["mal"]
    nouns_career_distances = nouns_distances["carrera"]
    nouns_family_distances = nouns_distances["familia"]

    # obtaining top factor% by proximity to category
    factor = 1
    good_tokens = list(nouns_good_distances.keys())[:int(len(nouns_good_distances)/factor)] 
    bad_tokens = list(nouns_bad_distances.keys())[:int(len(nouns_bad_distances)/factor)] 
    career_tokens = list(nouns_career_distances.keys())[:int(len(nouns_career_distances)/factor)]
    family_tokens = list(nouns_family_distances.keys())[:int(len(nouns_family_distances)/factor)]

    # obtaining count of top good/bad nouns
    count_good_nouns = {}
    count_bad_nouns = {}
    count_career_nouns = {}
    count_family_nouns = {}

    for token in good_tokens: 
        count_good_nouns[token] = nouns_count[token]

    for token in bad_tokens:
        count_bad_nouns[token] = nouns_count[token]

    for token in career_tokens:
        count_career_nouns[token] = nouns_count[token]
    
    for token in family_tokens:
        count_family_nouns[token] = nouns_count[token]
    
    print("count_good_nouns: ", count_good_nouns)
    print("count_bad_nouns: ", count_bad_nouns)
    print("count_career_nouns: ", count_career_nouns)
    print("count_family_nouns: ", count_family_nouns)
