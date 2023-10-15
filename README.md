# Measuring Biases in judges using textual data
The idea of this project was to obtain a measure of gender bias using publicly available data from [judicial cases in Peru](https://cej.pj.gob.pe/cej/forms/busquedaform.html) and word embeddings.

To achieve this goal, the pipeline of this repo did the following

1. Create a parsed Corpus for each judge found on the sample for all the available years
2. Create word embeddings for each judge using this data and Word2Vec
3. Use the word embeddings to obtain two gender slants
    3.1. Gender Career (Gender vs Career/Household oriented chores)
    3.2. Gender Moral (Gender vs Good/Bad)

The original idea comes from this [Kenya paper](http://users.nber.org/~dlchen/papers/Kenya_Courts_In_Group_Bias.pdf) which uses Glove instead of Word2Vec and doesn't work with textual data in spanish. As the original scripts didn't tackle those challenges and needed to be improved in terms of abstraction/softcoding/documentation, they were recreated from scratch and the final version is the one of this repository.

The statistics behind the methodology to obtain the slants can be found on page 41. 

## Pipeline Overview

- `1_create_judges_files.py` cleans the scraped data and creates a unique dataset per judge for each available year (e.g. judge_1-2017.pkl, judge_1-2018.pkl, ...)
- `2_judges_files_stacker.py` stacks these datasets at the judge level (e.g. judge_1-2017.pkl and judge_1-2018.pkl is stacked to create judge_1.pkl)
- `3_train_embeddings_w2v.py` trains w2vec models per judge using the stacked datasets created by `2_judges_files_stacker.py` and the `word_dimension_tokens` from `parameters.json` (explained on next section)

The functions to perform each of these tasks are defined on the respective `.py` files. However, there are subroutines used across the three files from the pipeline. These ones are stored in `cleaning_scripts/corpus_cleaner_funcs.py` or `cleaning_scripts/ner_gender_detection.py` depending on their functionality.

Note: The input comes from a private PSQL database where the data from this website was scraped.

## Deployment

- Create an environment to run the scripts
    - If you are using anaconda, install it using `environment.yml`
    - Else, create the environment using `runtime.txt` to find the right Python version and `requirements.txt` for the packages
- Create the file `parameters.json` on the folder where you'll run either the `.py` or `.sh`` scripts
```json
{
"paths": {
        "unix_paths": {"out_directory": "/burg/sscc/projects/data_text_iat/judges_data"}
	},
"parameters": {
        "year": "2018",
        "embeddings_file": "/burg/sscc/projects/data_text_iat/judges_data/SBW-vectors-300-min5.txt"
    },
"male_pronouns": ["él", "él mismo", "suyo", "sí", "consigo", "ese", "ese mismo", "aquel", "aquel mismo", "este", "este mismo", "esto", "aquello", "aquello mismo", "otro", "otro mismo", "alguno", "alguno mismo", "ninguno", "ninguno mismo", "varios", "varios mismos", "pocos", "pocos mismos", "muchos", "muchos mismos", "unos", "unos mismos", "mío", "tuyo", "nuestro", "vuestro", "cuyo", "cuántos", "cuánto", "cuantos", "cuanto", "todo", "tanto", "poco", "demasiado", "algunos", "todos", "tantos", "demasiados", "otros", "nosotros", "vosotros", "ellos", "el", "los", "míos", "tuyos", "nuestros", "vuestros", "suyos", "el que", "el cual", "los que", "los cuales", "cuyos", "mucho", "otro más", "cualquiera", "ambos", "sendos", "uno"],
"female_pronouns": ["nosotras", "vosotras", "ellas", "la", "las", "ella", "mía", "mías", "tuya", "tuyas", "nuestra", "nuestras", "vuestra", "vuestras", "suya", "suyas", "esta", "esa", "aquella", "la que", "la cual", "cuya", "cuanta", "las que", "las cuales", "cuyas", "cuantas", "cuánta", "cuántas", "alguna", "toda", "tanta", "poca", "demasiada", "otra", "mucha", "ninguna", "algunas", "todas", "tantas", "pocas", "demasiadas", "otras", "muchas", "varias", "otra más", "cualquiera", "ambas", "sendas", "una", "ella misma", "sí", "consigo", "esa misma", "aquella misma", "esta misma", "esto", "otra misma", "alguna misma", "ninguna misma", "varias mismas", "pocas mismas", "muchas mismas", "unas", "unas mismas"],
"word_dimension_tokens": {
    "male_names": ["luis", "juan", "carlos","antonio","miguel"],
    "female_names": ["rosa", "maría","pilar","isabel","ana"],
    "male": ["sr", "dr", "", "", ""],
    "female": ["ella", ""],
    "good": ["gran", "solidaria","sana","prudente","trabajadora", "razonabilidad"],
    "bad": ["agresor","morosos","mala","perjudicada","victima"],
    "career": ["pago", "trabajador","obreros", "empleadores", "obrero"],
    "family": ["familia","hijos","hija","padre", "madre"]
},
}
```
- Run the scripts according to the order indicated at the beginning of the file
    - After running `1_create_judges_files.py` and `2_judges_files_stacker.py`, Update the entries `male_pronouns`, `female_pronouns`, `word_dimension_tokens` according to your data
    - An automated version of this classification was attempted in `archive/2.5_obtain_words_per_category.py` using a BERT model trained with spanish data. The results were highly inaccurate so the script was discarded.
