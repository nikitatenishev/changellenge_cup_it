import re
from functools import reduce
from typing import List

import langdetect
import numpy as np
import pandas
import torch
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import (cosine_similarity, euclidean_distances,
                                      manhattan_distances)
from tqdm import notebook
from transformers import pipeline


def get_lang(string: str) -> str:
    """
    Detects language of a string
    :x: initial string
    :returns: language of a string
    """
    try:
        return langdetect.detect(string)
    except langdetect.LangDetectException:
        return "unknown"


def get_valid_stucture(
        data: pandas.core.frame.DataFrame,
        is_train: bool = True
        ) -> pandas.core.frame.DataFrame:
    """
    Converts initial data structure to table format
    :data: initial data
    :returns: data in table format
    """
    text, comments = [], []

    for val in notebook.tqdm(data.iterrows()):
        tmp = [val[1]["text"]] * 5
        text += tmp
        for comment in val[1]["comments"]:
            comments.append(comment["text"])

    assert len(text) == (data.shape[0] * 5)
    assert len(comments) == (data.shape[0] * 5)

    df = pandas.DataFrame(columns=["text", "comments"])
    df["text"] = text
    df["comments"] = comments

    if is_train:
        scores = [0, 1, 2, 3, 4] * data.shape[0]
        df["score"] = scores

    return df


def clean_punctuation(string: str) -> str:
    """
    Cleans string from punctuation characters
    :string: initial string
    :returns: cleaned string
    """
    return re.sub(r"[^\w\s]", ' ', string).strip()


def clean_domain(text: str) -> str:
    """
    Cleans string from url and leave only the second level domain
    Example, http://www.typeform.com/try ---> typeform,  Node.js ---> Node
    :text: initial string
    :returns: cleaned string
    """
    if len(re.findall("https", text)) != 0:
        try:
            return re.sub(
                r"https:?\D?\D?.+?\.?[a-z.-]+\.[a-z]{1,4}\/\S+",
                ' ' + re.findall(
                    r"https:?\D?\D?(www)?\.?([a-z.-]+)\.[a-z]{1,4}",
                    text
                )[0][1] + ' ', text
            )
        except IndexError:
            return text
    elif len(re.findall("http", text)) != 0:
        try:
            return re.sub(
                r"http:?\D?\D?.+?\.?[a-z.-]+\.[a-z]{1,4}\/\S+",
                ' ' + re.findall(
                    r"http:?\D?\D?(www)?\.?([a-z.-]+)\.[a-z]{1,4}",
                    text
                )[0][1] + ' ',
                text
            )
        except IndexError:
            return text
    else:
        try:
            return re.sub(
                r"[A-Z]?[a-z.-]+\.[a-z]{1,4}\s",
                ' ' + re.findall(
                    r"([A-Z]?[a-z.-]+)\.[a-z]{1,4}",
                    text
                )[0] + ' ',
                text
            )
        except IndexError:
            return text


def remove_stopwords(
        string: str,
        stopwords: List[str]
) -> str:
    """
    Removes stopwords defined by user in a string
    :x: initial string
    :stopwords: stopwords defined by user
    :returns: list of  words
    """
    tmp = [
        word
        for word in string.split(' ')
        if word not in stopwords
    ]
    return ' '.join(tmp)


def remove_whitespaces(string: str) -> str:
    """
    Cleans string from whitespace characters
    :string: initial string
    :returns: cleaned string
    """
    return re.sub(r"\s+", ' ', string)


def create_embeddings(
        lang_model: str,
        data: pandas.core.frame.DataFrame,
        column: str,
        output_filename: str,
        device: str
) -> None:
    """
    Creates text column embeddings and saves into file with name defined
    by user
    :lang_model: language model
    :data: initial data
    :column: column for which it's necessary to create embeddings
    column for which it's necessary to create embeddings
    :output_filename: name of file with embeddings
    :device: Device ('cuda' / 'cpu') that should be used for computation
    :returns: None
    """
    model = SentenceTransformer(model_name_or_path=lang_model, device=device)
    embeds = model.encode(data[column].values, show_progress_bar=True)
    np.save(f"auxiliary_data/{output_filename}", embeds)


def calulacte_cos_measure(
        df_coms: np.ndarray,
        df_text: np.ndarray,
        output_filename: str
) -> None:
    """
    Calculates cosine measure between text embeddings and comments
    embeddings and save result into "cos_measure" file
    :df_coms: comments embeddings
    :df_text: text embeddings
    :output_filename: name of file with cos measure
    :returns: None
    """
    cos_measure = np.zeros(df_coms.shape[0])
    for i in notebook.tqdm(range(df_text[::5].shape[0])):
        cos_measure[5*i:5*i + 5] = cosine_similarity(
            df_coms[5*i:5*i + 5], df_text[::5][i].reshape(1, -1)
        ).ravel()
    np.save(f"auxiliary_data/{output_filename}", cos_measure)


def calculate_euclidean_measure(
        df_coms: np.ndarray,
        df_text: np.ndarray,
        output_filename: str
) -> None:
    """
    Calculates euclidean measure between text embeddings and comments
    embeddings and save result into "euclidean_measure" file
    :df_coms: comments embeddings
    :df_text: text embeddings
    :output_filename: name of file with euclidean measure
    :returns: None
    """
    euclidean_measure = np.zeros(df_coms.shape[0])
    for i in notebook.tqdm(range(df_text[::5].shape[0])):
        euclidean_measure[5*i:5*i + 5] = euclidean_distances(
            df_coms[5*i:5*i + 5], df_text[::5][i].reshape(1, -1)
        ).ravel()
    np.save(f"auxiliary_data/{output_filename}", euclidean_measure)


def calculate_manhattan_measure(
        df_coms: np.ndarray,
        df_text: np.ndarray,
        output_filename: str
) -> None:
    """
    Calculates manhattan_measure between text embeddings and comments
    embeddings and save result into "manhattan_measure" file
    :df_coms: comments embeddings
    :df_text: text embeddings
    :output_filename: name of file with manhattan measure
    :returns: None
    """
    manhattan_measure = np.zeros(df_coms.shape[0])
    for i in notebook.tqdm(range(df_text[::5].shape[0])):
        manhattan_measure[5*i:5*i + 5] = manhattan_distances(
            df_coms[5*i:5*i + 5], df_text[::5][i].reshape(1, -1)
        ).ravel()
    np.save(f"auxiliary_data/{output_filename}", manhattan_measure)


def create_diff_features(
    data: pandas.core.frame.DataFrame
) -> np.ndarray:
    """
    Calculates difference in words amount between all comments
    :data: initial dataframe
    :returns: array with array with distinguished wordscounts between comments
    """
    res = np.zeros((data.shape[0], 5))
    for base_index in notebook.tqdm(range(5)):
        ans = []
        for i in notebook.tqdm(range(data.shape[0] // 5)):
            tmp = data.loc[5*i:5*i + 4, "comments"].values
            for j in range(5):
                length = np.setdiff1d(
                    np.asarray(tmp[base_index].lower().split(' ')),
                    np.asarray(tmp[j].lower().split(' ')),
                    assume_unique=False
                ).shape[0]
                if length == 0:
                    ans.append(len(tmp[j].lower().split(' ')))
                else:
                    ans.append(length)
        assert len(ans) == data.shape[0]
        res[:, base_index] = ans
    return res


def lemmatize(string: str) -> str:
    """
    Lemmatizes text
    :string: initial text
    :returns: lemmatized string
    """
    lemmatizer = WordNetLemmatizer()
    res = [
        lemmatizer.lemmatize(i, j[0].lower())
        if j[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(i)
        for i, j in pos_tag(word_tokenize(string))
    ]
    return ' '.join(res)


def get_repeating_words(text: str, comm: str) -> int:
    """
    Calculates amount of non-unique words between post text
    and comment text
    :text: post text
    :comm: comment text
    :returns: amount of non-unique words
    """
    candidate1 = np.asarray(text.split(' '))
    candidate2 = np.asarray(comm.split(' '))
    return reduce(np.intersect1d, (candidate1, candidate2))


def get_nunique_words(comm: str) -> int:
    """
    Calculates amount of unique words in comment text
    :comm: initial comment
    :returns: amount of unique words
    """
    res = re.findall(r"[a-zA-Z]+", lemmatize(comm))
    return len(set(res))


def remove_emoji(string: str) -> str:
    """
    Replaces unicode characters by empty char
    :data: initial string
    :returns: cleaned string
    """
    emoj = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+",
        re.UNICODE
    )
    return re.sub(emoj, '', string)


def get_text_len(text: str) -> int:
    """
    Gets text length
    :x: initial text
    :returns: text length
    """
    return text.count(' ') + 1


def replace_xa0(string: str) -> str:
    """
    Replaces \xa0 by empty char
    :string: initial string
    :returns: cleaned string
    """
    return string.replace("\xa0", '')


def create_toxic(
        df: pandas.core.frame.DataFrame,
        column: str,
        output_filename: str,
        device: int or str or torch.device
) -> None:
    """
    Defines toxicity of text and saves into file
    with name defined
    :df: initial dataframe
    :columns: selected columns
    :output_filename: output file name
    :device:device on which this pipeline will be allocated
    :returns: None
    """
    model_toxic = pipeline(model="unitary/toxic-bert", device=device)
    toxic_dict = model_toxic(
        df[column].tolist(),
        padding=True,
        max_length=512,
        batch_size=30,
        truncation=True
    )
    toxic = [i["score"] for i in notebook.tqdm(toxic_dict)]
    np.save(f"auxiliary_data/{output_filename}", toxic)


def check_equality_toxic(
        df: pandas.core.frame.DataFrame
) -> bool:
    """
    Checks if equal post text and comment text by toxicity
    :data: initial dataframe
    :return: toxicity equivalency flag
    """
    if (df['toxic'] > 0.5) and (df['toxic_text'] > 0.5):
        return True
    return False


def feature_hard_word(
        string: str,
        hardwords: List[str]
) -> bool:
    """
    Defines does text contains hard word
    :string: initia; text
    :hardwords: list of hard words
    :returns: availability hard word flag
    """
    for word in string.split(' '):
        if word in hardwords:
            return True
        return False


def resemblance_text(
        text: str,
        comments: str
) -> float:
    """
    Determines the similarity of two texts by the formula:
    (NN_r / NN_t * 0.35) + (VB_r / VB_t * 0.35) +
    + (RB_r / RB_t * 0.15) + (JJ_r/ JJ_t * 0.15)
    _r -> number of repeated part of speech
    _t -> the number of parts of speech in the text with which
    similarity is being sought
    :text: post text
    :comments: comment text
    :returns: similarity of two texts
    """
    text = lemmatize(text)
    # получение списка слов, которые есть в посте и комментарии
    repeating = get_repeating_words(text, comments)
    # получение списка частей речи поста
    parts_of_speech_text = [i[1] for i in pos_tag(word_tokenize(text))]
    # получение списка частей речи слов, которые есть в посте и комментарии
    parts_of_speech_repeating = [
        i[1] for i in pos_tag(word_tokenize(' '.join(repeating)))
    ]

    # количество отдельных частей речи в посте
    nn_count_text = len(re.findall("NN", ' '.join(parts_of_speech_text)))
    vb_count_text = len(re.findall("VB", ' '.join(parts_of_speech_text)))
    rb_count_text = len(re.findall("RB", ' '.join(parts_of_speech_text)))
    jj_count_text = len(re.findall("JJ", ' '.join(parts_of_speech_text)))
    # количество отдельных частей речи слов, которые есть в посте и комментарии
    nn_count_repeating = len(
        re.findall("NN", ' '.join(parts_of_speech_repeating))
    )
    vb_count_repeating = len(
        re.findall("VB", ' '.join(parts_of_speech_repeating))
    )
    rb_count_repeating = len(
        re.findall("RB", ' '.join(parts_of_speech_repeating))
    )
    jj_count_repeating = len(
        re.findall("JJ", ' '.join(parts_of_speech_repeating))
    )
    # избавление от ситуации, когда нет какой-то части речи в посте
    nn_calculation = [
        0 if nn_count_text == 0 else nn_count_repeating / nn_count_text * 0.35
    ]
    vb_calculation = [
        0 if vb_count_text == 0 else vb_count_repeating / vb_count_text * 0.35
    ]
    rb_calculation = [
        0 if rb_count_text == 0 else rb_count_repeating / rb_count_text * 0.15
    ]
    jj_calculation = [
        0 if jj_count_text == 0 else jj_count_repeating / jj_count_text * 0.15
    ]

    # расчет метрики сходства
    metric = (
        nn_calculation[0] +
        vb_calculation[0] +
        rb_calculation[0] +
        jj_calculation[0]
    )

    return metric
