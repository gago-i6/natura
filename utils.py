import pandas

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def similarity(model, _str1: str, _str2: str) -> float:
    v1 = model.transform([_str1])
    v2 = model.transform([_str2])
    return cosine_similarity(v1, v2)

def match(model, df: pandas.DataFrame, _str: str, threshold=0.95) -> int:
    _ids = []
    for i, row in df.iterrows():
        _sim = similarity(model=model, _str1=row['NAME'], _str2=_str)
        _id = row['ID']
        if _sim >= threshold and _id != -1:
            _ids.append(_id)

    return min(_ids) if _ids else -1