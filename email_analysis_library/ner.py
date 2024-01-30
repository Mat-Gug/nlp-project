import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_trf', disable=['parser'])

def _entities(document, ent_type):
    organizations = []
    doc = nlp(document)
    for token in doc:
        if str(token.ent_type_) == ent_type:
            organizations.append(str(token))
            
    return organizations

def get_organizations(corpus):
    organizations_dict = {}
    
    for document in tqdm(corpus, desc="Extracting organizations", unit="document"):
        organizations_dict[document] = _entities(document, "ORG")
    
    return organizations_dict