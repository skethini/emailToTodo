from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import spacy
import re
import sentenceData


def split_compound_sentence(sentence, nlp):
    """Splits a sentence into independent phrases if it's a compound sentence"""

    # Preprocesses text to ensure that sentences split by ; or - can be counted as compound sentences
    sentence = sentence.text
    sentence = sentence.replace("–", " – ")
    sentence = sentence.replace(";", " ;")
    sentence = nlp(sentence)

    groups = []
    startPos = 0

    # Adds each independent phrase to the groups list
    for endPos in range(len(sentence)):
        token = sentence[endPos]
        if (token.dep_ == "cc" and token.head.pos_ == "VERB") or (token.text == ";" or token.text == "–"):
            groups.append(sentence[startPos : endPos])
            startPos = endPos + 1

    # Adds the last independent phrase in the sentence to the groups list
    groups.append(sentence[startPos : len(sentence)])

    return groups


def rephrase_conditional(txt):
    """Rephrases a conditional sentence so that the main clause comes before the conditional clause"""

    #A pattern of an if clause followed by an independent clause
    pattern = r'(if\s.*?),\s*(.*?)$'

    matched = re.match(pattern, txt, re.IGNORECASE)

    #Switches the order of the sentence to put the main clause first
    if matched:
        if_clause = matched.group(1)
        if_clause[0].lower()
        main_clause = matched.group(2)
        main_clause = main_clause.strip(".")
        return f"{main_clause} {if_clause}"
    else:
      return txt



def pronoun_switch(txt):
    """Switches the pronouns of the text between first and second person"""

    pronoun_map = {
        "I": "you",
        "me": "you",
        "my": "your",
        "you're": "I'm",
        "mine": "yours",
        "you": "I",
        "your": "my",
        "yours": "mine",
        "yourselves": "myself",
        "myself": "yourself"
    }

    words = txt.split()
    
    # if any words in the sentence are in the pronoun map, switch the pronoun
    txt = [pronoun_map.get(word, word) for word in words]

    txt = " ".join(txt)
    return txt


def processText(txt):
    """Processes the text using two methods and by removing the ending punctuation"""

    txt = pronoun_switch(txt)
    txt = rephrase_conditional(txt)

    # removes ending punctuation
    if txt.endswith("?") or txt.endswith("!") or txt.endswith(".") or txt.endswith(","):
        txt = txt[:-1]

    return txt


def convertRequestToTodo(txt, nlp):
    """Converts a sentence in the form of a request into the form of a to-do or action item"""

    # processes text
    txt = processText(txt)
    txt = nlp(txt)

    verb = None
    verb_index = None

    # if this is a "let" request, returns the sentence as is
    if txt[0].text.lower() == "let":
        return txt

    # finds the last verb in the sentence
    for i in range(len(txt)):
        word = txt[i]
        if word.pos_ == "VERB":
            verb = txt[i]
            verb_index = i

    # returns the statement starting with the last verb, which turns it from a request to an action item
    if verb:
        statement = f"{verb.lemma_.capitalize()} {txt[verb_index + 1:]}"
        return statement
    else:
        return None



def train_model():
    """Trains a logistic regression model to classify text as a request, nonrequest, or question"""
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        sentenceData.sentences, sentenceData.labels, test_size=0.2, random_state=42
    )

    tfidf_vectorizer = TfidfVectorizer()

    # Convert text to TF-IDF features
    train_features = tfidf_vectorizer.fit_transform(train_texts)
    test_features = tfidf_vectorizer.transform(test_texts)

    # Train a logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(train_features, train_labels)

    # Make predictions on the test set
    predictions = model.predict(test_features)

    # print(classification_report(test_labels, predictions))
    return [model, tfidf_vectorizer]



def createTodoRequestLists(text):
    """Uses the LR model to classify each sentence as a request/question and add it to the respective list"""

    nlp = spacy.load("en_core_web_sm")
    
    text = nlp(text)
    requests = []
    questions = []

    # gets the model and vectorizer from the train_model() function
    modelVectorizer = train_model()
    model = modelVectorizer[0]
    tfidf_vectorizer = modelVectorizer[1]
    
    # classifies each sentence in text and adds to the respective list of requests or questions
    for sent in text.sents:

        # splits any compound sentences and uses a for loop to classify each independent phrase
        independentSentences = split_compound_sentence(sent, nlp)

        for sentence in independentSentences:
            preprocessed_sentence = tfidf_vectorizer.transform([sentence.text])
            prediction = model.predict(preprocessed_sentence)

            # adds the sentence to the request or question list 
            if prediction == "request":
                todo = convertRequestToTodo(sentence.text, nlp)
                requests.append(todo)
            elif prediction == "question":
                questions.append(sentence.text)
    
    return [requests, questions]
