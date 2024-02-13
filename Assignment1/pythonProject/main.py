# -------------------------------------------------------------------------
# AUTHOR: Chanrady Ho
# FILENAME: Similarity
# SPECIFICATION: Calculate cosine similarity between documents and find the highest value of a pair
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 1 and a half hour
# -----------------------------------------------------------*/
import string

# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def doc_term_matrix(document, terms):
    doc = [word.strip(string.punctuation) for word in document.split()]
    vector = {i: 0 for i in terms}
    for i in doc:
        if i in vector:
            vector[i] += 1
    return list(vector.values())


# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sport, like, one, support, olympic, games]
terms = ['soccer', 'favorite', 'sport', 'like', 'one', 'support', 'olympic', 'games']

d1_vector = doc_term_matrix(doc1, terms)
d2_vector = doc_term_matrix(doc2, terms)
d3_vector = doc_term_matrix(doc3, terms)
d4_vector = doc_term_matrix(doc4, terms)

print(d1_vector)
print(d2_vector)
print(d3_vector)
print(d4_vector)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
cosine_similarities = {}
cosine_similarities[12] = cosine_similarity([d1_vector], [d2_vector])[0][0]
cosine_similarities[13] = cosine_similarity([d1_vector], [d3_vector])[0][0]
cosine_similarities[14] = cosine_similarity([d1_vector], [d4_vector])[0][0]
cosine_similarities[23] = cosine_similarity([d2_vector], [d3_vector])[0][0]
cosine_similarities[24] = cosine_similarity([d2_vector], [d4_vector])[0][0]
cosine_similarities[34] = cosine_similarity([d3_vector], [d4_vector])[0][0]
print(cosine_similarities)

# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
print("The most similar documents are: dco1 and doc4 with cosine similarity =", cosine_similarities[14])
