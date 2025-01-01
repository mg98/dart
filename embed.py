from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

en = model.encode('titanic movie english')
es = model.encode('titanic movie spanish')

print(util.cos_sim(en, model.encode('titanic movie spanish')))