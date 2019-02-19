import wikipedia
import pickle
from content import Alltopics
summary = []
failed = []
for topic in Alltopics():
    try:
        summary.append(wikipedia.summary(tuple((topic,str(topic)))))
    except Exception as e:
        failed.append(tuple((topic,e)))
with open("summary.txt", "wb") as fp:
    pickle.dump(summary , fp)
with open('failed.txt', 'wb') as fp:
    pickle.dump('failed', fp)



