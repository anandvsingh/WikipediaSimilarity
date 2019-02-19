import wikipedia
from content import Alltopics
summary = []
failed = []
for topic in Alltopics():
    try:
        summary.append(wikipedia.summary(tuple((topic,str(topic)))))
    except Exception as e:
        failed.append(tuple((topic,e)))

print (summary[:5])


