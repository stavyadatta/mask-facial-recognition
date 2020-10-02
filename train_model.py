from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", default='output/embeddings.pickle', 
                help="embeddings features available")
ap.add_argument("-r", "--recognizer",default='output/recognizer.pickle', help="path to output recognozer")
ap.add_argument("-l", "--le", default='output/le.pickle', help='label encoder')

args = vars(ap.parse_args())

print("Loading face embeddings")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels

print("encoding labels")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
