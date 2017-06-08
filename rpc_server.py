from image_classification import Classifier
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client
import io
import json
import sys

model_type = sys.argv[1]
image_classifier = Classifier(model_type)

fp = open('config.json', 'r')
server_config = json.load(fp)
server_address = server_config['address']
server_port = server_config['port']

server = SimpleXMLRPCServer((server_address, int(server_port)))
server.register_introspection_functions()
fp.close()


def predict_image(img):
    img_stream = io.BytesIO(img.data)
    return image_classifier.predict(img_stream)

server.register_function(predict_image, "predict_image")

print("Server is running successfully...")
server.serve_forever()

