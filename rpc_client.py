# import xmlrpc.client
import xmlrpclib
import sys


image_path = sys.argv[1]

# server_proxy = xmlrpc.client.ServerProxy('http://localhost:8000')
server_proxy = xmlrpclib.ServerProxy('http://localhost:8000')
with open(image_path, 'rb') as fh:
    # image_binary = xmlrpc.client.Binary(fh.read())
    image_binary = xmlrpclib.Binary(fh.read())
    print('Send image {} to server...'.format(image_path))
    r = server_proxy.predict_image(image_binary)
    print(r)
