from xmlrpc.server import SimpleXMLRPCServer
import io
import json
import sys

from user_exception import CommandError


def load_config():
    """
    Load server configuration

    # Arguments:
        N/A
    # Return:
        server_address: IP address bound to RPC server
        server_port: port number
    """
    try:
        fp = open('config.json', 'r')
    except OSError:
        print('Fail to open server config file, use default config.')
        server_address = 'localhost'
        server_port = '8000'
    else:
        try:
            server_config = json.load(fp)
        except json.JSONDecodeError:
            print('Fail to parse server config, use default config.')
            server_address = 'localhost'
            server_port = '8000'
        else:
            server_address = server_config['address']
            server_port = server_config['port']
        fp.close()
    return server_address, server_port


def predict_image(img):
    """
    Predict service registered in RPC server

    # Arguments:
        img: binary data of an image, provided by client

    # Return:
        Predict results (str)
    """
    img_stream = io.BytesIO(img.data)
    return image_classifier.predict(img_stream)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        raise CommandError('Too many arguments.')

    if len(sys.argv) == 1:
        print('Use default model: VGG16.')
        model_type = 'vgg16'
    elif len(sys.argv) == 2:
        if sys.argv[1] not in {'vgg16', 'pcm'}:
            raise CommandError('Unsupported model. Use [vgg16 | pcm].')
        model_type = sys.argv[1]

    from image_classification import Classifier
    image_classifier = Classifier(model_type)

    server_address, server_port = load_config()
    print('Setup RPC server on {}:{}.'.format(server_address, server_port))

    server = SimpleXMLRPCServer((server_address, int(server_port)))
    server.register_introspection_functions()

    # register services
    server.register_function(predict_image, "predict_image")

    print("Server is running successfully...")
    server.serve_forever()

