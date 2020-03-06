import functools
import traceback

DATA_DIR = 'DATA_DIR'
INIT_METHOD = 'INIT_METHOD'
LOAD_DATA_LIST = 'LOAD_DATA_LIST'
LOAD_DATA = 'LOAD_DATA'
VISUALIZE_OUTPUT = 'VISUALIZE_OUTPUT'
OUTPUT_TYPE = 'OUTPUT_TYPE'
OPTIMIZER = 'OPTIMIZER'
ADV_OPTIMIZER = 'ADV_OPTIMIZER'

TRAINING_BATCH = 'TRAINING_BATCH'

CLASS_OUTPUT = 'CLASS_OUTPUT'
REGRESS_OUTPUT = 'REGRESS_OUTPUT'

# Data set partition
TRAIN = 'TRAIN'
VALIDATE = 'VALIDATE'
TEST = 'TEST'

# Training mode
FROM_SCRATCH = 'FROM_SCRATCH'
FINE_TUNING = 'FINE_TUNING'

models_name = []
models_imp = dict()
import_erros = []


def get_property(name, id):
    global models_imp, DATA_DIR
    try:
        return models_imp[(name, id)]
    except KeyError:
        dataset = models_imp[(name, DATA_DIR)]
        return models_imp[(dataset, id)]


def import_model():
    global models_name, models_imp, import_erros

    # --------------------------------------imagenet--------------------------------------
    try:
        from models.imagenet import imagenet_models_utils
        dataset_name = 'imagenet'

        models_imp[(dataset_name, LOAD_DATA_LIST)] = imagenet_models_utils.load_data_list
        models_imp[(dataset_name, VISUALIZE_OUTPUT)] = imagenet_models_utils.visualize_output
        models_imp[(dataset_name, OUTPUT_TYPE)] = CLASS_OUTPUT

        try:
            import keras.applications.xception as xcep
            name = 'Xception'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = xcep.Xception
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(299, 299),
                                                              preprocess=xcep.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 16
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.vgg16 as vgg16
            name = 'VGG16'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = vgg16.VGG16
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=vgg16.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 32
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.vgg19 as vgg19
            name = 'VGG19'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = vgg19.VGG19
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=vgg19.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 4
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.resnet50 as resnet50
            name = 'ResNet50'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = resnet50.ResNet50
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=resnet50.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 16
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.inception_v3 as inception_v3
            name = 'InceptionV3'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = inception_v3.InceptionV3
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(299, 299),
                                                              preprocess=inception_v3.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 24
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.inception_resnet_v2 as inception_resnet_v2
            name = 'InceptionResNetV2'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = inception_resnet_v2.InceptionResNetV2
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(299, 299),
                                                              preprocess=inception_resnet_v2.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 8
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.mobilenet as mobilenet
            name = 'MobileNet'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = mobilenet.MobileNet
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=mobilenet.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 48
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.densenet as densenet

            name = 'DenseNet121'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = densenet.DenseNet121
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=densenet.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 32

            name = 'DenseNet169'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = densenet.DenseNet169
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=densenet.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 32

            name = 'DenseNet201'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = densenet.DenseNet201
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=densenet.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 16
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.nasnet as nasnet

            name = 'NASNetLarge'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = nasnet.NASNetLarge
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(331, 331),
                                                              preprocess=nasnet.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 4

            name = 'NASNetMobile'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = nasnet.NASNetMobile
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=nasnet.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 64
        except ImportError:
            import_erros.append(traceback.format_exc())

        try:
            import keras.applications.mobilenetv2 as mobilenetv2
            name = 'MobileNetV2'
            models_name.append(name)
            models_imp[(name, DATA_DIR)] = dataset_name
            models_imp[(name, INIT_METHOD)] = mobilenetv2.MobileNetV2
            models_imp[(name, LOAD_DATA)] = functools.partial(imagenet_models_utils.load_data, target_size=(224, 224),
                                                              preprocess=mobilenetv2.preprocess_input)
            models_imp[(name, TRAINING_BATCH)] = 64
        except ImportError:
            import_erros.append(traceback.format_exc())

    except ImportError:
        import_erros.append(traceback.format_exc())

    # --------------------------------------driving--------------------------------------
    try:
        from models.driving import driving_models, driving_models_utils

        dataset_name = 'driving'

        models_imp[(dataset_name, LOAD_DATA_LIST)] = driving_models_utils.load_data_list
        models_imp[(dataset_name, LOAD_DATA)] = functools.partial(driving_models_utils.load_data,
                                                                  target_size=(100, 100))
        models_imp[(dataset_name, VISUALIZE_OUTPUT)] = driving_models_utils.visualize_output
        models_imp[(dataset_name, OUTPUT_TYPE)] = REGRESS_OUTPUT

        name = 'DaveOrig'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(driving_models.Dave_orig, load_weights=True)

        name = 'DaveNorminit'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(driving_models.Dave_norminit, load_weights=True)

        name = 'DaveDropout'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(driving_models.Dave_dropout, load_weights=True)
    except ImportError:
        import_erros.append(traceback.format_exc())

    # --------------------------------------mnist--------------------------------------
    try:
        from models.mnist import mnist_models_utils

        dataset_name = 'mnist'

        models_imp[(dataset_name, LOAD_DATA_LIST)] = mnist_models_utils.load_data_list
        models_imp[(dataset_name, LOAD_DATA)] = mnist_models_utils.load_data
        models_imp[(dataset_name, VISUALIZE_OUTPUT)] = mnist_models_utils.visualize_output
        models_imp[(dataset_name, OUTPUT_TYPE)] = CLASS_OUTPUT
        models_imp[(dataset_name, TRAINING_BATCH)] = 128
        models_imp[(dataset_name, OPTIMIZER)] = mnist_models_utils.get_optimizer()

        name = 'LeNet1'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(mnist_models_utils.load_lenet_model, name=name)

        name = 'LeNet4'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(mnist_models_utils.load_lenet_model, name=name)

        name = 'LeNet5'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(mnist_models_utils.load_lenet_model, name=name)
    except ImportError:
        import_erros.append(traceback.format_exc())

    # --------------------------------------thai_mnist--------------------------------------
    try:
        from models.thai_mnist import thai_mnist_models_utils

        dataset_name = 'thai_mnist'

        name = 'ThaiMnist'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = thai_mnist_models_utils.load_thai_mnist_model
        models_imp[(name, LOAD_DATA_LIST)] = thai_mnist_models_utils.load_data_list
        models_imp[(name, LOAD_DATA)] = thai_mnist_models_utils.load_data
        models_imp[(name, VISUALIZE_OUTPUT)] = thai_mnist_models_utils.visualize_output
        models_imp[(name, OUTPUT_TYPE)] = CLASS_OUTPUT
        models_imp[(name, TRAINING_BATCH)] = 64
    except ImportError:
        import_erros.append(traceback.format_exc())

    # --------------------------------------betago--------------------------------------
    try:
        from models.betago import betago_models_utils

        dataset_name = 'betago'

        name = 'Betago'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = betago_models_utils.load__betago_model
        models_imp[(name, LOAD_DATA_LIST)] = betago_models_utils.load_data_list
        models_imp[(name, LOAD_DATA)] = betago_models_utils.load_data
        models_imp[(name, VISUALIZE_OUTPUT)] = betago_models_utils.visualize_output
        models_imp[(name, OUTPUT_TYPE)] = CLASS_OUTPUT
    except ImportError:
        import_erros.append(traceback.format_exc())
    # --------------------------------------anime_faces--------------------------------------
    try:
        from models.anime_faces import anime_faces_models_utils

        dataset_name = 'anime_faces'

        name = 'AnimeFaces'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = anime_faces_models_utils.load_anime_faces_model
        models_imp[(name, LOAD_DATA_LIST)] = anime_faces_models_utils.load_data_list
        models_imp[(name, LOAD_DATA)] = anime_faces_models_utils.load_data
        models_imp[(name, VISUALIZE_OUTPUT)] = anime_faces_models_utils.visualize_output
        models_imp[(name, OUTPUT_TYPE)] = CLASS_OUTPUT
    except ImportError:
        import_erros.append(traceback.format_exc())
    # --------------------------------------cat_dog_conv--------------------------------------
    try:
        from models.cat_dog_conv import cat_dog_conv_models_utils

        dataset_name = 'cat_dog_conv'

        models_imp[(dataset_name, LOAD_DATA_LIST)] = cat_dog_conv_models_utils.load_data_list
        models_imp[(dataset_name, LOAD_DATA)] = cat_dog_conv_models_utils.load_data
        models_imp[(dataset_name, VISUALIZE_OUTPUT)] = cat_dog_conv_models_utils.visualize_output
        models_imp[(dataset_name, OUTPUT_TYPE)] = CLASS_OUTPUT

        name = 'CatDogBsicCNN'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(cat_dog_conv_models_utils.load_cat_dog_conv_model,
                                                            name="basic_cnn")

        name = 'CatDogAugmented'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(cat_dog_conv_models_utils.load_cat_dog_conv_model,
                                                            name="augmented")
    except ImportError:
        import_erros.append(traceback.format_exc())
    # --------------------------------------dog--------------------------------------
    try:
        from models.dog import dog_models_utils

        dataset_name = 'dog'

        name = 'Dog'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = dog_models_utils.load_dog_model
        models_imp[(name, LOAD_DATA_LIST)] = dog_models_utils.load_data_list
        models_imp[(name, LOAD_DATA)] = dog_models_utils.load_data
        models_imp[(name, VISUALIZE_OUTPUT)] = dog_models_utils.visualize_output
        models_imp[(name, OUTPUT_TYPE)] = CLASS_OUTPUT
    except ImportError:
        import_erros.append(traceback.format_exc())
    # --------------------------------------gender--------------------------------------
    '''
    try:
        from models.gender import gender_models_utils

        dataset_name = 'gender'

        name = 'Gender'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = gender_models_utils.load_gender_model
        models_imp[(name, LOAD_DATA_LIST)] = gender_models_utils.load_data_list
        models_imp[(name, LOAD_DATA)] = gender_models_utils.load_data
        models_imp[(name, VISUALIZE_OUTPUT)] = gender_models_utils.visualize_output
        models_imp[(name, OUTPUT_TYPE)] = CLASS_OUTPUT
    except ImportError:
        import_erros.append(traceback.format_exc())
    '''
    # --------------------------------------pokedex--------------------------------------
    try:
        from models.pokedex import pokedex_models_utils

        dataset_name = 'pokedex'

        name = 'Pokedex'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = pokedex_models_utils.load_pokedex_model
        models_imp[(name, LOAD_DATA_LIST)] = pokedex_models_utils.load_data_list
        models_imp[(name, LOAD_DATA)] = pokedex_models_utils.load_data
        models_imp[(name, VISUALIZE_OUTPUT)] = pokedex_models_utils.visualize_output
        models_imp[(name, OUTPUT_TYPE)] = CLASS_OUTPUT
    except ImportError:
        import_erros.append(traceback.format_exc())
    # --------------------------------------traffic_signs--------------------------------------
    '''
    try:
        from models.traffic_signs import traffic_signs_models_utils

        dataset_name = 'traffic_signs'

        models_imp[(dataset_name, LOAD_DATA_LIST)] = traffic_signs_models_utils.load_data_list
        models_imp[(dataset_name, VISUALIZE_OUTPUT)] = traffic_signs_models_utils.visualize_output
        models_imp[(dataset_name, OUTPUT_TYPE)] = CLASS_OUTPUT

        name = 'TrafficSignsModel1'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(traffic_signs_models_utils.load_traffic_sign_model,
                                                            model_name="Model1")
        models_imp[(name, LOAD_DATA)] = functools.partial(traffic_signs_models_utils.load_data, model_name="Model1")
        models_imp[(name, TRAINING_BATCH)] = 64

        name = 'TrafficSignsModel2'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(traffic_signs_models_utils.load_traffic_sign_model,
                                                            model_name="Model2")
        models_imp[(name, LOAD_DATA)] = functools.partial(traffic_signs_models_utils.load_data, model_name="Model2")
        models_imp[(name, TRAINING_BATCH)] = 32

        name = 'TrafficSignsModel3'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(traffic_signs_models_utils.load_traffic_sign_model,
                                                            model_name="Model3")
        models_imp[(name, LOAD_DATA)] = functools.partial(traffic_signs_models_utils.load_data, model_name="Model3")
        models_imp[(name, TRAINING_BATCH)] = 32
    except ImportError:
        import_erros.append(traceback.format_exc())
    '''

    # --------------------------------------cifar10--------------------------------------
    try:
        from models.cifar10 import cifar10_models_utils

        dataset_name = 'cifar10'
        models_imp[(dataset_name, LOAD_DATA_LIST)] = cifar10_models_utils.load_data_list
        models_imp[(dataset_name, LOAD_DATA)] = functools.partial(cifar10_models_utils.load_data)
        models_imp[(dataset_name, VISUALIZE_OUTPUT)] = cifar10_models_utils.visualize_output
        models_imp[(dataset_name, OUTPUT_TYPE)] = CLASS_OUTPUT
        models_imp[(dataset_name, TRAINING_BATCH)] = 128
        models_imp[(dataset_name, OPTIMIZER)] = cifar10_models_utils.get_optimizer()

        name = 'ResNet56v1'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(cifar10_models_utils.load_resnet_model, name=name)

        name = 'ResNet38v1'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(cifar10_models_utils.load_resnet_model, name=name)

        name = 'ResNet32v1'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = functools.partial(cifar10_models_utils.load_resnet_model, name=name)
        models_imp[(dataset_name, ADV_OPTIMIZER)] = cifar10_models_utils.get_adv_optimizer()

        name = 'WRN-34-10'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = cifar10_models_utils.load_wideresnet_model
        models_imp[(dataset_name, ADV_OPTIMIZER)] = cifar10_models_utils.get_adv_optimizer()

    except ImportError:
        import_erros.append(traceback.format_exc())

    # --------------------------------------cifar100-------------------------------------
    try:
        from models.cifar100 import cifar100_models_utils

        dataset_name = 'cifar100'
        models_imp[(dataset_name, LOAD_DATA_LIST)] = cifar100_models_utils.load_data_list
        models_imp[(dataset_name, LOAD_DATA)] = functools.partial(cifar100_models_utils.load_data)
        models_imp[(dataset_name, VISUALIZE_OUTPUT)] = cifar100_models_utils.visualize_output
        models_imp[(dataset_name, OUTPUT_TYPE)] = CLASS_OUTPUT
        models_imp[(dataset_name, TRAINING_BATCH)] = {'tensorflow': 128, 'cntk': 64, 'theano': 64}
        models_imp[(dataset_name, OPTIMIZER)] = cifar100_models_utils.get_optimizer()

        name = 'WRN-28-10'
        models_name.append(name)
        models_imp[(name, DATA_DIR)] = dataset_name
        models_imp[(name, INIT_METHOD)] = cifar100_models_utils.load_wideresnet_model

    except ImportError:
        import_erros.append(traceback.format_exc())

    imagenet_models = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'InceptionResNetV2', 'MobileNet',
                       'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetLarge', 'NASNetMobile',
                       'MobileNetV2']
    mnist_models = ['LeNet1', 'LeNet4', 'LeNet5']
    thaimnist_models = ['ThaiMnist']
    cifar10_models = ['ResNet56v1', 'ResNet38v1']
    cifar100_models = ['WRN-28-10']
    # traffic_models = ['TrafficSignsModel1', 'TrafficSignsModel2', 'TrafficSignsModel3']

    models_name = imagenet_models + mnist_models + thaimnist_models + cifar10_models + cifar100_models  # + traffic_models
