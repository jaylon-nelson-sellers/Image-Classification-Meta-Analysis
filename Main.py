
from LoadImageDatasets import load_mnist, load_fashion, load_cifar_10, load_cifar_100_coarse, load_cifar_100_fine
from RunTests import *
def __main__(mnist:bool=False, fashion:bool=False,cifar10:bool=False,coarse:bool=False,fine:bool=False,
             dummy:bool=False, sk:bool=False,boost:bool=False,nn:bool=False,cnn:bool=False):
    if mnist:
        dataset = load_mnist(debug=False)
        test_id = "MNIST"
    if fashion:
        dataset = load_fashion(debug=False)
        test_id = "Fashion"
    if cifar10:
        dataset = load_cifar_10()
        test_id = "CIFAR10"
    if coarse:
        dataset = load_cifar_100_coarse()
        test_id = "CIFAR Coarse"
    if fine:
        dataset = load_cifar_100_fine()
        test_id = "CIFAR Fine"
    if dummy:
        dummy_test(test_id,dataset_data=dataset)
    if sk:
        sk_test(test_id,dataset_data=dataset)
    if boost:
        xgb_tests(test_id,dataset_data=dataset)
    if nn:
        nn_tests(test_id,dataset_data=dataset)
    if cnn:
        cnn_tests(test_id,dataset_data=dataset)
    


__main__(mnist=True,fashion=False,cifar10=False,coarse=False,fine=False,dummy=False,sk=False,boost=False,nn=False,cnn=True)
