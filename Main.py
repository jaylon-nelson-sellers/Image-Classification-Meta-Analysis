
from LoadImageDatasets import combine_and_reshape_images, load_dataset, load_mnist, load_fashion, load_cifar_10, load_cifar_100_coarse, load_cifar_100_fine
from RunTests import *
def __main__(dataset_id:str,model_id:str):
    if model_id == "cnn":
        dataset = load_dataset(dataset_id,True)
        cnn_tests(dataset_id,dataset_data=dataset)
    if model_id == "boost":
        dataset = load_dataset(dataset_id)
        xgb_tests(dataset_id,dataset_data=dataset)
    dataset = load_dataset(dataset_id,False)
    if model_id == "dummy":
            dummy_test(dataset_id,dataset_data=dataset)
    if model_id == "sk":
        sk_test(dataset_id,dataset_data=dataset)
   
    if model_id == "nn":
        nn_tests(dataset_id,dataset_data=dataset)
    
    

dataset_id = "cifar10"
model_id = "cnn"
__main__(dataset_id,model_id)
