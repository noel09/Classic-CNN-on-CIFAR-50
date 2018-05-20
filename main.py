import sys

from dataset_loader import load_dataset
from models import m_alexnet, m_googlenet, m_resnet18, m_resnet50, m_resnet110



if __name__ == '__main__':
    training_dataset, testing_dataset = load_dataset()
    

    print("Menu:")
    print("1. AlexNet")
    print("2. GoogLeNet")
    print("3. ResNet-18")
    print("4. ResNet-50")
    print("5. ResNet-110")
    print("6. Exit")
    
    user_input = int(input("Enter your selection: "))
    
    if user_input > 5 or user_input < 1:
        sys.exit()
    
    epochs = int(input("\nNumber of epochs: "))
    lr = float(input("Learning rate: "))
    load_w = int(input("Load weights? (1 for yes, 0 for no) "))
    save_w = int(input("Save weights? (1 for yes, 0 for no) "))
    
    if load_w == 1:
        load_weights = True
    else:
        load_weights = False
        
    
    if save_w == 1:
        save_weights = True
    else:
        save_weights = False

    if user_input == 1:
        m_alexnet(training_dataset, testing_dataset, lr, epochs, load_weights, save_weights)
    elif user_input == 2:
        m_googlenet(training_dataset, testing_dataset, lr, epochs, load_weights, save_weights)
    elif user_input == 3:
        m_resnet18(training_dataset, testing_dataset, lr, epochs, load_weights, save_weights)
    elif user_input == 4:
        m_resnet50(training_dataset, testing_dataset, lr, epochs, load_weights, save_weights)
    elif user_input == 5:
        m_resnet110(training_dataset, testing_dataset, lr, epochs, load_weights, save_weights)
    else:
        sys.exit()