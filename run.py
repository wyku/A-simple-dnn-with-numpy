import numpy as np
import model
import data

if __name__ == '__main__':

    NUMBERS_OF_DATA = 2000
    NETWORK_SHAPE = [2, 16, 32, 16, 2]
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-1

    total_data = data.generate_circle_inputs(NUMBERS_OF_DATA)
    # 80% for training, 20% for testing
    train_data = total_data[:int(NUMBERS_OF_DATA * 0.8)]
    test_data = total_data[int(NUMBERS_OF_DATA * 0.8):]
    print(f"nums of train_data: {len(train_data)}, nums of test_data: {len(test_data)}")
    print('--------------------')

    network_shape = np.array(NETWORK_SHAPE)
    mlp_model = model.Network(network_shape)
    outputs = mlp_model.training(train_data=train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, test_data=test_data)
    
    data.plot_data(test_data, title="True fact")
    data.plot_data(outputs, title="Predict")
