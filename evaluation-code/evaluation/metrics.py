from matplotlib import pyplot as plt
import os

class Metrics:
    @staticmethod
    def plot_history(history, model_name, res_string):
        directory = f"./output/overfitting/{model_name}/"
        os.makedirs(directory, exist_ok=True)

        # Summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(f'{model_name} accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.show()
        # Save plot
        plt.savefig(os.path.join(directory, 'acc.png'))
        plt.close()

        # Summarize history for accuracy
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_name} loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.show()
        # Save plot
        plt.savefig(os.path.join(directory, 'loss.png'))
        plt.close()

        with open(os.path.join(directory, 'result.txt'), 'w') as file:
            file.write(res_string)