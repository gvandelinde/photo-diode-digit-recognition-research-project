from datatools.data_process import one_hot_encode_labels, load_candidate_map
from itertools import chain
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from .utils import get_timestr
import os

class KFoldEval:
    @staticmethod
    def k_fold_cross_validation(model_builder = None, model_name = "Undefined name", divide_into_frames:bool = None, normalize_per_channel:bool = None, n_splits:int = 5, log:bool = False, epochs:int= None):
        if model_builder is None or divide_into_frames is None or normalize_per_channel is None or epochs is None:
            raise Exception("Please define all arguments.")
    
        # Do KFold split and load candidates
        kf_cand = KFold(n_splits = n_splits)
        candidates = load_candidate_map() 

        candidate_keys = [it.lower() for it in list(candidates.keys())]
        kf_cand_split = kf_cand.split(X=candidate_keys)
        results = []
        histories = []
        resulting_pred = np.array([])
        resulting_actual = np.array([])
        # Prepare data to split
        X, y = KFoldEval.handle_data(candidates, normalize_per_channel, divide_into_frames)
        # Do KFold on keys, construct list of candidates off of that.
        for i, (train_keys, test_keys) in enumerate(kf_cand_split):
            print(f"Fold {i + 1}!")
            
            # Make copy to be sure we can use candidates dict entries in next rounds!
            training_candidates = np.array([X[candidate_keys[it]].copy() for it in train_keys])
            testing_candidates = np.array([X[candidate_keys[it]].copy() for it in test_keys])

            # print([len(i) for i in training_candidates])
            X_train = training_candidates.reshape(-1, 40, 5, 3)
            X_test = testing_candidates.reshape(-1, 40, 5, 3)

            print("Shape of training data: " + str(np.shape(X_train)))
            print("Shape of test data: " + str(np.shape(X_test)))
            # Hot Encoding labels
            #TODO: FIX THIS LINE BELOW (36)
            y_train, y_test, enc = one_hot_encode_labels(
                y_train=[y[candidate_keys[it]] for it in train_keys], 
                y_test=[y[candidate_keys[it]] for it in test_keys], 
            )
            cv_model = model_builder()

            # Do fitting
            history = cv_model.fit(
                np.array(X_train), 
                np.array(y_train), 
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                shuffle=True,
                verbose = 2
            )

            histories.append(history)

            # Create Confusion Matrix
            predictions = np.array([f"#{i}" for i in np.argmax(cv_model.predict(np.array(X_test)), axis=1)])
            pred = predictions.reshape(-1, 1)
            actual = enc.inverse_transform(y_test)

            # Save confusion result, extend np-arrays.
            resulting_pred = np.append(resulting_pred, pred)
            resulting_actual = np.append(resulting_actual, actual)

            # Append result from model evaluation
            results.append(cv_model.evaluate(np.array(X_test), np.array(y_test), verbose=2))

        if log:
            params = f"Params: {history.params}"
            means = f"Mean acc: {np.mean(np.array(results).T[1].T)} with an std of: {np.std(np.array(results).T[1].T)}\nMean loss: {np.mean(np.array(results).T[0].T)}"
            
            # Prepare directory for saving files to
            directory = f"./output/results/{model_name}_{get_timestr()}"
            os.makedirs(directory, exist_ok=True)

            # Make confusion matrix
            disp = ConfusionMatrixDisplay.from_predictions(resulting_pred, resulting_actual, normalize="true")

            # Plot the confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax)

            plt.title(f"{model_name} on {n_splits} Folds with {epochs} Epochs")
            
            # Save the plot to a file
            plt.savefig(f'{directory}/confusion_matrix.png')  # Provide the desired file name and extension

            # Close the plot to free up resources
            plt.close()

            # Get filepath for results log
            file_path = os.path.join(directory, 'results.txt')
            results_concatted = "\n".join([f"loss: {i}, accuracy: {j}" for i,j in results])
        
            # Save results to a file   
            with open(file_path, 'w') as file:
                file.write(params + "\n" + means + "\n\n")
                file.write("Stats for each fold:")
                file.write(results_concatted) 

        return results, histories

    @staticmethod
    def handle_data(candidates, normalize_per_channel, divide_into_frames):
        X_samples = []
        for it in candidates:
            X_samples.extend(candidates[it])

        # Do preprocessing
        for _, sample in X_samples:
            # sample.addPadding(target_size=2000)
            if normalize_per_channel :
                sample.nomalize_seperate_channels()
            else:
                sample.normalize()
            # sample.addPadding(2000)
            # print("downsampling!")

            sample.downsample(freq=100)
            if divide_into_frames: sample.divide_sample_into_frames()

        X = {}
        y = {}
        for label, sample in X_samples:
            # print(f"adding a sample from {sample.candidate}")
            if sample.candidate.lower() not in X.keys():
                X[sample.candidate.lower()] = []
                y[sample.candidate.lower()] = []
            sample_data = np.array(sample.data)
            if divide_into_frames: 
                # if np.shape(sample_data) != (40, 5, 3):
                #     print(np.shape(sample_data))
                #     print(sample.candidate.lower())
                sample_data = sample_data.reshape(40, 5, 3)
            X[sample.candidate.lower()].append(sample_data)
            y[sample.candidate.lower()].append(label)
        return X, y 

    @staticmethod
    def plot_history(history):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        # summerize loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()