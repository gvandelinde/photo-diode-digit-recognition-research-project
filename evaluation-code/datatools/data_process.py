import numpy as np
import matplotlib.pyplot as plotter
from .load_gestures import *
from sklearn.model_selection import train_test_split
from itertools import chain
from sklearn.preprocessing import OneHotEncoder, normalize
import random


# A loaded gesture will contain the following keys:
# ['timestamp', 'candidate', 'hand', 'gesture_type', 'target_gesture', 'resistance', 
#    'sample_rate', 'duration', 'samples', 'data']
class GestureData:
    def __init__(self, data, metadata):
        # Contain Raw data from photo diodes. 
        self.data = data 
        # Contain Object metadata.
        self.metadata = metadata
        # Unpack some metadata attributes
        self.candidate = metadata['candidate']
        # These attributes are for tracking preprocessing steps.
        self.framed = False
        self.normalised = False
        self.downsampled = False
   
    # Normalise data, and put it back into data attribute.  
    # NOTE: data is normalised across the 3 channels, meaning not per channel.
    # Thus, the highest and lowest total datapoints are used to normalize in between.
    def normalize(self):
        if self.framed:
            raise Exception("Can't normalize framed data (yet)")
        if self.normalised:
            print("You just tried to renormalised the data. Ain't happening son.")
            return
        self.normalised = True
        # Get all
        x = self.data
        data_norm = (x-np.min(x))/(np.max(x)-np.min(x))
        self.data = data_norm
        return 


    def nomalize_seperate_channels(self):
        x = self.data.T
        # print(normalize(self.data))

        # print(x.shape)
        s0 = normalize_arr(x[0])
        s1 = normalize_arr(x[1])
        s2 = normalize_arr(x[2])
        s = np.array([s0, s1, s2])

        self.data = s.T

    #NOTE: with interpolation!
    def downsample(self, freq=100):
        current_freq = self.metadata['sample_rate']
        step = int(current_freq / freq)
        if step < 1 or step > current_freq:
            raise Exception("You just chose an invalid frequency... :(")
        new_data = []
        for column in np.transpose(self.data):
            downsampled_column = [] 
            for i in range(0, len(column), step):
                downsampled_column.append(np.mean(column[i: i+step]))
            new_data.append(downsampled_column)
        # print(f"Message from your local downsampler: the shape we wil transpose is {np.shape(new_data)}") 

        self.data = np.transpose(new_data) 
        # Make metadata changes.
        self.metadata['samples'] = len(self.data)
        self.metadata['duration'] = len(self.data) / step
        self.metadata['sample_rate'] = freq
        return

    # Add padding on both size iff amount of samples is not target size
    def addPadding(self, target_size=2000):
        current_size = self.metadata['samples'] 
        if current_size == target_size:
            return
        #NOTE: this assumes target_size is larger then current.
        pad_size = int((target_size - current_size)/2)
        padded_data = [[], [], []]
        for i in range(3):
            front_padding = np.full((pad_size, ), self.data[0][i])
            back_padding = np.full((pad_size, ), self.data[-1][i])
            padded_data[i] = np.concatenate((front_padding, self.data.T[i], back_padding))
        self.data = np.transpose(padded_data)
            
    def divide_sample_into_frames(self):
        sample = np.array(self.data)
        frames = []
        frame_size = 5
        for i in range(0, len(sample), frame_size):
            # Take slice that becomes frame and flatten it.
            array = sample[i : i + frame_size]
            frames.append(array)
        self.data = frames
        self.framed = True
        return

    # Plots the data contained in the GestureData on a graph.
    # This method was modified from a copy from the DataCollection tool on GitHub:
    # https://github.com/arnedebeer/CSE3000-DataCollection/blob/main/data_collection_interface/gesture_data.py
    #
    def plot(self) -> None:
        # print(f"Normalized: {self.normalised}, Divided into frames: {self.framed}")
        # candidate = self.metadata['candidate']
        # target_gesture = self.metadata['target_gesture']
        title = f"{self.metadata['target_gesture']} with {self.metadata['hand']} by {self.metadata['candidate']}"

        # Create the e)plot, together with a section for the metadata.
        fig, plt = plotter.subplots(1)
        fig.subplots_adjust(bottom=0.3)

        # Plot the data.
        if self.framed:
            plt.plot(np.reshape(self.data, (-1, 3)))
        else:
            plt.plot(self.data)
        # Set the labels of the axes.
        plt.set_xlabel("Samples")
        plt.set_ylabel("Photodiode reading")
        plt.set_title(title)

        # Set the metadata of the plot.
        fig.text(0.1,0.15,'Sampling Rate: ' + str(self.metadata['sample_rate']) + 'Hz')
        fig.text(0.1,0.10,'Time: ' + str(self.metadata['duration']) + 's')
        fig.text(0.1,0.05,'Resistance: ' + str(self.metadata['resistance'] / 1000) + 'kOhm')


        # # Save location of the image.
        # path = "plots/" + title.lower().replace(" ", "_") + ".png"
        # create_directories(path)
        # plotter.savefig(path)

        plotter.show()

    def return_preprocessed(self, divide_into_frames:bool = False):
        self.addPadding()
        self.normalize()
        self.downsample(freq=100)
        if divide_into_frames: self.divide_sample_into_frames()
        return self.data 

    def __str__(self) -> str:
        return self.metadata.__str__() + f"\nShape of data: {np.shape(self.data)}"

# Related static methods

def normalize_arr(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def one_hot_encode_labels(y_train, y_test):
# One hot encode labels
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    y_train_reshaped = np.array(y_train).reshape(-1, 1)
    y_test_reshaped = np.array(y_test).reshape(-1, 1)
    enc = enc.fit(y_train_reshaped)

    y_train_ohe = enc.transform(y_train_reshaped)
    y_test_ohe = enc.transform(y_test_reshaped)
    return y_train_ohe, y_test_ohe, enc
 
def load_candidate_map(hand: Hand = Hand.right):
    # Have dicitonairy with all candidates
    candidate_map = {}
    amount_of_samples_per_digit = 2 ^ 31
    for name in DigitNames:
        loaded_gestures_per_candidate, smallest_found = load_digits_per_candidate(digit_name=name, hand=hand)
        # Pick smallest found samples from each candidate:
        amount_of_samples_per_digit = min(smallest_found, amount_of_samples_per_digit)
        for (cand, samples) in loaded_gestures_per_candidate:
            #NOTE: e5 gets excluded, since it has ragged data!
            if "e5" in cand:
                continue
            # See if candidate is already in map.
            for sample in samples:
                data = sample.pop('data')
                if cand in candidate_map:
                    if name.value in candidate_map[cand]:
                        candidate_map[cand][name.value].append(GestureData(data, sample))
                    else:
                        candidate_map[cand][name.value] = [GestureData(data, sample)]
                else:
                    candidate_map[cand] = {}
                    candidate_map[cand][name.value]= [GestureData(data, sample)]
    # print(f"smallest amount of samples on a digit: {amount_of_samples_per_digit}")
    # Take random amount of samples found for each candidate
    final_map = {}
    # print("make final map:")
    for i in candidate_map:
        for d in candidate_map[i]:
            if i not in final_map:
                final_map[i] = []
            # print(len(candidate_map[i][d]))
            digit_identifiers = [d for i in range(amount_of_samples_per_digit)]
            digit_specific_samples = random.sample(candidate_map[i][d], amount_of_samples_per_digit)
            final_map[i].extend(zip(digit_identifiers, digit_specific_samples))
    # print(np.shape(final_map[0]))
    return final_map 

def divide_cand_split(cand_map, hand: Hand = Hand.right, test_size:int = 0.2, shuffle: bool = True):
    # Expects a dictionary with candidates as keys. Items are an array of tuples like this: (Gesture, Data object)

    keys = sorted(list(cand_map.keys()))
    # Make test split on keys
    train_keys, test_keys = train_test_split(keys, shuffle=shuffle, test_size=test_size)


    # Fetch selected keys for split
    training_candidates = [cand_map[i] for i in train_keys]
    testing_candidates = [cand_map[i] for i in test_keys]

    # print(type(training_candidates[0][0][0]))
    # Merge all arrays into single array
    merged_train_list_candidates = list(chain(*training_candidates))
    merged_test_list_candidates = list(chain(*testing_candidates))

    # Unzip all the tuples
    y_train, X_train= list(zip(*merged_train_list_candidates))
    y_test, X_test = list(zip(*merged_test_list_candidates))

    return list(X_train), list(X_test), list(y_train), list(y_test)

# Most important method, used for getting all training data quickly.
def load_preprocessed_data_split_on_candidate(hand: Hand = Hand.right, test_size:int = 0.1, divide_into_frames:bool=False, shuffle=True, log=False, normalize_per_channel:bool = False):
    if log: print(f"Loading preprocessed data and splitting with the following parameters: hand: {hand}, test_size: {test_size}, divide into frames: {divide_into_frames}, shuffle: {shuffle}")
    # Create map of candadite with their data.
    cand_map = load_candidate_map(hand)

    # Take that amount of samples per candidate.
    X_train_samples, X_test_samples, y_train, y_test = divide_cand_split(cand_map, hand=hand, test_size=test_size, shuffle=shuffle)

    # Preprocessing
    for sample in X_train_samples:
        # sample.addPadding(target_size=2000)
        if normalize_per_channel :
            sample.nomalize_seperate_channels()
        else:
            sample.normalize()
        sample.downsample(freq=100)
        if divide_into_frames: sample.divide_sample_into_frames()
    for sample in X_test_samples:
        # sample.addPadding(target_size=2000)
        if normalize_per_channel:
            sample.nomalize_seperate_channels()
        else:
            sample.normalize()
        sample.downsample(freq=100)
        if divide_into_frames: sample.divide_sample_into_frames()

    # Now take out data from gesture_data class and reshape for the dl-model.
    # Also reshape where necessary.
    X_train = np.array([sample.data for sample in X_train_samples])
    if divide_into_frames: X_train = X_train.reshape(-1, 40, 5, 3)
    X_test = np.array([sample.data for sample in X_test_samples])
    if divide_into_frames: X_test = X_test.reshape(-1, 40, 5, 3)
    # USEFUL DEBUGGING CODE
    # ragged_indices = np.where([len(row) != len(X_train[0]) for row in X_train])
    # print(ragged_indices)
    # print([X_train_samples[i].candidate for i in ragged_indices])
    
    # One hot encode labels
    y_train, y_test, enc = one_hot_encode_labels(y_train, y_test)

    if log: print(f"size of training set: {len(X_train)}\nsize of test set: {len(X_test)}")

    # Return data
    return X_train, X_test, y_train, y_test, enc



# Maybe Not So Useful Helpers from here

def load_samples_for_digit(digit_name: DigitNames, hand: Hand = Hand.right):
    samples = load_digit_samples(digit_name, hand) 
    result = []
    for s in samples:
        data = s.pop('data')
        result.append(GestureData(data, s)) 
    return result

def load_all_preprocessed_digits_per_candidate(hand: Hand = Hand.right, divide_into_frames:bool = False):
        # Have dicitonairy with all candidates
        candidate_map = {}
        for name in DigitNames:
            loaded_gestures_per_candidate = load_digits_per_candidate(digit_name=name, hand=hand)
            for (cand, samples) in loaded_gestures_per_candidate:
                # See if candidate is already in map.
                for sample in samples:
                    data = sample.pop('data')
                    gesture = GestureData(data, sample)
                    if cand in candidate_map:
                        candidate_map[cand].append((name.value, gesture.preprocessed(divide_into_frames=divide_into_frames)))
                    else:
                        candidate_map[cand] = [(name.value, gesture.preprocessed(divide_into_frames=divide_into_frames))]

