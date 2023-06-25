# Import module by Research Group
from handledata import collector, gesture_data
from util import non_focusing_pause
# Import modules used.
import numpy as np
from matplotlib import pyplot as plt
import threading

# Though this is a lot of my own code, the main idea for getting around "block=False" on plots not working comes from
# my group member, Winstijn Smit, which I have to thank him for greatly.

DATASET_FOLDER = "data/"
DIGITS = ["#" + str(i) for i in range(10)]  # 0, #1, #2, #3, etc.

CANDIDATE = "g9"
GESTURE = DIGITS[9]
HAND = "right_hand"
SAVE_TO_FILE = True

SAMPLE_RATE = 1000
SAMPLE_DURATION = 2 * SAMPLE_RATE
START_DELAY = 500  # Delay before measurment starts in ms.
THRESHOLD = 10

# Commands
MEASUREMENT_DETECTION_LOOP = 0xAA
MEASUREMENT_START = 0xAB
RECALIBRATE = 0xAC
SET_SAMPLE_RATE = 0xAD

# Whole gesture selection UI is created from these constants.
GESTURE_TYPE = "digits"


class DataCollector:
    def __init__(self):
        # Collecting stuff
        self.collector = collector.Collector()
        self.resistance = None
        self.collected_data = []
        # Locking stuff
        self.lock = threading.Semaphore(0)

    # This is used to initialize setup
    def put_in_detection_mode(self):
        # Tries to recalibrate the device.
        if self.resistance == None:
            print("Recalibating the photodiodes!")
            self.resistance = self.collector.recalibrate()
        # Send command for measurement
        print("Sending signal to start continuous measurements!")
        self.collector.write_bytes(MEASUREMENT_DETECTION_LOOP)
        # Also send threshold
        self.collector.write_bytes(self.collector.to_bytes(THRESHOLD))

    # This method listens to the serial connection, and whenever the arduino sends the
    # start sign, #0xAA it will start data collection.
    def measurement_listening_loop(self):
        while True:
            # Read single character until start sign is detected.
            r = self.collector.readuint16()
            if (r == 0xAA):
                # Prepare data object for sample.
                duration_secs = float(SAMPLE_DURATION) / float(SAMPLE_RATE)
                data = gesture_data.GestureData(
                    self.resistance, SAMPLE_RATE, duration_secs)
                print()
                print(f"Ready to collect gesture of size: {duration_secs}")

                # Do data collection
                print("data is being collected!")
                data.collect(collector=self.collector)
                self.collected_data.append(data)
                # Release lock, which allows the main thread to plot the data.
                self.lock.release()


if __name__ == "__main__":
    # Load up dat plate
    data_collector = DataCollector()

    # Send Arduino command to do constant gesture detection and collection.
    print("Putting the ardiono in constant detection mode!")
    data_collector.put_in_detection_mode()
    # quit()
    print(f"Collecting {GESTURE} from {CANDIDATE}'s {HAND} at {SAMPLE_RATE}hz")

    plt.ion()

    # Pitch lookin lovely today.
    thread = threading.Thread(
        target=data_collector.measurement_listening_loop)
    thread.start()

    # Start UIs stuff
    while True:
        # While we wait for new gesture, keep plot windows alive.
        while data_collector.lock.acquire(blocking=True, timeout=0.01) == False:
            # was plt.pause(0.2), see non_focusing_pause()'s documentation.
            non_focusing_pause(0.2)

        # Newly detected gesture!
        gesture = data_collector.collected_data[-1]
        gesture.set_metadata(
            candidate=CANDIDATE,
            hand=HAND,
            gesture_type=GESTURE_TYPE,
            target_gesture=GESTURE,
        )

        # Draw plot on screen!
        gesture.plot(show=True)
        # plt.pause(0.001)
        # Save gesture to a file!
        if SAVE_TO_FILE:
            gesture.save_to_file()  # Save the data to a file.
