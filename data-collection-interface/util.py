from matplotlib import pyplot as plt
import matplotlib

# Function found on StackOverflow (url: https://stackoverflow.com/a/45734500) 
# What it does is implement pytplot.pause(), but without focussing the canvas afterwards.
# This gets rid of the jittery experience when using pyplot.pause() in the data collection process :)
def non_focusing_pause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return
