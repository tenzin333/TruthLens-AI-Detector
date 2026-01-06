import cv2
import numpy as np

def detect_frequency_artifacts(image_path):
    img = cv2.imread(image_path, 0) # Load as grayscale
    f = np.fft.fft2(img)           # Shift to frequency domain
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # High spikes in the corners of the spectrum usually mean AI!
    return magnitude_spectrum