import numpy as np

def gaussian_spectrum_1D(rand_signal, epsilon, sample_period=1.0):
    """
    Low pass filter a 1D signal with power spectral density (PSD):
    PSD(k) = 1 / (- k² / 2ϵ)

    Parameters
    ----------
    rand_signal : np.ndarray
        1D array with random values (white noise).
    epsilon : float
        Width of the Gaussian PSD.
    sample_period : float
        Sampling period of `rand_signal`. That means the time difference
        between consecutive signal measurements.

    Returns
    -------
    filtered_rand_signal : np.ndarray
        Low pass filtered signal (in time domain).
    """
    # The `rand_signal` signal is finite and discrete. Therefore, the Discrete
    # Fourier Transoform is used to get the Fourier coefficients for a finite
    # set of discrete Fourier frequencies (k). An efficient implementation of
    # the Discrete Fourier Transform is the Fast Fourier Transform, which we
    # will use below (np.fft.fft). The number of Fourier frequencies in your
    # transformed signal (in frequency domain) is the same as the number of
    # discrete time points in your original signal (in time domain).
    fft_rand_signal = np.fft.fft(rand_signal)

    # The np.fft.fft returned the Fourier coefficients. But for which
    # frequencies? Now that depends on the sampling period of the original
    # signal, that means the time difference between the discrete time points
    # at which you recorded (or in your case, randomly sampled) your signal.
    # To get the correct Fourier frequencies, you can use np.fft.fftfreq, which
    # takes your signal size and its sampling period as input.
    frq_rand_signal = np.fft.fftfreq(rand_signal.size, sample_period)

    # The PSD of a signal is defined as PSD(k) = |F(k)|^2, where F(k) is the
    # Fourier coefficient of frequency k. Therefore, we choose our filter in
    # frequency domain to be the sqrt of the PSD that we wan't to get.
    filter_low = np.sqrt(np.exp(- frq_rand_signal**2 / (2*epsilon)))
    # Normalize the filter
    filter_low = filter_low / np.sum(filter_low)

    # Now we apply the filter in Fourier domain (which is the same as
    # convolving in time domain, see Convolution Theorem)
    filtered_fft_rand_signal = fft_rand_signal * filter_low
    # And last, we transform the filtered signal back to the time domain, using the inverse Fast Fourier Transform (np.fft.ifft).
    filtered_rand_signal = np.real(np.fft.ifft(filtered_fft_rand_signal))

    # return filtered signal
    return filtered_rand_signal
