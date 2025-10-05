"""
Temporal waveforms for electromagnetic sources.

This module provides various time-dependent waveform functions for
electromagnetic sources, including Gaussian pulses, continuous waves,
and custom pulse shapes.
"""

from typing import Callable, Optional, Tuple, Union, Literal
import numpy as np


class Waveform:
    """
    Base class for time-dependent waveforms.

    Parameters
    ----------
    amplitude : float, optional
        Peak amplitude of the waveform, default=1.0.
    phase : float, optional
        Phase offset in radians, default=0.0.
    """

    def __init__(self, amplitude: float = 1.0, phase: float = 0.0):
        self.amplitude = amplitude
        self.phase = phase

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the waveform at given time(s).

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s) in seconds.

        Returns
        -------
        float or numpy.ndarray
            Waveform value(s) at specified time(s).
        """
        return self.evaluate(t)

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the waveform at given time(s).

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s) in seconds.

        Returns
        -------
        float or numpy.ndarray
            Waveform value(s) at specified time(s).
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class ContinuousWave(Waveform):
    """
    Continuous sine wave at a fixed frequency.

    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    amplitude : float, optional
        Peak amplitude of the waveform, default=1.0.
    phase : float, optional
        Phase offset in radians, default=0.0.
    """

    def __init__(self, frequency: float, amplitude: float = 1.0, phase: float = 0.0):
        super().__init__(amplitude, phase)
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the continuous wave at given time(s).

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s) in seconds.

        Returns
        -------
        float or numpy.ndarray
            Waveform value(s) at specified time(s).
        """
        return self.amplitude * np.sin(self.omega * t + self.phase)


class GaussianPulse(Waveform):
    """
    Gaussian pulse waveform.

    Parameters
    ----------
    frequency : float
        Center frequency in Hz.
    pulse_width : float
        Width of the Gaussian pulse in seconds.
    amplitude : float, optional
        Peak amplitude of the waveform, default=1.0.
    phase : float, optional
        Phase offset in radians, default=0.0.
    delay : float, optional
        Time delay for the pulse peak in seconds, default=3*pulse_width.
    """

    def __init__(
        self,
        frequency: float,
        pulse_width: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        delay: Optional[float] = None,
    ):
        super().__init__(amplitude, phase)
        self.frequency = frequency
        self.pulse_width = pulse_width
        self.omega = 2 * np.pi * frequency

        # Set default delay to 3 * pulse_width to ensure a clean start
        self.delay = delay if delay is not None else 3 * pulse_width

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the Gaussian pulse at given time(s).

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s) in seconds.

        Returns
        -------
        float or numpy.ndarray
            Waveform value(s) at specified time(s).
        """
        # Calculate the Gaussian envelope
        tau = (t - self.delay) / self.pulse_width
        envelope = np.exp(-0.5 * tau * tau)

        # Modulate the envelope with a sine wave
        return self.amplitude * envelope * np.sin(self.omega * t + self.phase)


class RickerWavelet(Waveform):
    """
    Ricker wavelet (Mexican hat) waveform.

    Parameters
    ----------
    frequency : float
        Peak frequency in Hz.
    amplitude : float, optional
        Peak amplitude of the waveform, default=1.0.
    delay : float, optional
        Time delay for the wavelet peak in seconds, default=1.5/frequency.
    """

    def __init__(
        self, frequency: float, amplitude: float = 1.0, delay: Optional[float] = None
    ):
        super().__init__(amplitude, 0.0)  # Ricker wavelet doesn't use phase
        self.frequency = frequency
        self.delay = delay if delay is not None else 1.5 / frequency

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the Ricker wavelet at given time(s).

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s) in seconds.

        Returns
        -------
        float or numpy.ndarray
            Waveform value(s) at specified time(s).
        """
        # Calculate the Ricker wavelet (second derivative of a Gaussian)
        tau = np.pi * self.frequency * (t - self.delay)
        tau_squared = tau * tau
        return self.amplitude * (1.0 - 2.0 * tau_squared) * np.exp(-tau_squared)


class CustomWaveform(Waveform):
    """
    Custom waveform defined by a user-provided function.

    Parameters
    ----------
    waveform_func : Callable[[float], float]
        Function taking time as input and returning waveform amplitude.
    amplitude : float, optional
        Scaling factor for the waveform, default=1.0.
    """

    def __init__(self, waveform_func: Callable[[float], float], amplitude: float = 1.0):
        super().__init__(amplitude, 0.0)  # Phase handled by the function
        self.waveform_func = waveform_func

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the custom waveform at given time(s).

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s) in seconds.

        Returns
        -------
        float or numpy.ndarray
            Waveform value(s) at specified time(s).
        """
        # Apply the user's function to the time(s)
        if isinstance(t, np.ndarray):
            return self.amplitude * np.array([self.waveform_func(ti) for ti in t])
        return self.amplitude * self.waveform_func(t)
