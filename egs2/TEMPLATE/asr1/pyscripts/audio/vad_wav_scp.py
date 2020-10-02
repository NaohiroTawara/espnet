#!/usr/bin/env python3

"""
Adapted from https://github.com/wiseman/py-webrtcvad
(MIT License)
"""

import collections
import argparse
from espnet.utils.cli_utils import get_commandline_args
from io import BytesIO
from pathlib import Path
import wave
import webrtcvad
import _webrtcvad
import logging
from tqdm import tqdm
import kaldiio


class Vad(object):
    def __init__(self, mode=None):
        self._vad = _webrtcvad.create()
        _webrtcvad.init(self._vad)
        if mode is not None:
            self.set_mode(mode)

    def set_mode(self, mode):
        _webrtcvad.set_mode(self._vad, mode)

    def is_speech(self, buf, sample_rate, length=None):
        length = length or int(len(buf) / 2)
        if length * 2 > len(buf):
            raise IndexError(
                'buffer has %s frames, but length argument was %s' % (
                    int(len(buf) / 2.0), length))
        return _webrtcvad.process(self._vad, sample_rate, buf, length)


def valid_rate_and_frame_length(rate, frame_length):
    return _webrtcvad.valid_rate_and_frame_length(rate, frame_length)



class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        #sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                start_time  = ring_buffer[0][0].timestamp
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            #voiced_frames.append(i)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                #sys.stdout.write('-(%s)' % ())
                end_time = frame.timestamp + frame.duration
                triggered = False
                yield start_time, end_time
                ring_buffer.clear()
                start_time = None
    if triggered:
        end_time = frame.timestamp + frame.duration
    #sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if start_time:
        yield start_time, end_time


def read_wave(wavpath):
    if wavpath.endswith("|"):
        # Streaming input e.g. cat a.wav |
        with kaldiio.open_like_kaldi(wavpath, "rb") as f:
            wf = wave.open(BytesIO(f.read()))
    else:
        wf = wave.open(wavpath, 'rb')
    num_channels = wf.getnchannels()
    assert num_channels == 1, "Only 1-ch is supported"
    sample_width = wf.getsampwidth()
    assert sample_width == 2, "Only 1-byte wave is supported"
    sample_rate = wf.getframerate()
    assert sample_rate in (8000, 16000, 32000, 48000), "Sample rate must be 8k, 16k, 32k, or 48k"
    pcm_data = wf.readframes(wf.getnframes())
    wf.close()
    return pcm_data, sample_rate


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='Apply VAD to waves list from "wav.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp")
    parser.add_argument("out_filename")
    parser.add_argument("--aggressiveness", type=int, default=3, help="Aggressiveness (0-3. 3 is the most aggressiveness)")
    parser.add_argument("--min_sec", type=float, default=0, help="minimum length (sec)")
    parser.add_argument("--audio-format", default="wav")
    args = parser.parse_args()

    with Path(args.scp).open("r") as fscp:
        with Path(args.out_filename).open('w') as out:
            for line in tqdm(fscp):
                uttid, wavpath = line.strip().split(None, 1)

                audio, sample_rate = read_wave(wavpath)
                vad = webrtcvad.Vad(args.aggressiveness)
                min_sec = args.min_sec
                frames = frame_generator(30, audio, sample_rate)
                frames = list(frames)
                segments = vad_collector(sample_rate, 30, 300, vad, frames)
                for i, (st, et) in enumerate(segments):
                    length =  et - st
                    if length > min_sec:
                        out.write('{}_{:06}-{:06} {} {:.02f} {:.02f}\n'.format(
                            uttid, int(st*100), int(et * 100), uttid, st, et))


if __name__ == '__main__':
    main()