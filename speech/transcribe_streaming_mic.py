#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:

    pip install pyaudio

Example usage:
    python transcribe_streaming_mic.py
"""

# [START speech_transcribe_streaming_mic]
from __future__ import division

import re
import sys
import pyaudio
import argparse
import pyttsx

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from six.moves import queue

# NLP Imports
from google.cloud import language
from google.cloud.language import enums as lang_enums
from google.cloud.language import types as lang_types

from google.cloud import translate


# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# Terminal colours
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        if sentence.sentiment.score < 0:
            print('- {}Sentence {} has a sentiment score of {}{}'.format(
                bcolors.FAIL, index, sentence_sentiment, bcolors.ENDC))
        elif sentence.sentiment.score == 0:
            print('- {}Sentence {} has a sentiment score of {}{}'.format(
                bcolors.WARNING, index, sentence_sentiment, bcolors.ENDC))
        else:
            print('- {}Sentence {} has a sentiment score of {}{}'.format(
                bcolors.OKGREEN, index, sentence_sentiment, bcolors.ENDC))

    print('- {}Overall Sentiment: score of {} with magnitude of {}{}'.format(
        bcolors.UNDERLINE,score, magnitude, bcolors.ENDC))
    return 0

def print_entities(entities):

    entity_type = ('UNKNOWN','PERSON','LOCATION','ORGANIZATION','EVENT','WORK_OF_ART','CONSUMER_GOOD','OTHER')

    for entity in entities:
        print('{}-- name:{}\n-- type:{}\n-- salience:{}{}'.format(
            bcolors.OKBLUE,entity.name,entity_type[entity.type],entity.salience,bcolors.ENDC))

def translation(text):
    translate_client = translate.Client()
    target = 'es'

    input=text
    translation = translate_client.translate(
        input,
        target_language=target)

    print('- ' + bcolors.HEADER + 'translation:"' + translation['translatedText'] + '"' + bcolors.ENDC)
    #print(u'English: {}'.format(text))
    #print(u'Spanish: {}'.format(translation['translatedText']))

def speak(text):
    engine = pyttsx.init()
    engine.say(text)
    engine.runAndWait()

def nlp_analyze(text):
    """Run a sentiment analysis request on text within a passed filename."""
    nlp_client = language.LanguageServiceClient()

    document = lang_types.Document(
        content=text,
        type=lang_enums.Document.Type.PLAIN_TEXT)

    annotations = nlp_client.analyze_sentiment(document=document)

    entities = nlp_client.analyze_entities(document=document).entities

    # Print the results
    print_result(annotations)
    print_entities(entities)

def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(bcolors.BOLD + '"' + transcript + overwrite_chars + '"' + bcolors.ENDC)
            nlp_analyze(transcript + overwrite_chars)
            translation(transcript + overwrite_chars)
            #speak(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0


def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'en-US'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses)


if __name__ == '__main__':
    main()
# [END speech_transcribe_streaming_mic]
