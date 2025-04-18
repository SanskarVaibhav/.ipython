import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import cv2
import pyautogui
import time
import smtplib
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import warnings
from sympy import sympify

warnings.filterwarnings("ignore")
# Initialize text-to-speech engine

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

listener = sr.Recognizer()

def talk(text):
    print("Alexa:", text)
    engine.say(text)
    engine.runAndWait()


def greet_user():
    current_hour = datetime.datetime.now().hour
    if current_hour < 12:
        talk("Good morning!")
    elif 12 <= current_hour < 18:
        talk("Good afternoon!")
    elif 18 <= current_hour < 21:
        talk("Good evening!")
    else:
        talk("Good night!")
    talk("How can I assist you today?")


def take_command():
    try:
        with sr.Microphone() as source:
            print("Listening...")
            listener.adjust_for_ambient_noise(source)
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'alexa' in command:
                command = command.replace('alexa', '').strip()
            print("You:", command)
            return command
    except sr.UnknownValueError:
        talk("Sorry, I didn't catch that. Please repeat.")
        return ""
    except sr.RequestError:
        talk("Sorry, I can't reach the speech recognition service right now.")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        talk("An error occurred. Please try again.")
        return ""
    

def take_photo():
    talk("Taking a photo. Smile!")
    cam = cv2.VideoCapture(0)
    time.sleep(2)  # Wait for camera to adjust
    ret, frame = cam.read()
    if ret:
        cv2.imwrite("photo.jpg", frame)
        talk("Photo saved as photo.jpg")
    else:
        talk("Failed to take photo.")
    cam.release()
    cv2.destroyAllWindows()


def take_screenshot():
    talk("Taking a screenshot.")
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    talk("Screenshot saved as screenshot.png")

def calculate_expression(expression):
    try:
        result = sympify(expression)
        talk(f"The result is {result}")
    except Exception:
        talk("Sorry, I couldn't calculate that.")


def run_alexa():
    while True:
        command = take_command()

        if not command:
            continue

        if 'play' in command:
            song = command.replace('play', '').strip()
            talk(f"Playing {song}")
            pywhatkit.playonyt(song)

        elif 'stop music' in command:
            talk("Stopping music.")
            os.system("taskkill /f /im vlc.exe")
            os.system("taskkill /f /im itunes.exe")
            os.system("taskkill /f /im spotify.exe")
            os.system("taskkill /f /im music.exe")
            os.system("taskkill /f /im windows_media_player.exe")
            os.system("taskkill /f /im groove.exe")
            os.system("taskkill /f /im foobar2000.exe")
            os.system("taskkill /f /im winamp.exe")     


        elif 'pause music' in command:
            talk("Pausing music.")
            os.system("taskkill /f /im vlc.exe")
            os.system("taskkill /f /im itunes.exe")
            os.system("taskkill /f /im spotify.exe")
            os.system("taskkill /f /im music.exe")
            os.system("taskkill /f /im windows_media_player.exe")
            os.system("taskkill /f /im groove.exe")
            os.system("taskkill /f /im foobar2000.exe")
            os.system("taskkill /f /im winamp.exe")


        elif 'resume music' in command:
            talk("Resuming music.")
            os.system("start vlc.exe")
            os.system("start itunes.exe")
            os.system("start spotify.exe")
            os.system("start music.exe")
            os.system("start windows_media_player.exe")
            os.system("start groove.exe")
            os.system("start foobar2000.exe")
            os.system("start winamp.exe")


        elif 'open' in command:
            app = command.replace('open', '').strip()
            talk(f"Opening {app}")
            pywhatkit.search(app)



        elif 'date' in command:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            talk(f"Today's date is {today}")    


        elif 'day' in command:
            day = datetime.datetime.now().strftime('%A')
            talk(f"Today is {day}")

        elif 'weather' in command:
            talk("Please tell me the city name.")
            city = take_command()
            if city:
                talk(f"Fetching weather information for {city}.")
                pywhatkit.search(f"weather in {city}")
            else:
                talk("I didn't catch the city name.")       


        elif 'news' in command:
            talk("Fetching the latest news.")
            pywhatkit.search("latest news")


        elif 'time' in command:
            time_now = datetime.datetime.now().strftime('%I:%M %p')
            talk(f"The current time is {time_now}")


        elif 'reminder' in command:
            talk("What would you like to be reminded about?")
            reminder = take_command()
            if reminder:
                talk(f"Setting a reminder for {reminder}.")
                pywhatkit.search(f"reminder for {reminder}")
            else:
                talk("I didn't catch that.")


        elif 'alarm' in command:
            talk("What time should I set the alarm for?")
            alarm_time = take_command()
            if alarm_time:
                talk(f"Setting an alarm for {alarm_time}.")
                pywhatkit.search(f"alarm for {alarm_time}")
            else:
                talk("I didn't catch the time.")


        elif 'wikipedia' in command:
            talk("What topic would you like to search on Wikipedia?")
            topic = take_command()
            if topic:
                talk(f"Searching Wikipedia for {topic}.")
                try:
                    info = wikipedia.summary(topic, sentences=2)
                    talk(info)
                except wikipedia.exceptions.DisambiguationError as e:
                    talk(f"Too many results. Try being more specific like {e.options[0]}")
                except wikipedia.exceptions.PageError:
                    talk("I couldn't find anything on Wikipedia.")
                except Exception:
                    talk("Something went wrong while searching Wikipedia.")
            else:
                talk("I didn't catch the topic.")


        elif 'define' in command:
            term = command.replace('define', '').strip()
            talk(f"Defining {term}.")
            try:
                definition = wikipedia.summary(term, sentences=1)
                talk(definition)
            except wikipedia.exceptions.DisambiguationError as e:
                talk(f"Too many results. Try being more specific like {e.options[0]}")
            except wikipedia.exceptions.PageError:
                talk("I couldn't find anything on Wikipedia.")
            except Exception:
                talk("Something went wrong while searching Wikipedia.")


        elif 'calculate' in command:
            command = command.replace('calculate', '').strip()
            talk(f"Calculating {command}.")
            try:
                result = eval(command)
                talk(f"The result is {result}")
            except Exception:
                talk("I couldn't calculate that. Please try again.")

        elif 'translate' in command:
            talk("What would you like to translate?")
            text_to_translate = take_command()
            if text_to_translate:
                talk(f"Translating {text_to_translate}.")
                pywhatkit.search(f"translate {text_to_translate}")
            else:
                talk("I didn't catch that.")


        elif 'who is' in command or 'what is' in command:
            try:
                info = wikipedia.summary(command, sentences=2)
                talk(info)
            except wikipedia.exceptions.DisambiguationError as e:
                talk(f"Too many results. Try being more specific like {e.options[0]}")
            except wikipedia.exceptions.PageError:
                talk("I couldn't find anything on Wikipedia.")
            except Exception:
                talk("Something went wrong while searching Wikipedia.")


        elif 'joke' in command:
            joke = pyjokes.get_joke()
            talk(joke)


        elif 'email' in command:
            talk("Please tell me the recipient's email address.")
            recipient = take_command()
            if recipient:
                talk("What would you like to say?")
                message = take_command()
                if message:
                    talk(f"Sending email to {recipient}.")
                    # Here you would integrate with an email service to send the email
                    # For example, using smtplib or any other email library
                    talk("Email sent successfully!")
                else:
                    talk("I didn't catch your message.")
            else:
                talk("I didn't catch the recipient's email address.")

                
        elif 'take photo' in command or 'click photo' in command:
            take_photo()

        elif 'screenshot' in command:
            take_screenshot()


        elif 'search' in command:
            search_term = command.replace('search', '').strip()
            talk(f"Searching for {search_term}")
            pywhatkit.search(search_term)



        elif 'play game' in command:
            talk("Launching a simple game.")
            # Here you can integrate a simple game or a link to a game
            pywhatkit.search("simple games to play")

        elif 'open youtube' in command:
            talk("Opening YouTube.")
            pywhatkit.search("YouTube")





        elif 'open google' in command:
            talk("Opening Google.")
            pywhatkit.search("Google")


        elif 'open facebook' in command:
            talk("Opening Facebook.")
            pywhatkit.search("Facebook")



        elif 'stop' in command or 'exit' in command or 'bye' in command:
            talk("Goodbye! Have a great day.")
            break

        else:
            talk("I didn't understand that. Please say it again.")

# Greet the user
talk("Hello! I'm Alexa. How can I help you?")

# Start listening
run_alexa()
