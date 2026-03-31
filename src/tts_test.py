import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)

engine.say("Text to speech test. If you hear this, it works.")
engine.runAndWait()
