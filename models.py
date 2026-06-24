#image models and target models
TARGET_MODELS= [
    "openai/gpt-4o",
    "google/gemini-3-pro-image"
    #we will add more models here
]

JUDGE_MODEL = [
    "google/gemini-2.5-flash-image" 
]
#WE WILL CHANGE THE JUDGE TO A MORE CAPABLE MODEL, PREFERABLY CLAUDE OPUS. 
# FOR THIS TRIAL RUN THE JUDGE MODEL IS NOT USED. WE JUST USE AN IF STATEMENT TO CHECK IF AN IMNAGE WAS PRODUCED.