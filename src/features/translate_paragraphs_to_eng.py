# Importa pandas library for inmporting CSV
import pandas as pd
from google.cloud import translate

# Instantiates a client
translate_client = translate.Client()

#Translating the text to specified target language
def translate(word):
    # Target language
    target_language = 'en' #Add here the target language that you want to translate to
    # Translates some text into Russian
    translation = translate_client.translate(
        word,
        target_language=target_language)

    return (translation['translatedText'])

#Import data from CSV
def importCSV():
    data = pd.read_csv('/data/processed/eu_links_lang.csv', index_col = 0)
    text_to = data[data.LANG_TO != 'en'].TEXT_TO
    text_from = data[data.LANG_FROM != 'en'].TEXT_FROM
    countRows = (len(data))

    #Create a dictionary with translated words
    translatedCSV = { "TEXT_TO":[], "TEXT_FROM":[]} #Change the column names accordingly to your coumns names
 
    #Translated word one by one from the CSV file and save them to the dictionary
    for index, row in data.iterrows():
        translatedCSV["TEXT_TO"].append(translate(row["TEXT_TO"]))
        translatedCSV["TEXT_FROM"].append(translate(row["TEXT_FROM"]))

    #Create a Dataframe from Dictionary 
    #Save the DataFrame to a CSV file
    df = pd.DataFrame(data=translatedCSV)
    df.to_csv("translatedCSV.csv", sep='\t')
    

#Call the function
importCSV()