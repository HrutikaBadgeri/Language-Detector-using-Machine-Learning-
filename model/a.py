import pickle

a = pickle.load(open('LanguageDetection_model.pkl', 'rb'))
cv = pickle.load(open('CountVectoriser.pkl', 'rb'))
le = pickle.load(open('LabelEncoder.pkl', 'rb'))

# function for predicting language
def predict(text):
    x = cv.transform([text])
    lang = a.predict(x)
    lang = le.inverse_transform(lang)
    print("The langauge is in",lang[0])

predict(" provides a community based knowledge portal for Analytics and Data Science professionals")
predict(" fournit un portail de connaissances basé sur la communauté pour les professionnels de l'analyse et de la science des données")
predict("توفر  بوابة معرفية قائمة على المجتمع لمحترفي التحليلات وعلوم البيانات")
predict(" proporciona un portal de conocimiento basado en la comunidad para profesionales de Analytics y Data Science.")
predict("അനലിറ്റിക്സ്, ഡാറ്റാ സയൻസ് പ്രൊഫഷണലുകൾക്കായി കമ്മ്യൂണിറ്റി അധിഷ്ഠിത വിജ്ഞാന പോർട്ടൽ അനലിറ്റിക്സ് വിദ്യ നൽകുന്നു")
predict(" - это портал знаний на базе сообщества для профессионалов в области аналитики и данных.")