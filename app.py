import streamlit as st
import pandas as pd
import joblib
import pandas as pd
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# Importing model
model1 = joblib.load('Dec_mdl.pkl')
model2 = joblib.load('Log_mdl.pkl')
model3 = joblib.load('XGB_mdl.pkl')

# Loading of dataset
df = pd.read_csv("fraudTest.csv")

# Getting inputs from user

cate = pd.DataFrame(df["category"].unique(), columns=['category'])
cate = cate.sort_values(by=['category'])
cate_list = [item for sublist in cate.values.tolist() for item in sublist]
cate_dict = {value: index for index, value in enumerate(cate_list)}

job_ = pd.DataFrame(df["job"].unique(), columns=['job'])
job_ = job_.sort_values(by=['job'])
job_list = [item for sublist in job_.values.tolist() for item in sublist]
job_dict = {value: index for index, value in enumerate(job_list)}

st.title("DETECTING ANOMALIES IN FINANCIAL TRANSACTIONS")

trans_date = st.date_input("Enter date of transaction : ", value="default_value_today")
trans_time = st.time_input("Enter time of transaction : ", value=None)
trans_date_trans_time = str(trans_date) + " " + str(trans_time)

cc_num = st.number_input("Enter credit card number : ", value="min")
merchant_name = st.text_input("Enter merchant name : ")

category = st.selectbox("Enter category : ",
                         ('personal_care', 'health_fitness', 'misc_pos', 'travel',
        'kids_pets', 'shopping_pos', 'food_dining', 'home',
        'entertainment', 'shopping_net', 'misc_net', 'grocery_pos',
        'gas_transport', 'grocery_net'),
        index=0,
        placeholder="Select category..."
)
amt = st.number_input("Enter amount : ", value="min")

first = st.text_input("Enter First Name of Credit Card Holder : ")
last = st.text_input("Enter Last Name of Credit Card Holder : ")

gender = st.selectbox("Enter Gender of Credit Card Holder : ",
                      ('Male', 'Female', 'Others'),
                      index=0,
                      placeholder="Select gender..."
)

street = st.text_input("Enter Street Address : ")
city = st.text_input("Enter City : ")
state = st.text_input(" Enter State : ")
zip = st.number_input("Enter Zip : ", value="min")
lat = st.number_input("Enter Latitude Location : ", value="min")
long = st.number_input("Enter Longitude Location : ", value="min")

city_pop = st.number_input("Enter Your City Population : ", value="min")
job = st.selectbox("Enter Your Job : ",
                   ('Mechanical engineer', 'Sales professional, IT',
       'Librarian, public', 'Set designer', 'Furniture designer',
       'Psychotherapist', 'Therapist, occupational',
       'Development worker, international aid', 'Advice worker',
       'Barrister', 'Pensions consultant', 'Freight forwarder',
       'Paramedic', 'Building surveyor', 'Materials engineer',
       'Commercial horticulturist', 'Town planner',
       'Radiographer, therapeutic', 'Surveyor, rural practice',
       'Energy manager', 'Prison officer',
       'Museum/gallery exhibitions officer', 'Media planner',
       'Seismic interpreter', 'Learning disability nurse',
       'Buyer, industrial', 'Scientific laboratory technician',
       'Counselling psychologist', 'Scientist, biomedical',
       'Audiological scientist', 'Stage manager',
       'Leisure centre manager', 'Community pharmacist', 'Risk analyst',
       'Clinical research associate', 'Librarian, academic',
       'Editor, magazine features', 'Ceramics designer', 'Musician',
       'Designer, furniture', 'Exhibition designer',
       'Engineer, automotive', 'Film/video editor', 'Web designer',
       'Environmental consultant', 'Production assistant, television',
       'Education officer, community',
       'Senior tax professional/tax inspector', 'Investment analyst',
       'Loss adjuster, chartered', 'Scientist, audiological',
       'Tax inspector', 'Primary school teacher',
       'Agricultural consultant', 'Geochemist', 'Geneticist, molecular',
       'Medical sales representative', 'Petroleum engineer',
       'Advertising account planner', 'Travel agency manager',
       'Music therapist', 'Neurosurgeon', 'Further education lecturer',
       'Therapist, sports', 'Operational researcher', 'Site engineer',
       'Communications engineer', 'Child psychotherapist',
       'Engineer, land', 'Dealer', 'Amenity horticulturist',
       'Electronics engineer', 'Research officer, political party',
       'Accounting technician', 'Licensed conveyancer',
       'Programmer, multimedia', 'Chief Financial Officer',
       'Health promotion specialist', 'Field seismologist', 'Chiropodist',
       'Special effects artist', 'Designer, ceramics/pottery',
       'Systems developer', 'Biomedical scientist',
       'Geophysicist/field seismologist', 'Electrical engineer',
       'Equities trader', 'Energy engineer', 'Police officer',
       'Music tutor', "Nurse, children's", 'Jewellery designer',
       'Therapist, art', 'Archivist', 'Production manager',
       'Farm manager', 'Editor, film/video', 'Engineer, production',
       'Journalist, newspaper', 'Drilling engineer',
       'Psychotherapist, child',
       'Programme researcher, broadcasting/film/video', 'Contractor',
       'Sub', 'Broadcast engineer', 'Animal nutritionist',
       'Intelligence analyst', 'Engineer, electronics',
       'Historic buildings inspector/conservation officer', 'Optometrist',
       'Firefighter', 'Copywriter, advertising', 'Designer, jewellery',
       'Scientist, physiological', 'Financial adviser',
       'Early years teacher', 'Insurance broker', 'Broadcast presenter',
       'Clothing/textile technologist', 'Probation officer',
       'Forensic psychologist', 'Telecommunications researcher',
       'Teacher, early years/pre',
       'Conservation officer, historic buildings',
       'Educational psychologist', 'Rural practice surveyor',
       'Patent attorney', 'Futures trader', 'Hospital doctor',
       'Chief Executive Officer', 'Chartered public finance accountant',
       'Charity fundraiser', 'Information systems manager',
       'Environmental manager', 'Designer, exhibition/display',
       'Retail banker', 'English as a second language teacher',
       'Building services engineer', 'Buyer, retail',
       'Surveyor, land/geomatics', 'Conservator, furniture',
       'Exhibitions officer, museum/gallery',
       'Accountant, chartered public finance', 'Secretary/administrator',
       'Production assistant, radio', 'Water engineer',
       'Insurance risk surveyor', 'Optician, dispensing',
       'Fisheries officer', 'Social research officer, government',
       'Curator', 'Applications developer', 'Chartered accountant',
       'Comptroller', 'Land', 'Control and instrumentation engineer',
       'Community education officer', 'Osteopath', 'Surgeon',
       'Sales executive', 'Lecturer, higher education',
       'Public relations account executive', 'Transport planner',
       'Investment banker, corporate', 'Camera operator',
       'Land/geomatics surveyor', 'Engineer, mining',
       'Armed forces training and education officer',
       'Manufacturing engineer', 'Technical brewer', 'Warden/ranger',
       'Radio broadcast assistant', 'Trade mark attorney', 'Metallurgist',
       'Heritage manager', 'Podiatrist', 'Counsellor', 'Gaffer',
       'Exercise physiologist', 'Engineer, technical sales',
       'Toxicologist', 'Magazine features editor', 'Lexicographer',
       'Editor, commissioning', 'Phytotherapist',
       'Careers information officer', 'Tax adviser',
       'Advertising copywriter', 'Occupational hygienist',
       'Accountant, chartered certified', 'Press photographer',
       'Wellsite geologist', 'Animator', 'Secondary school teacher',
       'Race relations officer', 'Tourism officer', 'Biomedical engineer',
       'Lecturer, further education', 'Public house manager',
       'Administrator', 'Minerals surveyor',
       'Scientist, clinical (histocompatibility and immunogenetics)',
       'Administrator, education', 'Market researcher',
       'Chief Operating Officer', 'Media buyer',
       'Armed forces logistics/support/administrative officer',
       'Civil engineer, contracting', 'Network engineer',
       'Engineer, drilling', 'Multimedia programmer',
       'Scientist, research (physical sciences)',
       'Special educational needs teacher', 'Geoscientist',
       'Psychologist, forensic', 'Horticulturist, commercial',
       'Furniture conservator/restorer', 'Chartered loss adjuster',
       'Tree surgeon', 'Scientist, marine',
       'Product/process development scientist', 'Financial trader',
       'Radio producer', 'Mining engineer',
       'Training and development officer', 'Chief Strategy Officer',
       'Theme park manager', 'Physiological scientist', 'Pathologist',
       'Art therapist', 'Teacher, special educational needs',
       'Television floor manager', 'Retail merchandiser',
       'Corporate investment banker', 'Pilot, airline', 'Statistician',
       'Public librarian', 'Insurance underwriter',
       'Teacher, English as a foreign language', 'Physiotherapist',
       'Teacher, adult education', 'Operations geologist',
       'Environmental education officer', 'IT trainer',
       'Research officer, trade union', 'Animal technologist',
       'Air cabin crew', 'Designer, industrial/product',
       'Engineer, civil (contracting)', "Politician's assistant",
       'Conservator, museum/gallery', 'Soil scientist', 'Arboriculturist',
       'Sports development officer', 'Television/film/video producer',
       "Barrister's clerk", 'Engineer, communications', 'Copy',
       'Water quality scientist', 'Immunologist',
       'Database administrator', 'Producer, radio', 'Cartographer',
       'Health physicist', 'Private music teacher', 'Social researcher',
       'Pension scheme manager', 'Psychologist, clinical',
       'Building control surveyor', 'Designer, textile', 'Barista',
       'Claims inspector/assessor', 'Systems analyst',
       'Structural engineer', 'Archaeologist',
       'Psychologist, counselling', 'Cytogeneticist', 'Health visitor',
       'Emergency planning/management officer',
       'Scientist, research (maths)', 'Herbalist',
       'Tourist information centre manager', 'Landscape architect',
       'Paediatric nurse', 'Video editor', 'Quantity surveyor',
       'Geologist, engineering', 'Therapist, drama', 'Doctor, hospital',
       'Hydrographic surveyor', 'Glass blower/designer',
       'Therapist, horticultural', 'Make', 'Physicist, medical',
       'Community arts worker', 'Embryologist, clinical', 'Bookseller',
       'Facilities manager', 'Theatre manager',
       'Dance movement psychotherapist', 'Scientist, research (medical)',
       'Commercial/residential surveyor',
       'Administrator, local government', 'Development worker, community',
       'Higher education careers adviser', 'Location manager',
       'General practice doctor', 'Art gallery manager',
       'Pharmacist, hospital', 'Teaching laboratory technician',
       'Chemical engineer', 'Geologist, wellsite',
       'Presenter, broadcasting', 'Community development worker',
       'Immigration officer', 'Chief Technology Officer',
       'Engineer, biomedical', 'Logistics and distribution manager',
       'Academic librarian', 'Commissioning editor',
       'Research scientist (physical sciences)',
       'Teacher, secondary school', 'Engineer, agricultural',
       'Naval architect', 'Clinical biochemist',
       'Restaurant manager, fast food', 'Data scientist',
       'Theatre director', 'Radiographer, diagnostic', 'Interpreter',
       'Psychiatric nurse', 'Research scientist (life sciences)',
       'Learning mentor', 'Lawyer',
       'Outdoor activities/education manager', 'Human resources officer',
       'Trading standards officer', 'Press sub',
       'Regulatory affairs officer', 'Pharmacist, community',
       'Engineering geologist', 'Engineer, control and instrumentation',
       'Horticultural therapist', 'Maintenance engineer',
       'Producer, television/film/video', 'Analytical chemist',
       'Surveyor, minerals', 'Hydrologist',
       'Manufacturing systems engineer', 'Dispensing optician',
       'Product designer', 'Architect', 'Education officer, museum',
       'Company secretary', 'Television production assistant',
       'Science writer', 'Quarry manager', 'Fitness centre manager',
       'Mudlogger', 'Medical physicist', 'Waste management officer',
       'Occupational psychologist', 'Colour technologist', 'Fine artist',
       'Call centre manager', 'Television camera operator',
       'Sport and exercise psychologist', 'Health service manager',
       'Interior and spatial designer', 'Nutritional therapist',
       'Clinical psychologist', 'Programmer, applications',
       'Retail manager', 'Engineer, materials', 'Diagnostic radiographer',
       'Designer, interior/spatial', 'Herpetologist',
       'Museum/gallery conservator', 'Nurse, mental health',
       'Public affairs consultant', 'Management consultant',
       'Industrial/product designer', 'Airline pilot', 'Retail buyer',
       'Psychologist, sport and exercise', 'Ambulance person',
       'Equality and diversity officer',
       'Engineer, broadcasting (operations)', 'Plant breeder/geneticist',
       'Engineer, maintenance', 'Chemist, analytical',
       'Chief Marketing Officer', 'Visual merchandiser',
       'Planning and development surveyor', 'Illustrator',
       'Merchandiser, retail', 'Psychiatrist', 'Mental health nurse',
       'Orthoptist', 'Magazine journalist', 'Doctor, general practice',
       'Product manager', 'Advertising account executive',
       'Surveyor, mining', 'Pharmacologist', 'Medical secretary',
       'Aid worker', 'Cabin crew', 'Nature conservation officer',
       'Public relations officer', 'Insurance claims handler',
       'Investment banker, operational',
       'Administrator, charities/voluntary organisations',
       'Oceanographer', 'Therapist, music', 'Local government officer',
       'Arts development officer', 'Biochemist, clinical',
       'Civil Service administrator', 'Clinical cytogeneticist', 'Writer',
       'Garment/textile technologist', 'Research scientist (maths)',
       'Museum education officer', 'Teacher, primary school',
       'Designer, multimedia', 'Textile designer',
       'Civil Service fast streamer', 'Environmental health practitioner',
       'Health and safety adviser', 'Designer, television/film set',
       'Associate Professor',
       'Chartered legal executive (England and Wales)',
       'Purchasing manager', 'Surveyor, hydrographic',
       'Hospital pharmacist', 'Research scientist (medical)',
       'Engineer, structural', 'Field trials officer',
       'Engineer, building services', 'Acupuncturist', 'Chief of Staff',
       'Records manager', 'Catering manager', 'Event organiser',
       'Engineer, petroleum', 'Production engineer',
       'Education administrator', 'IT consultant',
       'Horticultural consultant', 'Ecologist', 'Engineer, aeronautical',
       'Volunteer coordinator', 'Air broker',
       'Engineer, civil (consulting)', 'Estate manager/land agent',
       'Aeronautical engineer', 'Engineer, manufacturing',
       'Architectural technologist', 'Marketing executive',
       'Hotel manager', 'Tour manager', 'Professor Emeritus',
       'Oncologist', 'TEFL teacher', 'Economist',
       'English as a foreign language teacher', 'Hydrogeologist',
       'Medical technical officer', 'Charity officer',
       'Administrator, arts', 'Occupational therapist',
       'Solicitor, Scotland', 'Sports administrator', 'Artist',
       'Engineer, water', 'Operational investment banker',
       'Software engineer'),
       index=0,
       placeholder="Select job...",
)

dob = st.date_input("Enter Your Date of Birth : ", value="default_value_today")
trans = st.text_input("Enter Transaction ID : ")
unix_time = st.number_input("Enter UNIX Time of transaction : ", value="min")
merch_lat = st.number_input("Enter Latitude Location of Merchant : ", value="min")
merch_long = st.number_input("Enter Longitude Location of Merchant : ", value="min")

for i in job_dict.keys():
    if job == i:
        job = job_dict[i]

for i in cate_dict.keys():
    if category == i:
        category = cate_dict[i]

# Prediction using model
prediction_dec = model1.predict([[category, amt, lat, long, city_pop, job, unix_time]])
prediction_log = model2.predict([[category, amt, lat, long, city_pop, job, unix_time]])
prediction_xgb = model3.predict([[category, amt, lat, long, city_pop, job, unix_time]])

# Majority voting
prediction_list = [int(prediction_log), int(prediction_dec), int(prediction_xgb)]
counter = Counter(prediction_list)
most_common_prediction = counter.most_common(1)[0][0]

# Printing the result
if (st.button("Submit", type="primary")):
    st.markdown('## Prediction')
    if most_common_prediction == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
