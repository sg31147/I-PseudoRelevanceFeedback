import logging
import requests
import streamlit as st

# Configure logging
LOGGER = logging.getLogger(name='test')
LOGGER.setLevel(logging.INFO)

# Function to fetch data from the API
def fetch_icd_mapping():
    url = "http://localhost:8081/mapping"  # Replace with your actual API endpoint
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse the JSON response
        icd_data = response.json()["data"]
        
        # Combine keys and values into the desired format
        icd_mapping = [f"{icd}" for icd, definition in icd_data.items()]
        
        return icd_mapping
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return []  # Return an empty list in case of an error

# Set up the Streamlit app
st.set_page_config(
    page_title="Utilizing Pseudo-Relevance feedback for automated ICD-10 Medical Coding",
    layout="wide"  # Makes the layout wider
)

st.markdown(
    """
    <style>

        /* Increase font size for text area */
        textarea {
            font-size: 20px !important;
        }
        li {
            font-size: 20px !important;
        }

        /* Adjust sidebar width */
        [data-testid="stSidebar"] {
            min-width: 0px; /* Adjust as needed */
            max-width: 350px;
            background-color: #f8f9fa; /* Light gray background */
        }


        /* Optimize sidebar text */
        [data-testid="stSidebar"] * {
        
            font-size: 18px !important;  /* Increase font size */
            # font-weight: bold; /* Make text bold */
            color: #333333; /* Dark gray text */
         
        }



        /* Optimize sidebar headers */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            font-size: 20px !important;
            color: #000000; /* Blue color */
            text-align: center;
            margin-bottom: 10px;
        }

        .custom-button {{
            background-color: {theme_color}; 
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            text-align: center;
            display: inline-block;
            width: 100%;
        }}

        .custom-button:hover {{
            background-color: #548CFF; /* Slightly lighter on hover */
        }}

       
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    "<h1 style='font-size:35px;text-align: center; color: #004085;'>Utilizing Pseudo-Relevance feedback for automated ICD-10 Medical Coding</h1>",
    unsafe_allow_html=True
)

# Sidebar options
st.sidebar.header("Pseudo-Relevance Feedback")
Task = st.sidebar.selectbox(
    'Document Selection',
    ["Ranking", "Classification"]
)
if Task == "Classification":
    Precisionk = st.sidebar.selectbox(
        'Precision@k',
        ["8"],  # Set to "--" when Classification is selected
        index=0,
        disabled=True  # Ensure it's disabled
    )

else:
    Precisionk = st.sidebar.selectbox(
        'Precision@k',
        [i for i in range(1, 16)],  # Generates numbers 1 to 15
        index=7  # Default to 8th item (Precision@8)
    )



# Tuning Parameters
st.sidebar.header("Tuning Parameter")
iteration = st.sidebar.slider("Iteration", min_value=0, max_value=5, value=2)
TopKSelection = st.sidebar.slider("TopKSelection", min_value=1, max_value=15, value=10)
CosSim_Thresh = st.sidebar.slider("CosSim_Thresh", min_value=0.0, max_value=1.0, value=0.00, step=0.01)
alpha = st.sidebar.slider("alpha", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
beta = st.sidebar.slider("beta", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
gramma = st.sidebar.slider("gramma", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Create columns for wide layout
col1,col3,col2,col4 = st.columns([15,1,15,1])

import streamlit as st

with col1:
    st.subheader("Input Text")
    input_text = st.text_area(
        "Enter your text here for prediction:",
        value=("""
Name:  ___                   Unit No:   ___
 
Admission Date:  ___              Discharge Date:   ___
 
Date of Birth:  ___             Sex:   F
 
Service: PLASTIC
 
Allergies: 
Arava / ceftriaxone / ciprofloxacin / Cymbalta / Enbrel / 
methotrexate / Penicillins / shellfish derived / Sulfa 
(Sulfonamide Antibiotics) / aztreonam
 
Attending: ___.
 
Chief Complaint:
Right ___ finger pain
 
Major Surgical or Invasive Procedure:
Bedside I&D on ___ by Dr. ___

 
___ of Present Illness:
___ female presents with h/o Right ___ digit neuroma now POD8
from neuroma excision who presents from clinic with c/f surgical
site infection. Reports over last 48 hours has noticed increased
pain and redness around the incision. Reports increased
difficulty bending and moving the finger. Reports felt feverish
though did not check temperature. Reports went to clinic today
and was sent in for further evaluation and IV abx. 
Denies drainage. Reports the erythema is now advancing 
proximally
up the lateral aspect of the hand.  She denies chills, nausea,
vomiting, abdominal pain, chest pain, SOB. 
 
Past Medical History:
PAST MEDICAL HISTORY:  
- seronegative arthritis - previously on enbrel then humira 
which
was stopped in ___ due to concern for infections
- asthma
- obesity
- s/p left hip infection (___) - treated with 4 weeks
antibiotics, did not involve joint space per patient
- necrotizing fasciitis of the chest wall in ___, s/p extended
hospitalization and multiple debridements, ?___ strep
- ?culture negative endocarditis concurrent with nec. fasc. in
___

PAST SURGICAL HISTORY:
- multiple debridements for chest wall nec. fasc. (___)
 
Social History:
___
Family History:
No history of recurrent infections
 
Physical Exam:
Nonlabored on breathing on RA
Detailed examination of the RUE:
minimal erythema of the R ___ finger and ulnar aspect of palm, 
incision are without drainage
 
Pertinent Results:
___ 04:53AM BLOOD WBC-4.3 RBC-3.52* Hgb-10.2* Hct-31.5* 
MCV-90 MCH-29.0 MCHC-32.4 RDW-13.0 RDWSD-42.7 Plt ___
 
Brief Hospital Course:
The patient was seen and evaluated by the hand surgery team in 
the emergency department and was found to have Right ___ finger 
infection. She was admitted to the hand surgery service and was 
initially treated with IV vancomycin and cefepime and 
transitioned to oral clindamycin on discharge. On ___, a beside 
I&D was performed and she tolerated the procedure well. No 
obvious pus.

At the time of discharge, she was tolerating a PO diet, pain was 
well controlled. She is in agreement with the discharge.
 
Discharge Medications:
1.  Acetaminophen 650 mg PO 5X/DAY 
RX *acetaminophen 325 mg 2 capsule(s) by mouth 5 times a day 
Disp #*120 Capsule Refills:*0 
2.  Clindamycin 300 mg PO Q6H Duration: 14 Days 
RX *clindamycin HCl 300 mg 1 capsule(s) by mouth every 6 hours 
Disp #*56 Capsule Refills:*0 
3.  HYDROmorphone (Dilaudid) 2 mg PO Q4H:PRN Pain - Moderate 
RX *hydromorphone [Dilaudid] 2 mg 1 tablet(s) by mouth every 4 
hours Disp #*30 Tablet Refills:*0 
4.  ALPRAZolam 0.5 mg PO BID:PRN anxiety  
5.  Hydroxychloroquine Sulfate 200 mg PO DAILY  
6.  Montelukast 10 mg PO DAILY  
7.  Venlafaxine XR 75 mg PO DAILY  
8.  Xeljanz (tofacitinib) 5 mg oral BID  

 
Discharge Disposition:
Home
 
Discharge Diagnosis:
Right hand ___ finger infection

 
Discharge Condition:
Stable

 
Discharge Instructions:
Wound care:
Keep the wound clean and dry
Hydrogen peroxide/saline soaks twice daily for 20 minutes until 
___

Medications:
Resume your home medications
Take narcotics only for severe pain
Take clindamycin for 14 days for your finger infection

Warning signs:
Increasing pain, swelling or discharge of your finger
Fevers and chills
 
Followup Instructions:
___
"""
    ),
        placeholder="Enter your text here for prediction.",
        height=350
    )


# Output section

    st.subheader("Ground Truth")
    output_text = st.multiselect(
        "Enter the ground truth ICD-10 code(s):",
        options=fetch_icd_mapping(),
        default=[  "Y83.8", "T81.4XXA", "Y92.9", "J45.909", "02HV33Z", "E66.9","M19.90", "Z68.33", "L03.011"],
        help="Select the appropriate codes for ground truth."
    )


# Submit button

if st.button("Predict"):
    
    
    payload = {
        "Precisionk": Precisionk,
        "id": [0],
        "text": [input_text],
        "target": [output_text],
        "split": "test",
        "iteration": iteration,
        "TopKSelection": TopKSelection,
        "CosSim_Thresh": CosSim_Thresh,
        "alpha": alpha,
        "beta": beta,
        "gramma": gramma,
        "Task": Task
    }

    try:
        response = requests.post(
            url="http://localhost:8081/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
          
            if "data" in result:
                
                predictions = result["data"]["0"].get("result", {})
                match_percentage = result["data"]["0"].get("match_percentage", 0)

                if predictions:
                    with col2:
                        st.write("### Prediction Details:")
                        for code, description in predictions.items():
                            st.markdown(f"- **{code}**: {description}")
                        st.metric("Match Percentage", f"{match_percentage}%")
                else:
                    st.error("No predictions found in the response.")
            else:
                st.error("No predictions found in the response.")
        else:
            st.error(f"Error from backend: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred while connecting to the backend: {str(e)}")