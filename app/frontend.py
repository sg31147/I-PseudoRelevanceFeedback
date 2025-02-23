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
    page_title="Pseudo-Relevance Feedback on Deep Language Models for Medical Document Summarization",
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
    "<h1 style='text-align: center; color: #004085;'>Pseudo-Relevance Feedback on Deep Language Models for Medical Document Summarization</h1>",
    unsafe_allow_html=True
)

# Sidebar options
st.sidebar.header("Pseudo-Relevance Feedback")
Task = st.sidebar.selectbox(
    'Document Selection',
    ["Ranking", "Classification"]
)

Precisionk = st.sidebar.selectbox(
    'Precision@k',
    [i for i in range(1, 16)],  # Generates numbers 1 to 15
    index=7,
    disabled=(Task == "Classification")  # Disable if Task is Classification
)

# Tuning Parameters
st.sidebar.header("Tuning Parameter")
iteration = st.sidebar.slider("Iteration", min_value=0, max_value=10, value=2)
w_alpha = st.sidebar.slider("w_alpha", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
w_beta = st.sidebar.slider("w_beta", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
w_gramma = st.sidebar.slider("w_gramma", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
AvgTopR = st.sidebar.slider("AvgTopR", min_value=1, max_value=40, value=10)
AvgLowR = st.sidebar.slider("AvgLowR", min_value=1, max_value=40, value=10)

# Create columns for wide layout
col1,col3,col2,col4 = st.columns([15,1,15,1])

import streamlit as st

with col1:
    st.subheader("Input Text")
    input_text = st.text_area(
        "Enter your text here for prediction:",
        value=(
            "Name: [Patient's Name]\n"
            "Unit No: [Unit Number]\n"
            "Admission Date: [Date]\n"
            "Discharge Date: [Date]\n"
            "Date of Birth: [DOB]\n"
            "Sex: [M/F]\n"
            "Service: [Medical Service]\n"
            "Surgery: [Surgical Procedure]\n"
            "Allergies: Corgard, Vasotec\n"
            "\n"
            "Attending Physician: [Doctor's Name]\n"
            "Chief Complaint: Incarcerated inguinal hernia\n"
            "\n"
            "Major Surgical or Invasive Procedure:\n"
            "- Left inguinal hernia repair\n"
            "\n"
            "History of Present Illness:\n"
            "- Atrial fibrillation (AFib) on Apixaban\n"
            "- Coronary artery disease (CAD) s/p CABG\n"
            "- Bilateral carotid disease\n"
            "- COPD/emphysema with recent pneumonia\n"
            "- Presents for elective left inguinal hernia repair with large incarcerated sigmoid colon\n"
            "\n"
            "Past Medical History:\n"
            "- Bilateral moderate carotid disease\n"
            "- Congestive heart failure\n"
            "- Coronary artery disease\n"
            "- Gastroesophageal reflux disease (GERD)\n"
            "- Hypertension\n"
            "- Severe emphysema\n"
            "- Pulmonary hypertension\n"
            "- Right bundle branch block\n"
            "- Benign prostatic hypertrophy\n"
            "- Hyperlipidemia\n"
            "- Paroxysmal atrial fibrillation\n"
            "- History of histoplasmosis\n"
            "\n"
            "Past Surgical History:\n"
            "- Cardioversion\n"
            "- Right lower lobe lobectomy\n"
            "- Coronary bypass surgery\n"
            "\n"
            "Social & Family History: Non-contributory\n"
            "\n"
            "Physical Exam:\n"
            "- General: Awake and alert\n"
            "- CV: Irregularly irregular rhythm, normal rate\n"
            "- Respiratory: CTAB (Clear to auscultation bilaterally)\n"
            "- GI: Soft, appropriately tender near incision, non-distended\n"
            "- Incision: Clean, dry, intact, no erythema\n"
            "- Extremities: Warm and well-perfused\n"
            "\n"
            "Pertinent Results:\n"
            "- Brief hospital course: Patient admitted for left incarcerated inguinal hernia repair.\n"
            "- Postoperative course: Uncomplicated. Transferred to regular nursing floor after PACU stay.\n"
            "- Pain controlled with IV medications initially, then transitioned to PO meds.\n"
            "- Ambulating independently, tolerating regular diet.\n"
            "- Bowel regimen given, passed flatus, and voided independently.\n"
            "- Discharged home in stable condition with follow-up scheduled.\n"
            "\n"
            "Discharge Medications:\n"
            "- Amiodarone 200 mg PO daily\n"
            "- Apixaban 5 mg PO BID\n"
            "- Aspirin 81 mg PO daily\n"
            "- Docusate Sodium 100 mg PO BID\n"
            "- Losartan Potassium 50 mg PO daily\n"
            "- Omeprazole 40 mg PO daily\n"
            "- Triamterene/HCTZ 37.5/25 mg PO daily\n"
            "- Acetaminophen 500 mg PO Q6H PRN pain/fever (Max: 4g/day)\n"
            "- Oxycodone Immediate Release 5 mg PO Q4H PRN pain\n"
            "- Senna 8.6 mg PO HS while taking oxycodone\n"
            "- Align Bifidobacterium Infantis 4 mg PO daily\n"
            "- Coenzyme Q10 100 mg PO daily\n"
            "- Rosuvastatin Calcium 10 mg PO QPM\n"
            "- Vitamin D 1000 units PO daily\n"
            "\n"
            "Discharge Disposition: Home with service facility\n"
            "Discharge Diagnosis: Inguinal hernia\n"
            "\n"
            "Discharge Condition:\n"
            "- Mental status: Clear and coherent\n"
            "- Level of consciousness: Alert and interactive\n"
            "- Activity status: Ambulatory, independent\n"
            "\n"
            "Discharge Instructions:\n"
            "Dear Mr. [Last Name],\n"
            "It was a pleasure taking care of you. You were admitted for inguinal hernia repair and have recovered well. Please follow the recommendations below for a smooth recovery:\n"
            "\n"
            "Activity:\n"
            "- Do not drive until you stop taking pain medicine and feel capable of responding in an emergency.\n"
            "- You may climb stairs and go outside but avoid long-distance travel until cleared by your surgeon.\n"
            "- Avoid lifting more than 10 lbs for 6 weeks.\n"
            "- Light exercise is allowed when you feel comfortable; heavy exercise can resume after 6 weeks.\n"
            "- Avoid bathtubs/swimming pools until your incision heals (ask your doctor for specifics).\n"
            "\n"
            "How You May Feel:\n"
            "- You may feel weak for several weeks; naps may help.\n"
            "- Sore throat may occur due to intubation.\n"
            "- Temporary difficulty concentrating, poor appetite, or mild depression is normal.\n"
            "\n"
            "Incision Care:\n"
            "- Slight redness around the incision is normal.\n"
            "- Do not remove Steri-Strips for 2 weeks (or allow them to fall off naturally).\n"
            "- Gently wash around the incision.\n"
            "- Avoid direct sun exposure to the incision.\n"
            "- Call your surgeon if severe drainage or redness occurs.\n"
            "\n"
            "Bowel Management:\n"
            "- Constipation is common due to pain medications.\n"
            "- Use a stool softener (Colace) or mild laxative (Milk of Magnesia) as needed.\n"
            "- Call your surgeon if no bowel movement in 48 hours.\n"
            "- Diarrhea after surgery may occur; avoid anti-diarrheal medications unless directed.\n"
            "\n"
            "Pain Management:\n"
            "- Mild pain is expected but should improve daily.\n"
            "- Take prescribed pain medications as directed.\n"
            "- Contact your surgeon for severe pain, fever above 101°F, or sudden changes in pain quality.\n"
            "\n"
            "Medications:\n"
            "- Resume preoperative medications unless instructed otherwise.\n"
            "- If unsure about any medication, contact your surgeon.\n"
            "\n"
            "Follow-Up Instructions: [Insert follow-up appointment details]\n"
        ),
        placeholder="Enter your text here for prediction.",
        height=300
    )


# Output section

    st.subheader("Prediction Results")
    output_text = st.multiselect(
        "Ground Truth",
        options=fetch_icd_mapping(),
        default=["0YU60JZ", "E78.5", "I10.", "I25.10", "I27.2", "I45.10", "I48.0", "I50.9", "J43.9", "K21.9", "K40.30", "N40.0", "Z79.82", "Z87.891", "Z95.1"],
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
        "w_alpha": w_alpha,
        "w_beta": w_beta,
        "NavgTops": AvgTopR,
        "w_gramma": w_gramma,
        "NavgLow": AvgLowR,
        "Task": Task,
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
                predictions = result["data"]["0"].get("predictions_topk", {})
                match_percentage = result["data"]["0"].get("match_percentage_topk", 0)

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