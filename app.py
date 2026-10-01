import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Number Plate AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background-color: #0b1120;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 5px;
}

.card-subtitle {
    font-size: 14px;
    color: #94a3b8;
}

.stat-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.stat-number {
    font-size: 30px;
    font-weight: 800;
    color: #38bdf8;
}

.stat-label {
    font-size: 13px;
    color: #94a3b8;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1f2937;
}

.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    border: none;
    background: #2563eb;
    color: white;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

section[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #334155;
    border-radius: 14px;
    padding: 15px;
}

.custom-divider {
    height: 1px;
    background: #1f2937;
    margin: 30px 0;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("Best_Plate_Number_Detecting_Model.pt")


model = load_model()


with st.sidebar:

    st.markdown(
        "<h2 style='color:white;'>🚘 Number Plate AI</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#94a3b8;'>AI-powered number plate detection system</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        "<p style='color:white;font-weight:700;'>Detection Settings</p>",
        unsafe_allow_html=True
    )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            background:#111827;
            padding:15px;
            border-radius:12px;
            border:1px solid #1f2937;
        ">

        <p style="color:#94a3b8;margin:0;font-size:13px;">
        MODEL
        </p>

        <p style="color:white;font-weight:700;margin-top:5px;">
        YOLO
        </p>

        <p style="color:#94a3b8;margin:10px 0 0;font-size:13px;">
        STATUS
        </p>

        <p style="color:#22c55e;font-weight:700;margin-top:5px;">
        ● ONLINE
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    "<div class='main-title'>Number Plate Detection</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Upload a vehicle image and let the YOLO model automatically detect number plates.</div>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="card">
        <div class="card-title">Upload Vehicle Image</div>
        <div class="card-subtitle">
            Supported formats: JPG, JPEG, PNG
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">Original Image</div>
                <div class="card-subtitle">
                    Uploaded vehicle image
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">AI Detection</div>
                <div class="card-subtitle">
                    YOLO detection result
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        detect_button = st.button(
            "🔍 Detect Number Plate"
        )

        if detect_button:

            with st.spinner("AI is analyzing the image..."):

                results = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False
                )

            result = results[0]

            detected_image = result.plot()

            st.image(
                detected_image,
                channels="BGR",
                use_container_width=True
            )

            number_of_detections = len(result.boxes)

            if number_of_detections > 0:

                confidences = [
                    float(conf)
                    for conf in result.boxes.conf
                ]

                average_confidence = (
                    sum(confidences) /
                    len(confidences)
                )

                highest_confidence = max(confidences)

                st.markdown(
                    "<br>",
                    unsafe_allow_html=True
                )

                stat1, stat2, stat3 = st.columns(3)

                with stat1:

                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-number">
                                {number_of_detections}
                            </div>
                            <div class="stat-label">
                                Plates Detected
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with stat2:

                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-number">
                                {average_confidence:.1%}
                            </div>
                            <div class="stat-label">
                                Average Confidence
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with stat3:

                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-number">
                                {highest_confidence:.1%}
                            </div>
                            <div class="stat-label">
                                Highest Confidence
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.warning(
                    "No number plate detected. Try another image or lower the confidence threshold."
                )


    if detect_button and len(result.boxes) > 0:

        st.markdown(
            "<div class='custom-divider'></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">
                <div class="card-title">Detection Details</div>
                <div class="card-subtitle">
                    Details of detected number plates
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        detection_data = []

        for i, box in enumerate(result.boxes):

            class_id = int(box.cls[0])

            confidence_value = float(
                box.conf[0]
            )

            coordinates = box.xyxy[0].tolist()

            detection_data.append({
                "Detection": f"Plate {i + 1}",
                "Class": result.names[class_id],
                "Confidence": f"{confidence_value:.2%}",
                "X1": round(coordinates[0], 1),
                "Y1": round(coordinates[1], 1),
                "X2": round(coordinates[2], 1),
                "Y2": round(coordinates[3], 1)
            })

        df = pd.DataFrame(
            detection_data
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


else:

    st.markdown(
        """
        <div style="
            margin-top:30px;
            background:#111827;
            border:1px solid #1f2937;
            border-radius:16px;
            padding:50px;
            text-align:center;
        ">

        <div style="font-size:55px;">🚘</div>

        <h2 style="color:white;">
        Ready for Detection
        </h2>

        <p style="color:#94a3b8;">
        Upload a vehicle image above to start detecting number plates.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="footer">
        Number Plate AI • Powered by YOLO & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)