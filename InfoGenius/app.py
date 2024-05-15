import streamlit as st
import replicate
import requests
import zipfile
import io
from utils import icon
from streamlit_image_select import image_select
#
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def ImageGenerator():
   

    st.markdown("# :rainbow[Your Imagination Studio]")
    st.sidebar.header("ImageGenerator")
    # --- Initialize session state for generated images --- #
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None

    # --- Secret Sauce (API Tokens and Endpoints) --- #
    REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
    REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]


    # --- Placeholders for Images and Gallery --- #
    generated_images_placeholder = st.empty()
    gallery_placeholder = st.empty()

    # --- Sidebar Elements --- #
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Imagine any image you want and put down the details below**", icon="👋🏾")
            with st.expander(":rainbow[**Refine your output here**]"):
                # Advanced Settings (for the curious minds!)
                width = st.number_input("Width of output image", value=1024)
                height = st.number_input("Height of output image", value=1024)
                num_outputs = st.slider(
                    "Number of images to output", value=1, min_value=1, max_value=4)
                scheduler = st.selectbox('Scheduler', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                    'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
                num_inference_steps = st.slider(
                    "Number of denoising steps", value=50, min_value=1, max_value=500)
                guidance_scale = st.slider(
                    "Scale for classifier-free guidance", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
                prompt_strength = st.slider(
                    "Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of infomation in image)", value=0.8, max_value=1.0, step=0.1)
                refine = st.selectbox(
                    "Select refine style to use (left out the other 2)", ("expert_ensemble_refiner", "None"))
                high_noise_frac = st.slider(
                    "Fraction of noise to use for `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
            prompt = st.text_area(
                ":orange[**Enter prompt: start typing,✍🏾**]",
                value="A group of friends laughing and dancing at a wedding, joyful atmosphere, 35mm film photography")
            negative_prompt = st.text_area(":orange[**Things you don't want in image? 🙅🏽‍♂️**]",
                                        value="the absolute worst quality, distorted features",
                                        help="This is a negative prompt, basically type what you don't want to see in the generated image")

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)

    
    # --- Image Generation --- #
    if submitted:
        with st.status('👩🏾‍🍳 Waiting for the magic', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and do Squats in the meantime")
            try:
                # Only call the API if the "Submit" button was pressed
                if submitted:
                    # Calling the replicate API to get the image
                    with generated_images_placeholder.container():
                        all_images = []  # List to store all generated images
                        output = replicate.run(
                            REPLICATE_MODEL_ENDPOINTSTABILITY,
                            input={
                                "prompt": prompt,
                                "width": width,
                                "height": height,
                                "num_outputs": num_outputs,
                                "scheduler": scheduler,
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                                "prompt_stregth": prompt_strength,
                                "refine": refine,
                                "high_noise_frac": high_noise_frac
                            }
                        )
                        if output:
                            st.toast('Your image s generated!', icon='😍')
                            # Save generated image to session state
                            st.session_state.generated_image = output

                            # Displaying the image
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(image, caption="Generated Image 🎈",
                                            use_column_width=True)
                                    # Add image to the list
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Save all generated images to session state
                        st.session_state.all_images = all_images

                        # Create a BytesIO object
                        zip_io = io.BytesIO()

                        # Download option for each image
                        with zipfile.ZipFile(zip_io, 'w') as zipf:
                            for i, image in enumerate(st.session_state.all_images):
                                response = requests.get(image)
                                if response.status_code == 200:
                                    image_data = response.content
                                    # Write each image to the zip file with a name
                                    zipf.writestr(
                                        f"output_file_{i+1}.png", image_data)
                                else:
                                    st.error(
                                        f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}", icon="🚨")
                        # Create a download button for the zip file
                        st.download_button(
                            ":red[**Download All Images**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
                status.update(label="✅ Images generated!",
                            state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Encountered an error: {e}', icon="🚨")

    # If not submitted, chill here 🍹
    else:
        pass

    # --- Gallery Display for inspiration or just plain admiration --- #
    with gallery_placeholder.container():
        img = image_select(
            label="A short collection for you 😉",
            images=[
                "gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
                "gallery/cheetah.png"
            ],
            captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                    "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                    "A cheetah mother nurses her cubs in the tall grass of the Serengeti. The early morning sun beams down through the grass. National Geographic photography by Frans Lanting",
                    ],
            use_container_width=True
        )

def AskPDF():
    # PDF processing function
    def process_pdf(pdf,key):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # store_name = pdf.name[:-4]

       # if os.path.exists(f"{store_name}.pkl"):
            #with open(f"{store_name}.pkl", "rb") as f:
           #     VectorStore = pickle.load(f)
        #else:
        embedding = OpenAIEmbeddings(openai_api_key= key)
        VectorStore = FAISS.from_texts(chunks,embedding)
          #  with open(f"{store_name}.pkl", "wb") as f:
          #      pickle.dump(VectorStore, f)

        return VectorStore

    # Chatbot response function
    def get_chatbot_response(query, VectorStore,key):
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI(openai_api_key= key)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
        return response
    
    st.title("AI QA System")

    # PDF processing and QA
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    key= st.secrets["OPENAI_API_KEY"]

    if pdf is not None:
        VectorStore = process_pdf(pdf, key)

    # Chatbot UI
    if pdf:
        st.header("Chat with PDF")


        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask Me"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            query = prompt  # Use the chatbot's query as the PDF QA query
            response = get_chatbot_response(query, VectorStore,key)
            st.write(response)

            # Display the PDF QA answer in the chat
            st.session_state.messages.append({"role": "assistant", "content": response})

    

page_names_to_funcs = {
    "ImageGenerator": ImageGenerator,
    "AskPDF": AskPDF,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()