import streamlit as st
from rag_db_func import (
    list_files,
    process_uploaded_files,
    display_file_info,
    toggle_elements_visibility,
    download_elements_as_json,
    delete_file,
    delete_all_files,
    create_rag_data_section,
    display_rag_data_versions,
)

from user_logger import log_action
from language import translate


# Function to display the full Rag DB app
def display_rag_db_app():
    # Initialize session state keys if not already set
    if "rag_file_uploader_key" not in st.session_state:
        st.session_state.rag_file_uploader_key = 0
    if "rag_uploaded_files" not in st.session_state:
        st.session_state["rag_uploaded_files"] = []

    st.title(translate("File and Rag Data Manager"))

    # Sidebar Navigation
    st.sidebar.title(translate("Navigation"))

    # Define pages using keys
    page_keys = ["Upload File", "Manage Files", "Rag Data Management"]
    page_labels = [translate(key) for key in page_keys]

    # Show translated options, but get back selected key
    selected_index = st.sidebar.radio(
        translate("Go to"),
        options=range(len(page_keys)),
        format_func=lambda i: page_labels[i]
    )
    page = page_keys[selected_index]

    # Display the selected section
    if page == "Upload File":
        display_upload_file_section()
    elif page == "Manage Files":
        display_manage_files_section()
    elif page == "Rag Data Management":
        display_rag_data_management_section()

    # Sidebar button to return to neu_main
    if st.sidebar.button(translate("Back to Main Page")):
        st.session_state["current_page"] = "rag_main_page"
        log_action("Go to Rag Main page.")
        # Keep the previous status
        st.session_state["model_select"] = st.session_state["model_name"]
        st.session_state["llm_engine_select"] = st.session_state["llm_engine"]
        st.session_state["rag_toggle"] = st.session_state["rag_on"]
        st.session_state["rag_engine_select"] = st.session_state["rag_engine"]
        st.rerun()  # Refresh the page to switch back to neu_main


def update_file_uploader_key():
    # Delete the old file uploader key from session_state
    old_key = f"rag_file_uploader_{st.session_state.rag_file_uploader_key}"
    if old_key in st.session_state:
        del st.session_state[old_key]

    """Updates the key for file_uploader widget to reset it."""
    st.session_state.rag_file_uploader_key += 1


# Function to display the "Upload File" section
def display_upload_file_section():
    st.header(translate("Upload Multiple Files and Create Elements"))
    rag_uploaded_files = st.file_uploader(
        translate("Choose PDF files"),
        type=["pdf"],
        accept_multiple_files=True,
        key=f"rag_file_uploader_{st.session_state.rag_file_uploader_key}",
    )
    if rag_uploaded_files:
        st.session_state["rag_uploaded_files"] = rag_uploaded_files

    if st.session_state["rag_uploaded_files"]:
        if st.button(translate("Upload and Process All Files"), on_click=update_file_uploader_key):
            process_uploaded_files()
            log_action("uploaded_files :" + str(st.session_state["rag_uploaded_files"]))


# Function to display the "Manage Files" section
def display_manage_files_section():
    st.header(translate("View and Delete Files"))

    # Directly call `delete_all_files()` which already includes the delete button logic
    delete_all_files()

    files = list_files()

    if files:
        for file in files:
            display_file_info(file)
            toggle_elements_visibility(file)
            download_elements_as_json(file)
            delete_file(file)
            st.write("---")
    else:
        st.write(translate("No files available."))


# Function to display the "Rag Data Management" section
def display_rag_data_management_section():
    st.header(translate("Create and Manage Rag Data Versions"))
    create_rag_data_section()
    st.subheader(translate("Existing Rag Data Versions"))
    display_rag_data_versions()
