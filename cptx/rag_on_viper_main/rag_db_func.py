import streamlit as st
import datetime
import os
import json
import time
import torch
import pickle
from typing import Any, List, Dict
from io import BytesIO
from unstructured.partition.pdf import partition_pdf
from rag_db_operations import (
    list_files,
    get_file_by_name,
    list_elements_by_file_id,
    create_file_with_elements,
    delete_file_with_elements,
    create_rag_data_with_files,
    get_rag_data,
    delete_rag_data,
    list_rag_data_versions,
    delete_all_file_records,
)
import llm_rag_config as config
from user_logger import log_action
from language import translate


# Upload File module========================================
def categorize_elements(raw_pdf_elements: List[Any]) -> List[Dict[str, str]]:
    """
    Categorizes each parsed element from the PDF as either 'text' or 'table'.
    """
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(
                {"type": "table", "content": str(element)}
            )
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(
                {"type": "text", "content": str(element)}
            )
    return categorized_elements


def save_temp_file(uploaded_file, save_path):
    """Saves uploaded file to a temporary path."""
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return save_path


def parse_pdf(file_path):
    """Parses PDF to extract raw elements."""
    st.info(translate("Parsing PDF:  '{}'").format(os.path.basename(file_path)))
    start_time = time.time()
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=400,
        new_after_n_chars=380,
        combine_text_under_n_chars=200,
        image_output_dir_path="./parsing_data",
        languages=["eng","chi_sim"],
    )
    duration = time.time() - start_time
    duration_str = f"{duration:.2f}"
    st.info(translate("Parsing completed in '{}' seconds.").format(duration_str))
    return raw_pdf_elements


def categorize_and_save_elements(uploaded_file, raw_pdf_elements, upload_time):
    """Categorizes elements and saves them along with file info to the database."""
    file_name = uploaded_file.name
    file_size = len(uploaded_file.getvalue())

    # Categorize elements
    categorized_elements = categorize_elements(raw_pdf_elements)

    # Store in database
    try:
        file_id = create_file_with_elements(
            filename=file_name,
            upload_time=upload_time,
            elements=categorized_elements,
            status="generated",
            file_size=file_size,
        )
        if file_id:
            st.info(
                translate("File '{}' uploaded and processed with file ID '{}'.").format(file_name, file_id)
            )
        else:
            st.info(translate("Failed to upload file '{}'.").format(file_name))
    except Exception as e:
        st.error(translate("Error inserting '{}': '{}'").format(file_name, e))


def process_file(uploaded_file, upload_time):
    """Processes a single file, checking existence, parsing, categorizing, and saving."""
    file_name = uploaded_file.name

    # Check if file already exists in the database
    if get_file_by_name(file_name):
        st.info(
            translate("File '{}' already exists in the database. Skipping processing.").format(file_name)
        )
        return

    # Create save path
    os.makedirs(config.save_path, exist_ok=True)
    temp_save_path = os.path.join(config.save_path, file_name)

    # Process the file
    try:
        # Save the file temporarily
        save_temp_file(uploaded_file, temp_save_path)

        # Parse the PDF
        raw_pdf_elements = parse_pdf(temp_save_path)

        # Categorize and save elements
        categorize_and_save_elements(uploaded_file, raw_pdf_elements, upload_time)
        return True

    except Exception as e:
        st.error(translate("Error processing '{}': {}").format(file_name, e))
        return False

    finally:
        # Remove the temporary file after processing
        try:
            os.remove(temp_save_path)
        except OSError as e:
            st.warning(translate("Failed to delete temporary file '{}': {}").format(temp_save_path, e))


# Function to handle processing all uploaded files
def process_uploaded_files():
    upload_time = datetime.datetime.now()
    total_files = len(st.session_state["rag_uploaded_files"])
    progress_bar = st.progress(0)
    fail_process_files = []

    for idx, uploaded_file in enumerate(st.session_state["rag_uploaded_files"]):
        if process_file(uploaded_file, upload_time):
            progress_bar.progress((idx + 1) / total_files)  # Update progress bar
        else:
            fail_process_files.append(uploaded_file)

    if not fail_process_files:
        st.success(translate("All files have been processed successfully."))
    else:
        # Compute the joined list of filenames
        failed_files = ", ".join([file.name for file in fail_process_files])
        # Translate and format the string
        message = translate("Some files failed to process: {}").format(failed_files)
        st.warning(message)


    st.session_state["rag_uploaded_files"] = []


# Upload Manage Files module========================================
def display_file_info(file):
    """
    Displays file information
    """
    try:
        # Check if file is a dictionary and contains the expected keys
        if not isinstance(file, dict):
            raise ValueError("File data is not in the expected format (dictionary).")

        # Retrieve and display file information, with fallback for missing keys
        file_id = file.get('id', 'Unknown ID')
        filename = file.get('filename', 'Unknown Filename')
        status = file.get('status', 'Unknown Status')
        upload_time = file.get('upload_time', 'Unknown Upload Time')
        file_size = file.get('file_size', 'Unknown Size')

        st.write(f"**File ID**: {file_id}")
        st.write(f"**Filename**: {filename}")
        st.write(f"**Status**: {status}")
        st.write(f"**Upload Time**: {upload_time}")
        st.write(f"**File Size**: {file_size} bytes")

    except Exception as e:
        # Display an error message if file info cannot be shown
        st.error(translate("Error displaying file info: '{}").format(e))


# Helper function to toggle and show elements for a file
def toggle_elements_visibility(file):
    show_elements = st.checkbox(
        f"Show Elements for {file['filename']}",
        key=f"show_elements_{file['id']}",
    )
    if show_elements:
        elements = list_elements_by_file_id(file["id"])
        if elements:
            st.subheader(f"Elements for File: {file['filename']}")
            for element in elements:
                st.write(f"**Element Type**: {element['element_type']}")
                st.write(f"**Content**: {element['content']}")
                st.write("---")
        else:
            st.info(translate("No elements found for file '{}'.").format(file['filename']))


# Helper function to download elements as JSON for a file
def download_elements_as_json(file):
    elements = list_elements_by_file_id(file["id"])
    if elements:
        # Convert elements to JSON format
        json_data = json.dumps(elements, indent=4)
        # Display download button for JSON data
        st.download_button(
            label=translate("Download Elements for '{}'").format(file['filename']),
            data=json_data,
            file_name=f"{file['filename']}_elements.json",
            mime="application/json",
            key=f"download_{file['id']}",
        )


# Helper function to delete a specific file and its elements
def delete_file(file):
    if st.button(translate("Delete '{}'").format(file['filename']), key=f"delete_{file['id']}"):
        if delete_file_with_elements(file["id"]):
            st.success(translate("File '{}' and its elements were deleted.").format(file['filename']))

            log_action("delete_file_with_elements :" + str(file["id"]))
            st.rerun()
        else:
            st.error(translate("Failed to delete file '{}'.").format(file['filename']))


# Helper function to delete all files and their elements, then rerun the app
def delete_all_files():
    if st.button(translate("Delete All Files")):
        delete_all_file_records()
        log_action("delete_all_file_records")
        st.rerun()


# Rag Data Management module===================================================================
# Helper function to create Rag Data from selected files
def create_rag_data_section():
    files = list_files()
    file_choices = {file["filename"]: file["id"] for file in files}

    # Multi-select for files to include in the Rag Data version
    selected_files = st.multiselect(
        translate("Select Files to Create Rag Data"), options=list(file_choices.keys())
    )
    selected_file_ids = [file_choices[file] for file in selected_files]

    if selected_file_ids:
        version_name = st.text_input(translate("Enter Version Name"))
        generation_time = datetime.datetime.now()

        # Button to create Rag Data is disabled if version_name is empty
        create_button_disabled = not bool(version_name.strip())

        if st.button(translate("Create Rag Data"), disabled=create_button_disabled):
            rag_data_id = create_rag_data_with_files(
                version_name=version_name,
                generation_time=generation_time,
                file_ids=selected_file_ids,
                status="generated",
            )
            if rag_data_id:
                st.success(translate("Rag Data version '{}' created with ID '{}'.").format(version_name, rag_data_id))
                log_action(f"Rag Data version '{version_name}' created with ID {rag_data_id}.")
            else:
                st.error(translate("Failed to create Rag Data version."))


# Helper function to display each Rag Data version and its details
def display_rag_data_versions():
    rag_data_versions = list_rag_data_versions()
    if not rag_data_versions:
        st.info(translate("No Rag Data versions available."))
        return

    for rag_data in rag_data_versions:
        st.write(f"**Rag Data ID**: {rag_data['id']}")
        st.write(f"**Version Name**: {rag_data['version_name']}")
        st.write(f"**Status**: {rag_data['status']}")
        st.write(f"**Generation Time**: {rag_data['generation_time']}")

        # Use rag_data directly for download and delete options if it has all necessary details
        create_download_buttons(rag_data)
        create_delete_button(rag_data)


# Helper function to create download buttons for raw_data and embeddings
def create_download_buttons(rag_data):
    # Prepare raw_data for download as JSON
    raw_data_list = json.loads(rag_data["raw_data"])
    json_raw_data = json.dumps(raw_data_list, indent=4)
    st.download_button(
        label=translate("Download Raw Data as JSON for '{}'").format(rag_data['version_name']),
        data=json_raw_data,
        file_name=f"{rag_data['version_name']}_raw_data.json",
        mime="application/json",
        key=f"download_json_raw_{rag_data['id']}",
    )

    # Prepare raw_data and embeddings as .pt files
    raw_data_buffer, emb_data_buffer = prepare_tensors_for_download(rag_data)

    st.download_button(
        label=translate("Download Raw Data as .pt for '{}'").format(rag_data['version_name']),
        data=raw_data_buffer,
        file_name=f"{rag_data['version_name']}_raw_data.pt",
        mime="application/octet-stream",
        key=f"download_raw_{rag_data['id']}",
    )
    st.download_button(
        label=translate("Download Embedding Data as .pt for '{}'").format(rag_data['version_name']),
        data=emb_data_buffer,
        file_name=f"{rag_data['version_name']}_corpus_emb.pt",
        mime="application/octet-stream",
        key=f"download_emb_{rag_data['id']}",
    )


# Helper function to prepare raw_data and emb_data for download as .pt files
def prepare_tensors_for_download(rag_data):
    raw_data_list = json.loads(rag_data['raw_data'])
    corpus_emb_numpy = pickle.loads(rag_data['emb_data'])
    corpus_emb_tensor = torch.tensor(corpus_emb_numpy)

    raw_data_buffer = BytesIO()
    emb_data_buffer = BytesIO()

    torch.save(raw_data_list, raw_data_buffer)
    torch.save(corpus_emb_tensor, emb_data_buffer)

    # Save the files
    torch.save(corpus_emb_tensor, "data/I2_rag_db_corpus_emb_tensor.pt")

    raw_data_buffer.seek(0)
    emb_data_buffer.seek(0)

    return raw_data_buffer, emb_data_buffer


# Helper function to create a delete button for each Rag Data version
def create_delete_button(rag_data):
    if st.button(translate("Delete '{}' Rag Data").format(rag_data['version_name']), key=f"delete_rag_data_{rag_data['id']}"):
        if delete_rag_data(rag_data["id"]):
            st.success(translate("Rag Data version '{}' was deleted.").format(rag_data['version_name']))
            st.rerun()
        else:
            st.error(translate("Failed to delete Rag Data version '{}'.").format(rag_data['version_name']))

