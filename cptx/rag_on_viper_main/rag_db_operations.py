import os
import sqlite3
import datetime
import llm_rag_config as config
from typing import Optional, List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import pickle
import json
from tqdm import tqdm
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def is_qwen_model(model_name):
    return "qwen" in model_name.casefold()

# Database path and setup
DATABASE_PATH = config.rag_db_path + "rag_db/rag_db.db"

# Embedding models
@st.cache_resource
def loading_uae_model():
    emb_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
    return emb_model

@st.cache_resource
def loading_qwen_emb_model():
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    return tokenizer, model

def qwen_emb_encode(texts):
    if isinstance(texts, str):
        texts = [texts]

    tokenizer, model = loading_qwen_emb_model()
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state
    lengths = inputs["attention_mask"].sum(dim=1)
    eos_hidden = hidden[torch.arange(hidden.size(0)), lengths - 1, :]
    embeddings = F.normalize(eos_hidden, p=2, dim=1)

    return embeddings

def emb_encode(texts):
    return (
        qwen_emb_encode(texts)
        if is_qwen_model(st.session_state.model_name)
        else loading_uae_model().encode(texts, convert_to_tensor=True)
    )

# Initialize database and create tables if they don't exist
def initialize_database():
    conn = sqlite3.connect(DATABASE_PATH)  # Directly connect without using get_connection
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS files (
                        id INTEGER PRIMARY KEY,
                        filename TEXT UNIQUE NOT NULL,
                        upload_time TIMESTAMP NOT NULL,
                        status TEXT DEFAULT 'pending',
                        file_size INTEGER
                      )"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS elements (
                        id INTEGER PRIMARY KEY,
                        file_id INTEGER NOT NULL,
                        element_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        FOREIGN KEY(file_id) REFERENCES files(id)
                      )"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS rag_data (
                        id INTEGER PRIMARY KEY,
                        version_name TEXT UNIQUE NOT NULL,
                        raw_data TEXT NOT NULL,
                        emb_data BLOB,
                        generation_time TIMESTAMP NOT NULL,
                        status TEXT DEFAULT 'generated'
                      )"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS rag_data_elements_link (
                        id INTEGER PRIMARY KEY,
                        rag_data_id INTEGER NOT NULL,
                        element_id INTEGER NOT NULL,
                        FOREIGN KEY(rag_data_id) REFERENCES rag_data(id),
                        FOREIGN KEY(element_id) REFERENCES elements(id)
                      )"""
    )

    conn.commit()
    conn.close()

# Database connection setup
def get_connection():
    """
    Establishes a connection to the database. Initializes the database if it does not exist.

    :return: SQLite connection object
    """
    # Check if the database file exists; if not, initialize the database
    if not os.path.exists(DATABASE_PATH):
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)  # Create directory if it doesn't exist
        initialize_database()

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enables dictionary-style row access
    return conn

# CRUD Operations for Files Table==================================================================
def create_file(
    filename: str,
    upload_time: datetime.datetime,
    status: str = "pending",
    file_size: Optional[int] = None,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO files (filename, upload_time, status, file_size)
                      VALUES (?, ?, ?, ?)""",
        (filename, upload_time, status, file_size),
    )
    conn.commit()
    file_id = cursor.lastrowid
    conn.close()
    return file_id


def get_file_by_id(file_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_file_by_name(filename: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a file record by its filename.

    :param filename: The name of the file to retrieve
    :return: A dictionary containing the file record, or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM files WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# TODO delete file need also delete element and update rag_data_elements and rag_data_files
def delete_file_by_id(file_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def delete_file_by_name(filename: str) -> bool:
    """
    Deletes a file record by its filename.

    :param filename: The name of the file to delete
    :return: True if the file was deleted, False if the file was not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM files WHERE filename = ?", (filename,))
    deleted = cursor.rowcount > 0  # rowcount > 0 indicates that a row was deleted
    conn.commit()
    conn.close()
    return deleted


def update_file_metadata(file_id: int, **kwargs) -> bool:
    """
    Updates metadata fields for a file.

    :param file_id: ID of the file to update
    :param kwargs: Key-value pairs of the fields to update (e.g., filename, status, file_size)
    :return: True if the update was successful, False otherwise
    """
    valid_fields = {"filename", "status", "file_size", "upload_time"}
    fields_to_update = {k: v for k, v in kwargs.items() if k in valid_fields}

    if not fields_to_update:
        print("No valid fields to update.")
        return False

    conn = get_connection()
    cursor = conn.cursor()
    set_clause = ", ".join([f"{field} = ?" for field in fields_to_update.keys()])
    values = list(fields_to_update.values()) + [file_id]

    query = f"UPDATE files SET {set_clause} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()

    return updated


def list_files(limit: Optional[int] = None, order_by: str = "upload_time", descending: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieves a list of all files, optionally limited and ordered.

    :param limit: Maximum number of files to retrieve
    :param order_by: Column to order the results by (default is "upload_time")
    :param descending: Whether to sort in descending order (default is True)
    :return: A list of dictionaries containing file records
    """
    conn = get_connection()
    cursor = conn.cursor()

    order_direction = "DESC" if descending else "ASC"
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"SELECT * FROM files ORDER BY {order_by} {order_direction} {limit_clause}"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# High level api for Files Table======================================================================
def create_file_with_elements(filename: str, upload_time: datetime.datetime, elements: List[Dict[str, str]], status: str = "pending", file_size: Optional[int] = None) -> Optional[int]:
    """
    Creates a file record and its associated elements.

    :param filename: Name of the file
    :param upload_time: Timestamp of the file upload
    :param elements: List of dictionaries containing element details (type and content)
    :param status: Processing status, default is "pending"
    :param file_size: Size of the file in bytes
    :return: The ID of the created file, or None if the file already exists
    """
    # Create the file record
    file_id = create_file(filename, upload_time, status, file_size)

    if file_id is None:
        print(f"File '{filename}' already exists. No elements created.")
        return None

    # Create each element and associate it with the file
    for element in elements:
        create_element(file_id, element["type"], element["content"])

    return file_id


def delete_file_with_elements(file_identifier: Any) -> bool:
    """
    Deletes a file and all associated elements, identified by file_id or filename.

    :param file_identifier: Either the file_id (int) or filename (str) to delete
    :return: True if the file and its elements were deleted, False otherwise
    """
    # Determine if we have a file_id or filename and retrieve file_id if necessary
    if isinstance(file_identifier, int):
        file_id = file_identifier
    elif isinstance(file_identifier, str):
        file_record = get_file_by_name(file_identifier)
        if not file_record:
            print(f"No file found with filename '{file_identifier}'.")
            return False
        file_id = file_record["id"]
    else:
        print("Invalid file identifier. Must be an integer file_id or a string filename.")
        return False

    # Delete all elements associated with this file
    delete_elements_by_file_id(file_id)

    # Delete the file itself
    file_deleted = delete_file_by_id(file_id)

    if file_deleted:
        print(f"File with ID '{file_id}' and its associated elements were deleted successfully.")
    else:
        print(f"File with ID '{file_id}' could not be deleted.")

    return file_deleted


def delete_all_file_records():
    """Deletes all records from files, elements, and rag_data_elements_link tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Delete all records from related tables
    cursor.execute("DELETE FROM rag_data_elements_link")
    cursor.execute("DELETE FROM elements")
    cursor.execute("DELETE FROM files")

    conn.commit()
    conn.close()


# CRUD Operations for Elements Table==================================================================
def create_element(file_id: int, element_type: str, content: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO elements (file_id, element_type, content)
                      VALUES (?, ?, ?)""",
        (file_id, element_type, content),
    )
    conn.commit()
    element_id = cursor.lastrowid
    conn.close()
    return element_id


def get_element(element_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM elements WHERE id = ?", (element_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_element_ids_by_file_id(file_id: int) -> List[int]:
    """
    Retrieves all element IDs associated with a given file ID.

    :param file_id: The ID of the file whose elements are to be retrieved
    :return: A list of element IDs associated with the specified file ID
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM elements WHERE file_id = ?", (file_id,))
    rows = cursor.fetchall()

    element_ids = [row["id"] for row in rows]

    conn.close()

    return element_ids


def list_elements_by_file_id(file_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves all elements associated with a specific file.

    :param file_id: ID of the file to list elements for
    :return: A list of dictionaries containing element records for the specified file
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM elements WHERE file_id = ?", (file_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def delete_element(element_id: int) -> bool:
    """
    Deletes a specific element and removes any associated links from rag_data_elements_link.

    :param element_id: The ID of the element to be deleted
    :return: True if the element and links were deleted, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Delete links from rag_data_elements_link for this element
    cursor.execute("DELETE FROM rag_data_elements_link WHERE element_id = ?", (element_id,))

    # Delete the element itself
    cursor.execute("DELETE FROM elements WHERE id = ?", (element_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


def delete_elements_by_file_id(file_id: int) -> bool:
    """
    Deletes all elements associated with a specified file ID and removes their links from rag_data_elements_link.

    :param file_id: The ID of the file whose elements and associated links are to be deleted
    :return: True if elements and links were deleted, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Find element IDs associated with the file_id
    cursor.execute("SELECT id FROM elements WHERE file_id = ?", (file_id,))
    element_ids = [row["id"] for row in cursor.fetchall()]

    # Delete links from rag_data_elements_link for these elements
    cursor.executemany("DELETE FROM rag_data_elements_link WHERE element_id = ?", [(element_id,) for element_id in element_ids])

    # Delete the elements themselves
    cursor.execute("DELETE FROM elements WHERE file_id = ?", (file_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


def generate_combined_content(element_ids: List[int]) -> str:
    """
    Generates combined content from specified elements, with 'text' elements first,
    followed by 'table' elements.

    :param element_ids: List of element IDs whose content will be combined
    :return: Combined content string with text elements first, then table elements
    """
    conn = get_connection()
    cursor = conn.cursor()

    text_content_parts = []
    table_content_parts = []

    # Fetch content for each element and organize by type
    for element_id in element_ids:
        cursor.execute(
            "SELECT content, element_type FROM elements WHERE id = ?", (element_id,)
        )
        element = cursor.fetchone()

        if element:
            if element["element_type"] == "text":
                text_content_parts.append(element["content"])
            elif element["element_type"] == "table":
                table_content_parts.append(element["content"])
        else:
            raise ValueError(f"Element with ID {element_id} does not exist")

    conn.close()

    # Combine content parts with 'text' elements first, then 'table' elements
    combined_content = []
    for e in tqdm(text_content_parts + table_content_parts, desc="Processing elements"):
        combined_content.append(e)

    return combined_content


# CRUD Operations for Rag Data Table==================================================================
def create_rag_data_with_elements(
    version_name: str,
    generation_time: datetime.datetime,
    element_ids: List[int],
    status: str = "generated",
) -> int:
    """
    Creates a new rag_data record by combining content from specified elements.

    :param version_name: Name of the rag_data version
    :param generation_time: Timestamp of when the version was created
    :param element_ids: List of element IDs whose content will be combined for this rag_data version
    :param status: Status of the rag_data version
    :return: The ID of the created rag_data version
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Generate the combined content using the helper function
    raw_data = generate_combined_content(element_ids)
    # Serialize raw_data as a JSON string for TEXT storage
    raw_data_str = json.dumps(raw_data)

    # Generate the embedding
    corpus_emb = emb_encode(raw_data)

    # Serialize numpy array to binary format
    emb_data = pickle.dumps(corpus_emb)

    # Insert into rag_data
    # NOTE for raw_data when query need raw_data_list = json.loads(rag_data_details['raw_data'])
    #      for emb_data when query need corpus_emb = pickle.loads(rag_data_details['emb_data'])
    #      and                          corpus_emb_tensor = torch.tensor(corpus_emb)
    cursor.execute(
        """INSERT INTO rag_data (version_name, raw_data, emb_data, generation_time, status)
                      VALUES (?, ?, ?, ?, ?)""",
        (version_name, raw_data_str, emb_data, generation_time, status),
    )
    rag_data_id = cursor.lastrowid

    # Link each element to the new rag_data in rag_data_elements_link
    for element_id in element_ids:
        cursor.execute(
            """INSERT INTO rag_data_elements_link (rag_data_id, element_id)
                          VALUES (?, ?)""",
            (rag_data_id, element_id),
        )

    conn.commit()
    conn.close()

    return rag_data_id


def create_rag_data_with_files(
    version_name: str,
    generation_time: datetime.datetime,
    file_ids: List[int],
    status: str = "generated"
) -> int:
    """
    Creates a rag_data entry by combining elements from specified files and links it to those files.

    :param version_name: Name of the rag_data version
    :param generation_time: Timestamp of when the version was created
    :param file_ids: List of file IDs to gather elements from and link with this rag_data record
    :param status: Status of the rag_data version (default is "generated")
    :return: The ID of the created rag_data version
    """
    # Gather all element IDs from the specified file IDs
    element_ids = []
    for file_id in file_ids:
        element_ids.extend(get_element_ids_by_file_id(file_id))

    # Create the rag_data entry using the gathered element IDs
    rag_data_id = create_rag_data_with_elements(
        version_name=version_name,
        generation_time=generation_time,
        element_ids=element_ids,
        status=status
    )

    return rag_data_id


def get_rag_data(rag_data_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rag_data WHERE id = ?", (rag_data_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_rag_data(rag_data_id: int) -> bool:
    """
    Deletes a rag_data entry and its associated links in Rag Data Elements Link Table
    and Rag Data Files Link Table.

    :param rag_data_id: The ID of the rag_data entry to delete
    :return: True if the rag_data entry and its links were deleted, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Delete associated links in Rag Data Elements Link Table
    cursor.execute("DELETE FROM rag_data_elements_link WHERE rag_data_id = ?", (rag_data_id,))

    # Delete the rag_data entry itself
    cursor.execute("DELETE FROM rag_data WHERE id = ?", (rag_data_id,))
    deleted = cursor.rowcount > 0  # Check if rag_data was deleted

    conn.commit()
    conn.close()

    return deleted


def remove_elements_from_rag_data(version_name: str, element_ids: List[int]) -> bool:
    """
    Removes specified elements from a rag_data's content based on element_ids
    and updates the rag_data content by re-generating it using remaining elements.

    :param version_name: The version_name of the rag_data to update
    :param element_ids: A list of element IDs to remove from the rag_data
    :return: True if the elements were removed and the rag_data updated, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM rag_data WHERE version_name = ?", (version_name,))
    rag_data_row = cursor.fetchone()

    if not rag_data_row:
        conn.close()
        return False

    rag_data_id = rag_data_row["id"]

    # Check if each element_id is linked to this rag_data_id and delete links if found
    element_ids_to_remove = []
    for element_id in element_ids:
        cursor.execute(
            "SELECT element_id FROM rag_data_elements_link WHERE rag_data_id = ? AND element_id = ?",
            (rag_data_id, element_id),
        )
        if cursor.fetchone():  # Link exists
            element_ids_to_remove.append(element_id)

    if not element_ids_to_remove:
        conn.close()
        return False

    # Delete links in rag_data_elements_link for specified element_ids
    cursor.executemany(
        "DELETE FROM rag_data_elements_link WHERE rag_data_id = ? AND element_id = ?",
        [(rag_data_id, eid) for eid in element_ids_to_remove],
    )

    # Fetch all remaining element IDs linked to this rag_data_id
    cursor.execute(
        "SELECT element_id FROM rag_data_elements_link WHERE rag_data_id = ?",
        (rag_data_id,),
    )
    remaining_element_ids = [row["element_id"] for row in cursor.fetchall()]

    # Re-generate the combined content using the remaining elements
    updated_content = generate_combined_content(remaining_element_ids)

    # Update the content in rag_data
    cursor.execute(
        "UPDATE rag_data SET raw_data = ? WHERE id = ?", (updated_content, rag_data_id)
    )

    conn.commit()
    conn.close()

    return True


def delete_element_from_all_rag_data(element_id: int) -> None:
    """
    Deletes all links of a specific element_id in rag_data_elements_link
    and updates the associated rag_data to remove the element's content.
    The content of each affected rag_data is re-generated using the remaining elements.

    :param element_id: The ID of the element to remove from all rag_data
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Find all rag_data that link to this element_id
    cursor.execute(
        "SELECT DISTINCT rag_data_id FROM rag_data_elements_link WHERE element_id = ?",
        (element_id,),
    )
    rag_data_ids = [row["rag_data_id"] for row in cursor.fetchall()]

    # Delete all links for the specified element_id in rag_data_elements_link
    cursor.execute(
        "DELETE FROM rag_data_elements_link WHERE element_id = ?", (element_id,)
    )

    # Update the content of each affected rag_data
    for rag_data_id in rag_data_ids:
        # Fetch all remaining element IDs linked to this rag_data_id
        cursor.execute(
            "SELECT element_id FROM rag_data_elements_link WHERE rag_data_id = ?",
            (rag_data_id,),
        )
        remaining_element_ids = [row["element_id"] for row in cursor.fetchall()]

        # Re-generate the combined content using the remaining elements
        updated_content = generate_combined_content(remaining_element_ids)

        # Update the rag_data content
        cursor.execute(
            "UPDATE rag_data SET raw_data = ? WHERE id = ?",
            (updated_content, rag_data_id),
        )

    conn.commit()
    conn.close()


def remove_files_from_rag_data(version_name: str, file_ids: List[int]) -> bool:
    """
    Removes all elements associated with the specified file IDs from the specified rag_data version.

    :param version_name: The version name of the rag_data to update
    :param file_ids: List of file IDs whose elements should be removed from the rag_data
    :return: True if elements were removed and content re-generated, False otherwise
    """
    # Gather all element IDs associated with the specified file IDs
    element_ids_to_remove = []
    for file_id in file_ids:
        element_ids_to_remove.extend(get_element_ids_by_file_id(file_id))

    # Remove these elements from the specified rag_data version
    return remove_elements_from_rag_data(version_name, element_ids_to_remove)


def delete_files_from_all_rag_data(file_ids: List[int]) -> None:
    """
    Deletes elements associated with specified files from all rag_data versions.

    :param file_ids: List of file IDs whose elements should be removed from all rag_data versions
    """
    # Collect all element IDs associated with the provided file IDs
    element_ids_to_remove = []
    for file_id in file_ids:
        element_ids_to_remove.extend(get_element_ids_by_file_id(file_id))

    # Remove each element from all rag_data entries where it appears
    for element_id in element_ids_to_remove:
        delete_element_from_all_rag_data(element_id)

    print("Specified files' elements removed from all rag_data versions.")


def list_rag_data_versions(limit: Optional[int] = None, order_by: str = "generation_time", descending: bool = True) -> List[Dict[str, Any]]:
    """
    Lists all rag_data versions with optional limit and ordering.

    :param limit: Maximum number of versions to retrieve (optional)
    :param order_by: Column to order the results by (default is "generation_time")
    :param descending: Whether to order in descending order (default is True)
    :return: A list of dictionaries containing rag_data records
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Determine the order direction
    order_direction = "DESC" if descending else "ASC"
    # Apply limit if provided
    limit_clause = f"LIMIT {limit}" if limit else ""

    # SQL query to retrieve rag_data records with sorting and limit
    query = f"SELECT * FROM rag_data ORDER BY {order_by} {order_direction} {limit_clause}"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    # Convert each row to a dictionary and return the list
    return [dict(row) for row in rows]

# CRUD Operations for Rag Data Elements Link Table==========================================================
def create_rag_data_elements_link(rag_data_id: int, element_id: int) -> int:
    """
    Creates a link between a rag_data entry and an element in the rag_data_elements_link table.

    :param rag_data_id: The ID of the rag_data entry to link
    :param element_id: The ID of the element to link
    :return: The ID of the created link record
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO rag_data_elements_link (rag_data_id, element_id)
                      VALUES (?, ?)""",
        (rag_data_id, element_id),
    )
    conn.commit()
    link_id = cursor.lastrowid
    conn.close()
    return link_id


def get_rag_data_elements_link(link_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific link from the rag_data_elements_link table by its ID.

    :param link_id: The ID of the link to retrieve
    :return: A dictionary containing the link data, or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rag_data_elements_link WHERE id = ?", (link_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_rag_data_elements_link(link_id: int) -> bool:
    """
    Deletes a specific link from the rag_data_elements_link table by its ID.

    :param link_id: The ID of the link to delete
    :return: True if the link was deleted, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM rag_data_elements_link WHERE id = ?", (link_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted
