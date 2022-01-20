import streamlit as st
import os
import dataclasses

from typing import List, Optional

@dataclasses.dataclass
class Artifacts:
    final_model: str
    mode: str
    model_path: str
    uploaded_args_to_names: Optional[None]
    uploaded_images: List[Optional[None]]
    uploaded_labels: Optional[None]

    def __iter__(self):
        return [self.final_model, self.mode, self.model_path,
                self.uploaded_images, self.uploaded_labels, self.uploaded_args_to_names]


def write_to_file(uploadedfile):
    # Code credits: https://blog.jcharistech.com/2021/01/21/how-to-save-uploaded-files-to-directory-in-streamlit-apps/
    path_to_file = os.path.join("C:\\Users\\adity\\temp_streamlit", uploadedfile.name)
    with open(path_to_file, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return path_to_file

