import streamlit as st
import matplotlib.pyplot as plt
from pypdf import PdfReader
import re
import copy
import os, fnmatch
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
from stqdm import stqdm

    

def parse_roadmap(roadmap_file = 'roadmap.pdf'):
    reader = PdfReader(roadmap_file)
    consolidated_roadmap = ''
    date_matches = []
    for page in range(reader.get_num_pages()):
        consolidated_roadmap += reader.pages[page].extract_text()
    date_pattern = r'\b[A-Za-z]+(?:\. ?| ?\.?| )?\d+\/'
    date_matches += re.findall(date_pattern, consolidated_roadmap)
    refined_date_matches = [] 
    assert "Plan" in date_matches[0]
    for match in date_matches:
        if "Plan" not in match:
            refined_date_matches += [match.split('/')[0]]
    return consolidated_roadmap, refined_date_matches

def extract_roadmap_modules(consolidated_roadmap, start_date, end_date):
    start_date_line = consolidated_roadmap.find(start_date)
    end_date_line = consolidated_roadmap.find(end_date)
    filtered_roadmap = consolidated_roadmap[start_date_line : end_date_line]
    module_pattern = r'\d[A-Z]-\d{2}'
    roadmap_matches = re.findall(module_pattern, filtered_roadmap)
    lessons = []
    module_names = []

    for roadmap_names in roadmap_matches:
        module_names += ['BAND ' + roadmap_names[1] + ' MODULE ' + roadmap_names[0]]
        lessons += ['Lesson ' + str(int(roadmap_names[3:5]))]
    return module_names, lessons, roadmap_matches


def parse_student_record(record_name, module_names, lessons):
    reader = PdfReader(record_name)
    consolidated_student_record = ''
    for page in range(reader.get_num_pages()):
        consolidated_student_record += reader.pages[page].extract_text()
    report_date = consolidated_student_record.splitlines()[0]
    student_name = consolidated_student_record.splitlines()[2]
    assert "AM" in student_name
    test_marks = np.zeros(len(module_names))
    test_total = np.zeros(len(module_names))
    for lesson_ind, lesson in enumerate(lessons):

        # 1. Isolate the module
        current_module_name = module_names[lesson_ind]
        next_module_name = current_module_name[:-1] + str(int(current_module_name[-1]) + 1)
        current_module_ln = consolidated_student_record.find(current_module_name)
        next_module_ln = consolidated_student_record.find(next_module_name)
        text_module = consolidated_student_record[current_module_ln:next_module_ln]
        next_lesson = lesson[:-1] + str(int(lesson[-1]) + 1)
        current_lesson_ln = text_module.find(lesson)
        next_lesson_ln = text_module.find(next_lesson)
        text_lesson = text_module[current_lesson_ln:next_lesson_ln]
        matches = re.findall(r'\b\d+/\d+\b', text_lesson)
        
        if len(matches) != 0:
            match_line = text_lesson.find(matches[-1])
            incomplete_check = text_lesson[match_line:].find("incomplete")
        else:
            incomplete_check = -1
        
        if len(matches) == 0 or incomplete_check != -1:
            test_total[lesson_ind] = 5
            test_marks[lesson_ind] = 0
        else:

            score = matches[-1]
            test_total[lesson_ind] = score[-1]
            test_marks[lesson_ind] = score[0]
    
    return report_date, student_name, test_total, test_marks

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def main():
    # Intro Messages
    st.title("Tarbiyah Grader")
    st.text("Salaamun Alaykum, welcome to the unofficial tarbiyah-grader")
    st.text("Previously, teachers had to laboriously go through each student report, use only specific modules from roadmap, and copy the test results of each lesson separately")
    st.text("Now, once the roadmap and student reports are uploaded, all this can be done with just a click of a button")
    
    # Upload roadmap
    uploaded_roadmap = st.file_uploader("Upload Roadmap", accept_multiple_files = False, type = "pdf")

    if 'consolidated_roadmap' not in st.session_state:
        st.session_state.consolidated_roadmap = None
    
    if 'refined_date_matches' not in st.session_state:
        st.session_state.refined_date_matches = None
        
    if 'sbox_index' not in st.session_state:
        st.session_state.sbox_index = 0
    
    if (st.button("Analyze Roadmap")):
        try:
            consolidated_roadmap, refined_date_matches = parse_roadmap(uploaded_roadmap)
            st.session_state.consolidated_roadmap = consolidated_roadmap
            st.session_state.refined_date_matches = refined_date_matches
            st.session_state.sbox_index = len(refined_date_matches) - 1 
            st.text("Roadmap Analyzed")
            
        except:
            st.text("Sorry, we couldn't process the roadmap. Please ensure you have uploaded the correct roadmap file. We encourage you to submit a bug report to: Tarbiyyaguide.sun@al-muntadhir.ca")

                    
    
    start_date = st.selectbox("Analysis Start Date", st.session_state.refined_date_matches)
    end_date = st.selectbox("Analysis End Date (excluding)", st.session_state.refined_date_matches, 
                            index = st.session_state.sbox_index)
    

    

     
    
    # Upload student reports
    std_fnames = st.file_uploader("Upload (Multiple) Student PDF reports", accept_multiple_files=True, type="pdf")
    
        
    if (st.button("Compile Result")):
        
        # Parse roadmap       
        module_names, lessons, roadmap_matches = extract_roadmap_modules(st.session_state.consolidated_roadmap, start_date, end_date)        
        
        total_lessons = int(len(roadmap_matches))
        names = []
        scores = np.zeros([len(std_fnames), len(module_names)])
        totals = []
        frac_completed_modules = np.zeros(len(std_fnames))
        
        try: 
            for std_ind, std_fname in enumerate(stqdm(std_fnames)):
                csv_fname = std_fname.name.strip('.pdf') + '.csv'

                # Parse student report and extract relevant information
                report_date, student_name, test_total, test_marks = parse_student_record(std_fname, module_names, lessons)

                # Store relevant information for use later in table/plot/csv
                names += [student_name]
                scores[std_ind,:] = 100 * test_marks/test_total
                totals += [100 * np.sum(test_marks/test_total) / len(lessons)]
                num_missed = len(np.argwhere(test_marks == 0))

                frac_completed_modules[std_ind] = 100 * (total_lessons - num_missed )/ total_lessons

            totals = np.array(totals)

            df_full_data = {'Name': names}
            df_full_data['Total'] = totals  # Add the 'Total' column
            for ind_module, module in enumerate(roadmap_matches):
                df_full_data[module] = scores[:,ind_module]

            df_full = pd.DataFrame(df_full_data)



            df_disp = {'Name': names}
            df_disp['Total Score'] = (totals)  # Add the 'Total' column
            df_disp['% Module Completed'] = frac_completed_modules  # Add the 'Total' column

            pd.set_option('display.float_format', '{:10.2f}'.format)

            df_disp = pd.DataFrame(df_disp)
            df_disp.style.format(precision=0)
            st.table(df_disp)

            plot_df = pd.DataFrame({"Percentage Module Completion": frac_completed_modules})


            fig = px.histogram(plot_df, nbins = 5, barmode = 'group', title = 'Module Completion %')


            st.text("Visualization of completed modules in your class")

            st.plotly_chart(fig)

            csv = convert_df(df_full)

            st.text("You may now download the compiled student result")

            st.download_button(
               "Download Compiled Result",
               csv,
               "compiled_result.csv",
               "text/csv",
               key='download-csv'
            )

            st.text("Privacy notice: The uploaded and generated data is deleted once page is refreshed")
        except:
            st.text("Sorry, we couldn't process the student reports. Please ensure you have uploaded the correct file. We encourage you to submit a bug report to: Tarbiyyaguide.sun@al-muntadhir.ca")
            
    
if __name__ == "__main__":
    main()
