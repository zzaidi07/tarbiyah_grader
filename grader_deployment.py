import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader
import re
import copy
import csv
import os, fnmatch
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px

def parse_roadmap(roadmap_file = 'roadmap.pdf'):
    
    reader = PdfReader(roadmap_file)
    pattern = r'\d[A-Z]-\d{2}'
    roadmap_matches = []
    page = reader.pages[1]
    for page in reader.pages:
        text = page.extract_text()
        roadmap_matches += re.findall(pattern, text)

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
        incomplete_check = text_lesson.find("incomplete")
        matches = re.findall(r'\b\d+/\d+\b', text_lesson)
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



def write_into_csv(csv_fname, report_date, student_name,
                   roadmap_matches, test_total, test_marks
                  ):

    with open(csv_fname, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([report_date])
        writer.writerow([student_name])
        writer.writerow(['Module Lesson', 'Total', 'Score', 'Percentage'])  # Header row
        for n, t, s in zip(roadmap_matches, test_total, test_marks):
            writer.writerow([n, t, s, 100*s/t])
        writer.writerow(['Total', np.sum(test_marks/test_total)])

def main():
    # Intro Messages
    st.title("Tarbiyah Grader")
    st.text("Salaamun Alaykum, welcome to the unofficial tarbiyah-grader")
    st.text("Previously, teachers had to laboriously go through each student report, use only specific modules from roadmap, and copy the test results of each lesson separately")
    st.text("Now, once the roadmap and student reports are uploaded, all this can be done with just a click of a button")
    
    # Upload roadmap
    uploaded_roadmap = st.file_uploader("Upload Roadmap", accept_multiple_files = False, type = "pdf")
    # Upload student reports
    std_fnames = st.file_uploader("Upload (Multiple) Student PDF reports", accept_multiple_files=True, type="pdf")
    
    if (st.button("Analyze")):
        # Parse roadmap
        module_names, lessons, roadmap_matches = parse_roadmap(uploaded_roadmap)
        total_lessons = int(len(roadmap_matches))
        names = []
        scores = np.zeros([len(std_fnames), len(module_names)])
        totals = []
        frac_completed_modules = np.zeros(len(std_fnames))
        
        for std_ind, std_fname in enumerate(std_fnames):
            csv_fname = std_fname.name.strip('.pdf') + '.csv'
            
            # Parse student report and extract relevant information
            report_date, student_name, test_total, test_marks = parse_student_record(std_fname, module_names, lessons)
            
            # Store relevant information for use later in table/plot/csv
            names += [student_name]
            scores[std_ind,:] = 100 * test_marks/test_total
            totals += [100 * np.sum(test_marks/test_total) / len(lessons)]
            num_missed = len(np.argwhere(test_marks == 0))

            frac_completed_modules[std_ind] = 100 * (total_lessons - num_missed )/ total_lessons
            write_into_csv(csv_fname, report_date, student_name,roadmap_matches, test_total, test_marks)
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
        
#        fig = px.histogram(pd.DataFrame({"A":[1,1,1,2,2,3,3,3,4,4,4,5]}),x="A", 
#text_auto=True)
#fig.show()
        fig = px.histogram(plot_df)

#        fig = px.histogram(frac_completed_modules, x = "% Module Completion", y = "# Students")

        # Plot!
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

    
if __name__ == "__main__":
    main()