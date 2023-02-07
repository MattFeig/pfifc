import pandas as pd
import os
import numpy as np

def data_loader(sub_demo_list):
   
    """Returns an roi x roi x participant np array, 
    for the particpants specified in the input subject demographic list"""

    conn_directory = '../data/connectivity_data'
    conn = []

    for index, row in sub_demo_list.iterrows():
        sub = row.VC
        
        # Check which type of subject ID is in the demo list. 
        # Depending on the study ID, there corresponding connectivity csv path will need a different name formatting
    
        if (sub[4:9] == 'MSCPI') | (sub[4:7] == 'NDA') | (sub[4:8] == 'LoTS'):
            csv = f'{sub}_Gordon_Subcort_0.2FDcens_CONCAT_ROIordered_zmat.csv'
        elif sub[4:6] == 'NT':
            csv = f'{sub}ses-screen_task-rest_DCANBOLDProc_v4.0.0_Gordon_subcorticals_0.2_5contig_FD_zmat.csv'  
        elif (sub[:2] == 'vc') | (sub[:2] == 'VC') | (sub[:3] == 'NIC') | (sub[:4] == 'SAIS') :
            # These need more work to get proper formatting. Some hardcoding for 2 partcipants for conveniece
            if '_2' in sub:
                sub = sub.replace('_2', 'V2')
            if '_' in sub:
                sub = sub.replace('_', '')   
            if sub == 'SAISV209':
                sub = 'SAIS209'
            if sub == 'vctb0053east':
                sub = 'TB0053E'                
            sub = sub.upper()
            csv = f'sub-{sub}_Gordon_Subcort_0.2FDcens_CONCAT_ROIordered_zmat.csv'
        else:
            print(sub) 
            break

        filepath = os.path.join(conn_directory, csv)

        if os.path.exists(filepath):
            sub_conn = np.genfromtxt(filepath, delimiter=',')
            conn.append(sub_conn)
        else:
            print(sub)
            break
    
    return np.stack(conn,2)