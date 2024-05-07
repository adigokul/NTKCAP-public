from bs4 import BeautifulSoup
import os
def halpe26_xml_update(dir_Empty_prj):
    scaling = os.path.join(dir_Empty_prj,'opensim','Scaling_Setup_Pose2Sim_Halpe26.xml')
    IK = os.path.join(dir_Empty_prj,'opensim','IK_Setup_Pose2Sim_Halpe26.xml')
    #### scaling_update
    
    with open(scaling, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, 'xml')
    for tag in bs_data.find_all('model_file'):
        tag.string = 'Model_Pose2Sim_Halpe26.osim'
    for tag in bs_data.find_all('marker_file'):
        tag.string='Empty_project_filt_0-30.trc'
    for tag in bs_data.find_all('output_model_file'):
        tag.string='Model_Pose2Sim_Halpe26_scaled.osim'
    for tag in bs_data.find_all('time_range'):
        tag.string ='0.0166667 2.5'
        
    with open(scaling,'w') as f:
        f.write(bs_data.prettify())
    #### IK_update
    

    with open(IK, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, 'xml')
    for tag in bs_data.find_all('model_file'):
        tag.string ='Model_Pose2Sim_Halpe26_scaled.osim'
    for tag in bs_data.find_all('marker_file'):
        tag.string = 'Empty_project_filt_0-30.trc'
    for tag in bs_data.find_all('output_motion_file'):
        tag.string = 'Balancing_for_IK.mot'
    for tag in bs_data.find_all('time_range'):
        tag.string ='0.016666666666666601 10000'
    with open(IK,'w') as f:
        f.write(bs_data.prettify())