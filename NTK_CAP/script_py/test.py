from mmpose.apis import MMPoseInferencer

img_path = r'C:\Users\user\Desktop\NTKCAP\Patient_data\snowboard\2024_05_23\raw_data\Apose\videos\3.mov'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('body26')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
results = [result for result in result_generator]