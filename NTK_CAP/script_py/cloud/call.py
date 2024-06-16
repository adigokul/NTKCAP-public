import gltfConverter as gltfConver


def main():
    import argparse
    print('main was called')
    ## Input parsing.
    ## =============
    parser = argparse.ArgumentParser(
        description="Generate a gltf file corresponding to the passed in osim file with mot files.")
    # Required arguments.
    parser.add_argument('osim_file_path',
                        metavar='osimfilepath', type=str,
                        help="filename for osim file (including path).")
    parser.add_argument('mot_file_paths',
                        metavar='motfilepath', type=str, nargs='+',
                        help="filenames for mot files (including path).")
    
    parser.add_argument('--output', type=str,
                        help="Write the result to this filepath. "
                             "Default: the report is named "
                             "<_file_path>.gltf")
    
    args = parser.parse_args()
    print(args)
    osimModel = args.osim_file_path
    motions = args.mot_file_paths
    # 傳入 osim 檔案 以及 motion 的 mot 檔案
    gltfConver.build(osimModel, motions, args.output)

main()
