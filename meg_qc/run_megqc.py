import argparse
import os
import sys
import shutil

def run_megqc():
    from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc

    dataset_path_parser = argparse.ArgumentParser(description= "parser for MEGqc: --inputdata(mandatory) path/to/your/BIDSds --config path/to/config  if None default parameters are used)")
    dataset_path_parser.add_argument("--inputdata", type=str, required=True, help="path to the root of your BIDS MEG dataset")
    dataset_path_parser.add_argument("--config", type=str, required=False, help="path to config file")
    args=dataset_path_parser.parse_args()

    path_to_megqc_installation= os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))

    #parent_dir = os.path.dirname(os.getcwd())
    #print(parent_dir)
    print(path_to_megqc_installation)

    data_directory = args.inputdata
    print(data_directory)

    if args.config == None:
        url_megqc_book = 'https://aaronreer.github.io/docker_workshop_setup/settings_explanation.html'
        text = 'The settings explanation section of our MEGqc User Jupyterbook'

        print('You called the MEGqc pipeline without the optional \n \n --config <path/to/custom/config>  argument. \n \n MEGqc will proceed with the default parameter settings. Detailed information on the user parameters in MEGqc and their default values can be found in here: \n \n')
        print(f"\033]8;;{url_megqc_book}\033\\{text}\033]8;;\033\\")
        print("\n \n")
        user_input = input('Do you want to proceed with the default settings? (y/n): ').lower().strip() == 'y' 
        if user_input == True:
            config_file_path = path_to_megqc_installation + '/settings/settings.ini'
        else:
            print("Use the \n \n get-megqc-config --target_directory <path/to/your/target/directory> \n \n 2command line prompt. This will copy the config file to a target destination on your machine.YOu can edit this file, e.g adjust all user parameters to your needs, and run the pipeline command again \n run-megqc \n with the \n --config parameter \n providing a path to your customized config file") 

    else:
        config_file_path = args.config

    internal_config_file_path=path_to_megqc_installation + '/settings/settings_internal.ini'

    make_derivative_meg_qc(config_file_path, internal_config_file_path, data_directory)


    print('MEGqc has completed the calculation of metrics. Results can be found in' + data_directory +'/derivatives/MEGqc/calculation')

    user_input = input('Do you want to run the MEGqc plotting module on the MEGqc results? (y/n): ').lower().strip() == 'y'

    if user_input == True:
        make_plots_meg_qc(data_directory)
        return
    else:
        return
