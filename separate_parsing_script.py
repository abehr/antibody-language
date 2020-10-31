import os, gzip, json

def list_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def write_full_seqs_to_csv(data_directory, csv_filename_light, csv_filename_heavy):
    with open(csv_filename_light, 'w') as file_light:
        with open(csv_filename_heavy, 'w') as file_heavy:
            for f in list_file_paths(data_directory):
                heavy = False
                light = False
                meta_line = True
                for line in gzip.open(f, 'rb'):
                    if meta_line:
                        metadata = json.loads(line)
                        meta_line = False
                        if metadata['Chain']=='Heavy':
                            heavy = True
                        elif metadata['Chain']=='Light':
                            light = True
                        continue
                    basic_data = json.loads(line)
                    full_seq = basic_data['seq']
                    if heavy:
                        file_heavy.write(full_seq + '\n')
                    elif light:
                        file_light.write(full_seq + '\n')

if __name__ == '__main__':
    # data_directory = 'OAS_Data'
    data_directory = 'TW05B_IGHE_' #'Rubelt_TW05_'
    # csv_filename = 'full_seq_data.csv'
    csv_filename_light = 'full_light_seq_data.csv'
    csv_filename_heavy = 'full_heavy_seq_data.csv'
    write_full_seqs_to_csv(data_directory, csv_filename_light, csv_filename_heavy)
