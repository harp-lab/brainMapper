import json
import sys
import re
import os

def count_branches(edges):
    incident_count = dict()
    for u, v in edges.items():
        if u in incident_count:
            incident_count[u] += 1
        else:
            incident_count[u] = 0
    
    branch_count = 0
    for v in incident_count.values():
        if(v > 2):
            branch_count += 1

    return branch_count

def get_nodes(data):
    nodes = []
    for node in data['nodes'].keys():
        nodes.append(node)    
    return nodes

def get_edges(data):
    return data['edges']

def extract_from_path_name(file_path):
    
    extracted_info={
        "graph_dir":None,
        "graph_file":None,
        "clusterer":None,
        "epsilon":None,
        "minpts":None,
        "filter":None,
        "interval":None,
        "overlap":None
    }

    # Given input string with single backslashes
    input_string = file_path

    # Use regex to extract the graph_dir and graph_file
    # Match the graph_dir pattern
    graph_dir_match = re.search(r'/([^/]+_eps_\d+_minpts_\d+_\w+)/', input_string)
    # Match the graph_file pattern
    graph_file_match = re.search(r'/([^/]+\.csv_\d+_\d+\.json)$', input_string)

    if graph_dir_match and graph_file_match:
        # Extract the graph_dir and graph_file
        graph_dir = graph_dir_match.group(1)  # e.g., 'dbscan_eps_20_minpts_5_l2norm'
        graph_file = graph_file_match.group(1)  # e.g., 'mapper_act_brain.csv_30_40.json'
        # print("GRAPH_DIR=", graph_dir)
        # print("GRAPH_FILE=",graph_file)
        # sys.exit(1)        
        # Use regex to extract parameters from graph_dir
        param_match = re.match( r'(?P<file_prefix>[\w.]+)_(?P<clusterer>\w+)_eps_(?P<eps>\d+)_minpts_(?P<minpts>\d+)_filter_(?P<filter>\w+)', graph_dir)

            
        if param_match:
            clusterer = param_match.group('clusterer')  # e.g., 'dbscan'
            epsilon = int(param_match.group('eps'))          # e.g., 20
            minpts = int(param_match.group('minpts'))    # e.g., 5
            filter_type = param_match.group('filter')     # e.g., 'l2norm'
        else:
            raise ValueError("Could not extract parameters from graph_dir")

        # Use regex to extract interval and overlap from graph_file
        file_match = re.match(r'(?P<file_prefix>[\w.]+)_(?P<interval>\d+)_(?P<overlap>\d+)\.json', graph_file)

        if file_match:
            interval = int(file_match.group('interval'))  # e.g., 30
            overlap = int(file_match.group('overlap'))      # e.g., 40
        else:
            raise ValueError("Could not extract interval and overlap from graph_file")

        # Print the extracted values
        extracted_info['graph_dir']=graph_dir
        extracted_info['graph_file']=graph_file
        extracted_info['clusterer']=clusterer
        extracted_info['epsilon']=epsilon
        extracted_info['minpts']=minpts
        extracted_info['filter']=filter_type
        # print(filter_type)
        extracted_info['interval']=interval
        extracted_info['overlap']=overlap
        
        return extracted_info
    else:
        print(f"{file_path}: Could not find the necessary components in the input string.")
        sys.exit(1)


    
def extract_info(file_path):
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    nodes = get_nodes(data)
    edges = get_edges(data)

    n_branch = count_branches(edges)

    extracted_info = extract_from_path_name(file_path)
    extracted_info['nodes'] = len(nodes)
    extracted_info['edges'] = len(edges)
    extracted_info['n_branch'] = n_branch
    
    return extracted_info
    # entry = f"{graph_name},{interval},{overlap},{epsilon},{min_sample},{n_branch}"
    

def log_info(data_rows, log_file):
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    


    # Check if data_rows is a list of dictionaries
    if isinstance(data_rows, list) and all(isinstance(row, dict) for row in data_rows):
        # Get headers from the keys of the first dictionary
        headers = data_rows[0].keys()
        
        # Write headers to the log file if the file is new
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, 'a') as file:  # Use 'a' to append instead of 'w'
            if not file_exists:
                # Write header line only if the file is new
                file.write(','.join(headers) + '\n')
                
            # Write each row of data
            for row in data_rows:
                values = ','.join(str(row.get(header, '')) for header in headers)
                file.write(values + '\n')
    else:
        raise ValueError("data_rows must be a list of dictionaries.")

if __name__=="__main__":

    if len(sys.argv) < 4: 
        print("Usage: python3 extract_graph.info.py [graph_dir] [log_dir] [mapper_spec]")
    
        
        # print(sys.argv[3])
        # print(sys.argv[4])
        sys.exit(1)  # Exit the script with a non-zero status code

    graph_dir = sys.argv[1]
    log_dir = sys.argv[2]
    mapper_spec = sys.argv[3]

    graph_source = f"{graph_dir}/{mapper_spec}"
    log_file = f"{log_dir}/{mapper_spec}.out"

    extracted_rows = []
    for filename in os.listdir(graph_source):
        if 'mapper' in filename:
            file_path = os.path.join(graph_source, filename)
            if os.path.isfile(file_path):  # Check if it's a file
                extracted_rows.append(extract_info(file_path))
    
    # print(extracted_rows)
    log_info(extracted_rows, log_file)