import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/app/Fall25-FieldSession-Stratom/install/bag_reader_node_py'
