from collections import OrderedDict, namedtuple
import datetime
import os
import numpy as np
import pandas as pd
import pickle
import gzip
import csv
import time
import copy

"""
This module was built during the publicly funded Australian Research Council DP150100962 project at the University of
Melbourne, Australia. It includes a set of Classes to conduct Structural Path Analysis (SPA) on input-output and process
data using environmental, social or financial satellites. 
"""

__authors__ = 'Andre Stephan (ORCID: 0000-0001-9538-3830), ' \
              'Paul-Antoine Bontinck (ORCID: 0000-0002-4072-1334)'
__version__ = '1.4'
__status__ = 'Release version'

# global variables, used notably when writing csv files
INITIAL_HEADER_LIST = [
    '% of total intensity: ',
    'direct intensity of last node in pathway',
    'total intensity of pathway'
]

flow = namedtuple('flow', 'name, unit')  # declare here to enable pickling of referencing objects


def _get_clean_filename(name: str) -> str:
    """
    Removes all unacceptable characters from a name
    :param name: a string to be processed
    :return: a filename str
    """
    for char in "\/:*?<>|":
        name = name.replace(char, '')
    return name


def _save_pickle(data, filename: str = None, directory: str = None, file_extension: str = None, path: str = None):
    """
    Use pickle/gzip to compress and save data
    :param data: data to save
    :param filename: name given to the pickle file, without the extension
    :param directory: directory where file will be saved
    :param file_extension: extension of the file to be saved
    :param path: the direct path to the file
    :return:
    """
    if path is None:
        path = directory + '\\' + filename + '.' + file_extension

    with gzip.open(path, 'wb', compresslevel=9) as file_data:
        pickle.dump(data, file_data, protocol=4)
    print('File saved at ' + path)
    return True


def _load_pickle(filename: str = None, directory: str = None, file_extension: str = None, path: str = None):
    """
    Use pickle/gzip to load a compressed file
    :param filename: name of pickle file to open, without the extension
    :param directory: directory where file will be found
    :param file_extension: extension of the file to be loaded
    :param path: the direct path to the file
    :return: a variable containing the loaded data
    """
    if path is None:
        path = directory + '\\' + filename + '.' + file_extension

    with gzip.open(path, 'rb', compresslevel=9) as file_data:
        loaded_data = pickle.load(file_data)
    return loaded_data


def load_instance_from_file(path: str, type_: type):
    """
    Unpickles a file and initiates an instance of an object based on the loaded vars
    :param path: the path to the pickled file
    :param type_: the type of object to load, e.g. a Supply Chain object
    :return: the loaded object
    """
    try:
        loaded = _load_pickle(path=path)
        loaded_instance = type_(**loaded)
        print('Successfully loaded the file ' + path + ' as an instance of ' + str(type_))
        return loaded_instance
    except TypeError:
        raise TypeError('Could not load the file ' + path + ' as an instance of ' + str(type_))


class Interface:
    """
    Handles the creation of the supply chain object from csv files
    """

    def __init__(self, a_matrix_file_path: str, infosheet_file_path: str, thresholds_file_path):
        """
        Instantiates the Interface class by providing the paths to all required csv files
        :param a_matrix_file_path:
        :param infosheet_file_path:
        :param thresholds_file_path:
        """
        print('Reading infosheet...', end='')
        self.infosheet_dataframe = self._read_file(self, infosheet_file_path)
        print('Done')

        print('Extracting names of satellites...', end='')
        self.flows_dict = self._get_flows_dict()
        print('Done')

        print('Reading Thresholds...', end='')
        self.thresholds_dict = self._read_file(self, thresholds_file_path, output='thresholds_dict')
        print('Done')

        print('Reading A matrix...', end='')
        self.a_matrix = self._read_file(self, a_matrix_file_path, output='array')
        print('Done')

        print('Validating read data...', end='')
        self._validate_loaded_files()
        print('Done')

        print('Generating vectors of direct and total multipliers...', end='')
        self.dr_vectors_dict, self.tr_vectors_dict = self._generate_multipliers_dicts()
        print('Done')

        print('------ Ready to conduct the Structural Path Analysis ------')

    @staticmethod
    def _read_file(self, path, output='df'):
        """
        Reads the specified xls or csv file and converts it into a pandas dataframe by default or a numpy array by
        choice
        :param path: Path of the xls file to be read
        :param output: Specify dataframe or array if a numpy array is needed
        :return: ordered dict of ordered dict
        """
        if path.endswith('.csv'):
            if output is 'df':
                read_data = pd.read_csv(path, sep=',', encoding='ISO-8859-1')
            elif output == 'array':
                read_data = pd.read_csv(path, sep=',').values
            elif output == 'thresholds_dict':
                read_data = pd.read_csv(path, sep=',', skiprows=0, index_col=0, squeeze=True).to_dict()
            else:
                raise ValueError('Output must be "df" or "array" or "thresholds_dict"')
        elif path.endswith('.xls') or path.endswith('.xlsx'):
            if os.path.getsize(path) < 10 ** 8:
                if output is 'df':
                    read_data = pd.read_excel(path, sep=',', encoding='ISO-8859-1')
                elif output == 'array':
                    read_data = pd.read_excel(path, sep=',').values
                elif output == 'thresholds_dict':
                    read_data = pd.read_csv(path, sep=',', skiprows=0, index_col=0, squeeze=True).to_dict()
                else:
                    raise ValueError('Output must be "df" or "array" or "thresholds_dict"')
            else:
                read_data = None
                print('The excel file ' + path +
                      ' is too large and will take too long to load. Please convert it to csv and load the csv '
                      'instead')
        else:
            raise KeyError('The file should be either a .csv or .xls file')

        return read_data

    def _validate_loaded_files(self):
        """
        Checks the number of sectors/processes in the A matrix and those in the infosheet dataframe
        Checks that all units have been provided in the sector infosheet
        Checks that the flows entered in the thresholds csv file are the same as those in the infosheet
        :return:
        """

        if self.a_matrix.shape[0] != self.a_matrix.shape[1]:
            raise TypeError('Your A matrix is not square.\n Please try loading a different file')
        else:
            a_matrix_sectors_num = self.a_matrix.shape[0]
            infosheet_dataframe_len = self.infosheet_dataframe.shape[0]

            if a_matrix_sectors_num != infosheet_dataframe_len:
                raise ValueError('Your A matrix contains ' + str(a_matrix_sectors_num)
                                 + ' sectors while your sectors information file contains ' + str(
                    infosheet_dataframe_len) +
                                 ' sectors.\n Please reload matching files')

        satellite_set = set([satellite[3:] for satellite in list(self.infosheet_dataframe.columns.values) if
                             satellite.startswith('DR') or satellite.startswith('TR')])
        for satellite in satellite_set:
            if '(' not in satellite or ')' not in satellite:
                raise TypeError('No unit specified for satellite: ' + satellite)

        if set(self.thresholds_dict.keys()) != set(self.flows_dict.keys()):
            raise ValueError(
                'The flows defined in the infosheet are not the same as those defined in the thresholds file')

        print('The A matrix loaded is square and contains ' + str(
            self.a_matrix.shape[0]) + ' sectors/processes across ' + str(len(
            self._get_regions())) + ' region(s), and is described in the infosheet provided, for ' + str(len(
            self.flows_dict.keys())) + ' satellite(s)...', end='')

    def _get_regions(self, ref_dataframe: pd.DataFrame = None):
        """
        Reads the regions in ref_dataframe and determines which regions are actually included in the  file
        :param ref_dataframe: The infosheet containing all information about sectors/processes
        :return: List of regions
        """

        if ref_dataframe is None:
            ref_dataframe = self.infosheet_dataframe
        return list(set(ref_dataframe['Region'].values))

    def _get_flows_dict(self, ref_dataframe: pd.DataFrame = None) -> dict:
        """
        Reads the keys in ref_dataframe and determines which environmental flows are included in the excel file
        :param ref_dataframe: The ordered dictionary generated by xls_to_dict which contains the information and
        environmental multipliers for each sector
        :return: List of flows
        """

        if ref_dataframe is None:
            ref_dataframe = self.infosheet_dataframe

        attributes_list = list(ref_dataframe.columns.values)

        # generate a dictionary of flows names and flows named tupled by removing 'DR' or 'TR' from the string
        # and extracting the unit in between hyphens '()'
        flows_dict = {
            item[3:item.index('(') - 1]: flow(item[3:item.index('(') - 1].capitalize(),
                                              item[item.index('(') + 1:item.index(')')]) for
            item in attributes_list if (item.startswith('DR_') or item.startswith('TR_'))
        }

        return flows_dict

    def _generate_multipliers_dicts(self, ref_dataframe=None, flows_dict: dict = None):
        """
        Generates the dicts of multipliers based on the reference dataframe and the flows to consider
        :param ref_dataframe: the infosheet dataframe
        :param flows_dict: the dictionary of flows considered
        :return: two dictionaries, one for direct requirements and the other for total reqs. They contain the flow
        as key and a numpy array of intensities/multipliers as value
        """

        if ref_dataframe is None:
            ref_dataframe = self.infosheet_dataframe
        if flows_dict is None:
            flows_dict = self.flows_dict

        dr_dict, tr_dict = {}, {}

        for flow, flow_tuple in flows_dict.items():
            dr_dict[flow] = ref_dataframe['DR_' + flow + '_(' + flow_tuple.unit + ')'].values
            tr_dict[flow] = ref_dataframe['TR_' + flow + '_(' + flow_tuple.unit + ')'].values

        return dr_dict, tr_dict


class Node:
    """
    Represent individual nodes in an input-output supply chain.
    """

    _slots_dict = {
        'index_reference': int,
        'stage': int,
        'direct_intensities': dict,
        'total_intensities': dict,
        'nature': str,
        'metadata': dict,
        'root_node_boolean': bool,
        'unit': str
    }

    __slots__ = list(_slots_dict.keys())

    def __init__(self, index_reference: int, stage: int, direct_intensities: dict,
                 total_intensities: dict, unit: str, nature: str = 'io' or 'process', ):
        """
        Instantiates the node class
        :param index_reference: index reference of the node, insert sector or process ID (ie the numerical value
        associated with the sector or process), based on the information sheet provided by the user
        :param stage: stage of the node in the supply chain, i.e. 0, 1, 2, ..., n
        :param direct_intensities: dictionary containing DIRECT environmental flows as keys (e.g. energy, water, GHG)
        and their intensities as values (e.g. 0.5)
        :param total_intensities: dictionary containing TOTAL environmental flows as keys (e.g. energy, water, GHG)
        and their intensities as values (e.g. 0.5)
        :param nature: nature of the data (Process or IO)
        :param unit: the functional unit of the sector or process represented by that node
        :param region: the geographic region of the node
        :return: a handle to the node object created
        """
        self.index_reference = index_reference  # numerical index from 1 to the total number of sector or processes
        self.stage = stage
        self.direct_intensities = direct_intensities
        self.total_intensities = total_intensities
        self.nature = nature  # IO or Process data
        self.root_node_boolean = False  # used to check if we are dealing with the root node (Stage 0)
        self.unit = unit

        self.metadata = None  # can be added via custom scripts

    def __repr__(self):
        """
        Representation of a Node object when printed.
        :return: a string
        """
        return 'Index reference %s [S%s]' % (self.index_reference, self.stage)

    def __eq__(self, other):
        """
        Compares an instance of node to another by checking the equality of its ID
        :param other: the other object used in the comparison
        :return: bool
        """
        return self.get_node_id() == other.get_node_id()

    def __hash__(self):
        """
        Must be implemented to be able to use nodes in sets
        :return:
        """
        return hash(self.get_node_id())

    def __ne__(self, other):
        """
        Compares an instance of node to another by checking the inequality of its ID
        :param other: the other object used in the comparison
        :return: bool
        """
        return not self == other

    def __setstate__(self, pickled_dict):
        """
        Sets the attributes from a pickled state
        :param pickled_dict: the pickled __dict__
        :return:
        """
        for var in pickled_dict.keys():
            try:
                setattr(self, var, pickled_dict[var])
            except AttributeError:
                pass

    def __getstate__(self):
        """
        Generates a dict of internal variables
        :return: a dict object of internal variables
        """
        dict_to_pickle = {}
        for var in self.__slots__:
            try:
                dict_to_pickle[var] = getattr(self, var)
            except AttributeError:  # accommodates earlier versions of the instance which don't use __slots__
                pass
        return dict_to_pickle

    def get_node_id(self):
        """
        Generates the node ID by compounding its name, region and stage
        :return: node_ID string
        """
        list_of_strings = [str(item) for item in [self.index_reference, self.stage, self.nature]]
        node_id = '_'.join(list_of_strings)
        return node_id

    def get_metadata(self):
        """
        Returns the metadata or description of the node
        :return:
        """
        return self.metadata

    def get_flows(self) -> list:
        """
        Returns a list of the flows stored in the node
        :return: a list of flows
        """
        return list(self.direct_intensities.keys())

    def get_node_attribute(self, ref_dict: dict, attribute: str):
        """
        Returns an attribute of the Node object
        :param ref_dict: dictionary of attributes
        :param attribute: attribute
        :return: a string
        """
        try:
            if self.root_node_boolean and attribute == 'Name':
                return 'DIRECT Stage 0'
            else:
                return ref_dict[self.index_reference][attribute]
        except AttributeError:  # use for pickled nodes that do not use __slots__
            return ref_dict[self.index_reference][attribute]


class Transaction:
    """
    Represents a single transaction between two nodes in a supply chain.
    """

    _slots_dict = {
        'origin_node': Node,
        'target_node': Node,
        'tech_coeff': float,
        'accum_tech_coeff': float,
        'nature': str
    }

    __slots__ = list(_slots_dict.keys())

    def __init__(self, origin_node: "Node Object", target_node: "Node Object", tech_coeff: float,
                 accum_tech_coeff: float, nature='io' or 'process'):
        """
        Instantiates the transaction class
        :param origin_node: Node object from which the transaction originates
        :param target_node: Node object at which the transaction ends
        :param tech_coeff: absolute technological intensity of the transaction, not related to the studied
        node downstream
        :param accum_tech_coeff: technological intensity of the transaction in relation to the studied node downstream
        :param nature: nature of the transaction, either Input-Output or Process
        :return
        """
        self.origin_node = origin_node
        self.target_node = target_node
        self.tech_coeff = tech_coeff
        self.accum_tech_coeff = accum_tech_coeff
        self.nature = nature

    def __repr__(self):
        """
        Representation of a Transaction object when printed.
        :return: a string
        """
        return 'transaction from %s to %s' % (self.origin_node, self.target_node) + '\t' + '/// ' + \
               '[Tech coeff: %s / Accumulated tech coeff: %s]' % (self.tech_coeff, self.accum_tech_coeff)

    def __setstate__(self, pickled_dict):
        """
        Sets the attributes from a pickled state
        :param pickled_dict: the pickled __dict__
        :return:
        """
        for var in pickled_dict.keys():
            try:
                setattr(self, var, pickled_dict[var])
            except AttributeError:
                pass

    def __getstate__(self):
        """
        Generates a dict of internal variables
        :return: a dict object of internal variables
        """
        dict_to_pickle = {}
        for var in self.__slots__:
            try:
                dict_to_pickle[var] = getattr(self, var)
            except AttributeError:  # accommodates earlier versions of the instance which don't use __slots__
                pass
        return dict_to_pickle

    def get_nodes(self, as_list=None):
        """
        Returns the origin and target nodes
        :param as_list: define format
        :return: origin and target node
        """
        if as_list is None:
            return self.origin_node, self.target_node
        else:
            return [self.origin_node, self.target_node]

    def get_stage(self):
        """
        Returns the stage of the transaction
        :return: stage of the origin node
        """
        return self.origin_node.stage


class Pathway:
    """
    Represents a single pathway consisting of a series of nodes and transactions
    """

    _slots_dict = {
        'nodes': list,
        'transactions': list,
        'nature': str,
        'number': int
    }

    __slots__ = list(_slots_dict.keys())

    def __init__(self, nodes: list = None, transactions: list = None, nature: str = 'io' or 'process',
                 number: int = None):
        """
        Instantiates the Pathway class.
        :param nodes: contains a list of consecutive Node Objects
        :param transactions: contains a list of consecutive Transaction Objects
        :param nature: type of data (i.e. input-output or process)
        :param number: adds the number of the pathway in the order of extraction, can be used for lists
        :return:
        """
        if nodes is None:
            nodes = []
        if transactions is None:
            transactions = []

        self.nodes = [node for node in nodes]
        self.transactions = [transaction for transaction in transactions]
        self.nature = nature
        self.number = number

    def __repr__(self):
        """
        Representation of a Pathway Object when printed
        :return: a string
        """
        transaction_string = ''
        for transaction in self.transactions:
            transaction_string += '[T' + str(self.transactions.index(transaction)) + ']' + \
                                  Transaction.__repr__(transaction) + '\n'
        return 'Pathway: \n%s' % transaction_string

    def __eq__(self, other):
        """
        Compares an instance of pathway to another by checking the equality of its attributes
        :param other: the other object used in the comparison
        :return: bool
        """
        return self.get_id() == other.get_id()

    def __hash__(self):
        """
        Must be implemented to be able to use pathways in sets
        :return:
        """
        return hash(self.get_id())

    def __ne__(self, other):
        """
        Compares an instance of pathway to another by checking the equality of its attributes
        :param other: the other object used in the comparison
        :return: bool
        """
        return not self == other

    def __setstate__(self, pickled_dict):
        """
        Sets the attributes from a pickled state
        :param pickled_dict: the pickled_dict_
        :return:
        """
        for var in pickled_dict.keys():
            try:
                setattr(self, var, pickled_dict[var])
            except AttributeError:
                pass

    def __getstate__(self):
        """
        Generates a dict of internal variables
        :return: a dict object of internal variables
        """
        dict_to_pickle = {}
        for var in self.__slots__:
            try:
                dict_to_pickle[var] = getattr(self, var)
            except AttributeError:  # accommodates earlier versions of the instance which don't use __slots__
                pass
        return dict_to_pickle

    def get_nature(self) -> str:
        """
        Retrieves nature of pathway, i.e. process or input-output
        :return: a string
        """
        return self.nodes[0].nature

    def get_id(self):
        """
        Generates the node ID by compounding its name, region and stage
        :return: pathway_id
        """
        list_of_strings = [str(item.get_node_id()) for item in self.nodes]
        pathway_id = '_'.join(list_of_strings)
        return pathway_id

    def get_short_id(self) -> str:
        """
        Generates a short ID for the pathway, by stitching the sector/process indices
        :return: pathway_short_id
        """
        list_of_node_indices = [str(node.index_reference) for node in self.nodes]
        pathway_short_id = '_'.join(list_of_node_indices)
        return pathway_short_id

    def get_num_nodes(self):
        """
        Retrieves the number of nodes
        :return: The number of nodes (int)
        """
        return len(self.nodes)

    def get_num_transactions(self):
        """
        Retrieves the number of transactions
        :return: The number of transactions (int)
        """
        return len(self.transactions)

    def _check_for_node(self, node):
        """
        Checks if the node exists in the Pathway Object
        :param node: a Node object
        :return: True or False
        """
        if node in self.nodes:
            return True
        else:
            return False

    def _add_nodes(self, transaction):
        """
        Adds the nodes of the transaction to the nodes list
        :param transaction: A transaction object
        :return:
        """
        origin_node, target_node = transaction.get_nodes()

        for node in [origin_node, target_node]:
            if self._check_for_node(node):
                pass
            else:
                self.nodes.insert(node.stage - 1, node)

    def _del_node(self, transaction):
        """
        Deletes the origin_node of the transaction as it is the end of the pathway
        :param transaction: a Transaction object
        :return:
        """
        origin_node, target_node = transaction.get_nodes()
        index = self.nodes.index(origin_node)  # make sure that there are no duplicate nodes
        del self.nodes[index]

    def add_transaction(self, transaction):
        """
        Adds a transaction to the Pathway Object by updating both the nodes and transactions list
        :param transaction: instance of a Transaction Object
        :return:
        """
        stage = transaction.get_stage()
        self.transactions.insert(stage, transaction)
        self._add_nodes(transaction)
        if stage > len(self.transactions):
            print("The specified transaction has been appended to the end of the pathway")

    def del_transaction(self, transaction):
        """
        Remove a transaction from the Pathway Object
        :param transaction: location of the Transaction Object to be removed
        :return: a list of Transaction Objects in the Pathway Object
        """
        stage = transaction.get_stage()
        try:
            del self.transactions[stage]
            self._del_node(transaction)
        except IndexError:
            "The specified stage is too far upstream and is not contained in the pathway"
        return self.transactions

    def get_direct_env_intensity(self, flows: list):
        """
        Computes the total direct environmental of the pathway
        :param flows: a list of flows for which we want the intensity
        :return: intensity of the pathway as a dict with flows as keys and the respective total intensity as values
        """
        intensity = {}
        for flow in flows:
            for node in self.nodes:
                try:
                    # convert to float to avoid Numpy type errors in GUI
                    intensity[flow] += float(node.direct_intensity.flow)
                except KeyError:
                    raise KeyError('The requested environmental flow [%s] could not be found.' % str(flow))
        return intensity

    def get_tech_intensity(self, stage: int = None):
        """
        Determines the total technological intensity of the pathway or up to a certain stage if specified
        :param stage: stage at which to get the technological intensity of the pathway
        :return: tech intensity as a float
        """

        if stage is None:
            stage = -1
        return self.transactions[stage - 1].accum_tech_coeff

    def get_fraction_of_total_intensity_for(self, flow: str, rounding: int = None, percentage: bool = False):
        """
        Calculates the ratio of the direct requirement of the last node to the total requirement of the root node
        :param flow: the name of the environmental flow
        :param rounding: the number of digits potentially used for rounding
        :param percentage: return as percentage
        :return: float: fraction
        """
        fraction = self.nodes[-1].direct_intensities[flow] / self.nodes[0].total_intensities[flow]

        # convert to float to avoid Numpy type errors in GUI
        fraction = float(fraction)

        if rounding is not None:
            fraction = round(fraction, rounding)

        if percentage:
            return fraction * 100
        else:
            return fraction

    def get_stage_name(self, ref_dict: dict, stage: int = None, for_gui: bool = False, include_stage: bool = False):
        """
        returns the name of the node for the specified stage
        :param ref_dict: the sector definition dict of the supply chain object
        :param stage: the stage in question
        :param for_gui: boolean flagging if the function is called from a gui
        :param include_stage: adds a prefix stating the stage of this node
        :return:
        """
        if stage is None:
            stage = len(
                self.nodes) - 1  # need this syntax, not '-1', to return last item since we need to print stage
        try:
            if not include_stage:
                return self.nodes[stage].get_node_attribute(ref_dict, 'Name')
            else:
                return 'S%s %s' % (str(stage), self.nodes[stage].get_node_attribute(ref_dict, 'Name'))
        except IndexError:
            if not for_gui:
                raise IndexError('The stage your are trying to access has not been reached in this pathway')
            else:  # used for gui
                if stage == 1:
                    if not include_stage:
                        return self.nodes[0].get_node_attribute(ref_dict, 'Name')
                    else:
                        return 'S0 %s' % self.nodes[0].get_node_attribute(ref_dict, 'Name')
                else:
                    return ''

    def get_intensity(self, type_: str, flow: str, rounding: int = None, multiplier=None):
        """
        Returns the direct or total intensity of the pathway
        :param type_: the intensity type, i.e. direct or total
        :param flow: the name of the environmental flow
        :param rounding:the number of digits potentially used for rounding
        :param multiplier: multiply the intensity by this number, typically used to scale IO intensities using price
        :return: float
        """
        if type_ == 'direct':
            intensity = self.nodes[-1].direct_intensities[flow]
        elif type_ == 'total':
            intensity = self.nodes[-1].total_intensities[flow]
        else:
            raise (TypeError('The type of intensity to calculate must be "direct" or "total"'))

        # convert to float to avoid Numpy type errors in GUI
        intensity = float(intensity)

        if isinstance(multiplier, float):
            intensity *= multiplier

        if rounding is not None:
            intensity = round(intensity, rounding)

        return intensity

    def print_pathway(self, flow: str, ref_dict: dict, streamlined: bool = True or False,
                      to_csv: bool = True or False, multiregional: bool = False):
        """
        Prints the pathway in ASCii characters as a csv file.
        :param flow: Flow being analysed
        :param ref_dict: the sector definition dict of the supply chain object
        :param streamlined: whether or not the output should be streamlined
        :param to_csv: whether the output is printed to a csv file or not
        :param multiregional: wheter or not we should plot the region of the node
        :return: a string of output summarising data for the pathway
        """
        line_list = list()
        if to_csv:
            line_list.append(
                "{:.6%}".format(  # format the percentage of total requirements represented by the pathway
                    self.get_fraction_of_total_intensity_for(flow)))
        else:
            line_list.append(self.get_fraction_of_total_intensity_for(flow))
        line_list.append(self.get_intensity('direct', flow))
        line_list.append(self.get_intensity('total', flow))

        if self.get_num_nodes() > 1:
            start = 1
        elif self.get_num_nodes() == 1:
            start = 0
        else:
            raise IndexError('The pathway is empty, there are no nodes to print')

        for node in self.nodes[start:]:
            if streamlined:
                line_list.append(node.get_node_attribute(ref_dict, 'Name'))
            else:
                line_list.append(node.direct_intensities[flow])
                line_list.append(node.get_node_attribute(ref_dict, 'Name'))
            if multiregional:
                line_list[-1] += '(' + node.get_node_attribute(ref_dict, 'Region') + ')'
        if to_csv:
            output_string = ''
            for item in line_list:
                output_string += str(item) + '\t'
            return output_string
        else:
            return line_list

    def get_flows(self) -> list:
        """
        Retrieves the set of flows from all Node Objects.
        :return: a list of flows
        """
        flows_set = set()
        for node in self.nodes:
            flows_set = set.union(flows_set, set(node.get_flows()))
        return list(flows_set)

    def get_unit(self, ref_dict: dict) -> str:
        """
        Retrieves the functional unit of the root node
        :param ref_dict: the sector definition dict of the supply chain object
        :return: a string
        """
        try:
            return self.nodes[0].get_node_attribute(ref_dict, 'Unit')
        except IndexError:
            raise IndexError('Tried to retrieve the unit of the root node but this pathway has no nodes at all')


class PathwayList(list):
    """
    A class inheriting list but overriding some of its functions to allow for calculations specific to lists of
    pathways in the context of hybrid analysis.
    """

    _slots_dict = {'object_type': Pathway}
    __slots__ = ['object_type']

    def __init__(self, *args):
        """
        Initiate the list instance based on list
        """
        self.object_type = Pathway
        list.__init__(self, *args)

    def __setstate__(self, pickled_dict):
        """
        Sets the attributes from a pickled state
        :param pickled_dict: the pickled __dict__
        :return:
        """
        for var in pickled_dict.keys():
            try:
                setattr(self, var, pickled_dict[var])
            except AttributeError:
                pass

    def __getstate__(self):
        """
        Generates a dict of internal variables
        :return: a dict object of internal variables
        """
        dict_to_pickle = {}
        for var in self.__slots__:
            try:
                dict_to_pickle[var] = getattr(self, var)
            except AttributeError:  # accommodates earlier versions of the instance which don't use __slots__
                pass
        return dict_to_pickle

    def get_nature(self):
        """
        Retrieves nature of pathway
        :return: a string
        """
        try:
            nature = self[0].get_nature()
            return nature
        except (AttributeError, IndexError):
            return None

    def calculate_direct_requirements_dict(self, flows_list: list):
        """
        Calculate the sum of all direct requirements of each last node found in the list of pathways provided
        :param flows_list: list of flows for which direct requirements should be summed
        :return: a dictionary of summed direct requirements
        """
        summed_direct_requirement_dict = {}
        for flow in flows_list:
            summed_direct_requirement_dict[flow] = 0.
            for pathway in self:
                dr = pathway.get_intensity('direct', flow)
                if dr is not None:
                    summed_direct_requirement_dict[flow] += dr
                else:
                    pass
        return summed_direct_requirement_dict

    def multiply(self, value, flows=None):
        """
        Multiplies the direct requirements of last nodes by the specified value if they are io nodes
        and returns a copy of the list. Used to convert IO results from flow by financial output to flow by physical
        output.
        :param value: the scalar value by which to multiply
        :param flows: if None, get all flows from first node
        :return: a PathwayList Object
        """
        temp = copy.deepcopy(self)
        if flows is None:
            flows = self[0].get_flows()
        for flow in flows:
            for pathway in temp:
                if pathway.nature == 'io':
                    pathway.nodes[-1].direct_intensities[flow] *= value
        return temp

    def get_first_n_pathways(self, flow: str, number_of_pathways: int = None, value: float = None):
        """
        Returns a list of pathways sorted in descending order for a particular flow, up to a 'n' limit.
        :param flow: The particular flow for which the analysis is performed
        :param number_of_pathways: The number of pathways to be returned
        :param value: Price of the material being analysed, use only if pathways_list has not been multiplied before
        :return: A sorted and sliced PathwayList Object
        """

        def get_last_node_direct_intensity(pathway):
            """
            Nested function to extract the direct intensity of the last node of a pathway, for a given flow. Used to
            rank pathways
            :param pathway: Pathway object
            :return: Direct intensity value for a specific flow
            """
            if pathway.nature == 'io' and value is not None:
                return pathway.nodes[-1].direct_intensities[flow] * value
            else:
                if pathway.nodes[-1].direct_intensities[flow] is not None:
                    return pathway.nodes[-1].direct_intensities[flow]
                else:
                    return -1  # use to put None items at the end

        if number_of_pathways is not None:
            return PathwayList(sorted(self, key=get_last_node_direct_intensity)[:number_of_pathways][::-1])
        else:
            return PathwayList(sorted(self, key=get_last_node_direct_intensity)[::-1])

    def get_pathways_as_dataframe(self, flows_list: str, nb_stages: int, ref_dict: dict, streamlined=True or False,
                                  header_list=None):
        """
        Returns the PathwayList object as a pandas.DataFrame Object formatted for printing as a csv or Excel file.
        :param flows_list: list of flows considered in the SPA
        :param nb_stages: number of stages considered in the SPA
        :param ref_dict: The dictionary of reference of the SupplyChain object, containing metadata
        :param streamlined: whether the output is streamlined or not
        :param header_list: list of strings used as header for the pandas.DataFrame Object
        :return:ranked pandas DataFrame of pathways

        """
        if header_list is None:
            header_list = copy.deepcopy(INITIAL_HEADER_LIST)
            for stage in range(nb_stages):
                if streamlined:
                    pass
                else:
                    header_list.append('Stage %s direct intensity' % str(stage + 1))
                header_list.append('Stage %s' % str(stage + 1))
        else:
            header_list = header_list

        pathway_dict = OrderedDict()
        for header_index in range(len(header_list)):
            pathway_dict[header_list[header_index]] = []

        for pathway in self:
            pathway_data_to_print = pathway.print_pathway(flows_list, ref_dict, streamlined, to_csv=False)
            for header_index in range(len(header_list)):
                try:
                    pathway_dict[header_list[header_index]].append(pathway_data_to_print[header_index])
                except:
                    pathway_dict[header_list[header_index]].append('')

        pathway_df = pd.DataFrame.from_dict(pathway_dict, orient='columns')

        return pathway_df

    def get_per_stage_results(self, root_node: 'Node Object', nb_stages: int, flow_coverage: float, flow: str,
                              as_absolute=True or False):
        """
        Calculates the results of the SPA on a per stage basis. Includes the sum of all direct
        requirements of Stage nodes, as well as the percentage of the total requirements this represents.
        :param root_node: Root node of the analysis (original target node)
        :param nb_stages: Number of stages considered in the SPA
        :param flow_coverage: Total coverage of the SPA
        :param flow: flow for which the analysis is run
        :param as_absolute: whether the results are provided as absolute values or as a percentage of the total.
        :return: stage_total_dict
        """
        stage_total_dict = OrderedDict()

        for stage in range(nb_stages + 1):
            stage_total_dict['Stage %s' % str(stage)] = 0.
        stage_total_dict['Stage 0'] = 0.
        stage_total_dict['Remainder'] = 0.

        for pathway in self:
            node_value = pathway.nodes[-1].direct_intensities[flow]
            node_stage = 'Stage %s' % str(pathway.nodes[-1].stage)
            stage_total_dict[node_stage] += float(node_value)
        stage_total_dict['Remainder'] = root_node.total_intensities[flow] * (1 - flow_coverage)

        if not as_absolute:
            for stage_number, stage_total in stage_total_dict.items():
                value = stage_total_dict[stage_number] / root_node.total_intensities[flow]
                stage_total_dict[stage_number] = value

        return stage_total_dict

    def get_nb_pathways_per_flow(self, flows=None):
        """
        Function used to return the number of nodes which include data for each specific flow (because of thresholds
        testing, some flows are not covered in the direct intensities dict of the nodes).
        :param flows: list of flows for which the analysis should be conducted.
        :return: node_count_dict
        """
        if flows is None:
            flows = self[0].get_flows()
        # print(flows)
        node_count_dict = {flow: 0 for flow in flows}
        for pathway in self:
            for flow in flows:
                if pathway.nodes[-1].direct_intensities[flow] is not False:
                    node_count_dict[flow] += 1
                else:
                    pass
        return node_count_dict

    def get_most_contributing_nodes(self, sectors_list: list, sectors_dict: dict, flows: list = None):
        """
        Iterates through all Pathway objects in the list and account for the overall contribution of a sector
        :param sectors_list: List of all sectors or processes covered by the database
        :param sectors_dict: Sectors_definition_dict object - included in the relevant .lci file
        :param flows: List of flows to be covered
        :return: dict
        """

        if flows is None:
            flows = self[0].get_flows()

        root_total_intensities = {}
        for flow in flows:
            root_total_intensities[flow] = 0.

        contribution_dict = {}
        contribution_dict['Direct Stage 0'] = {}
        for sector in sectors_list:
            contribution_dict[sector] = {}
            for flow in flows:
                contribution_dict[sector][flow] = 0.
        for flow in flows:
            contribution_dict['Direct Stage 0'][flow] = 0.

        for pathway in self:
            for flow in flows:
                last_node = pathway.nodes[-1]
                if last_node.direct_intensities[flow] is not False:
                    if last_node.root_node_boolean:
                        node_sector = 'Direct Stage 0'
                        # print(sectors_dict[last_node.index_reference]['Name'])
                        root_total_intensities[flow] = last_node.total_intensities[flow]
                    else:
                        node_sector = sectors_dict[last_node.index_reference]['Name']
                    contribution_dict[node_sector][flow] += last_node.direct_intensities[flow]

        for sector, data in contribution_dict.items():
            for flow in flows:
                contribution_dict[sector][flow] = contribution_dict[sector][flow] / root_total_intensities[flow]

        return contribution_dict

    def update_pathways_from_new_spa(self, spa_pathways_results_dict: dict, return_no_match_list: bool = False):
        """
        Trims a Pathway Object based on a specified max_stage and threshold dictionary
        :param spa_pathways_results_dict: a dictionary of spa short_ids and pathways as k, v
        that is used to replace current pathways
        :param return_no_match_list: a boolean flag to return or not the short_ids of pathways that could'nt be replaced
        :return: a PathwayList Object
        """
        self_as_ids = map(lambda x: x.get_short_id(), self)
        new_list = PathwayList()
        no_match_list = PathwayList()
        for spa_id in self_as_ids:
            try:
                new_list.append(spa_pathways_results_dict[spa_id])
            except KeyError:
                no_match_list.append(spa_id)
        if return_no_match_list:
            return new_list, no_match_list
        else:
            return new_list


class TemporaryList(list):
    """
    A class inheriting list but overriding some of its functions.
    """

    _slots_dict = {
        'parent_sc': 'SupplyChain',
        '_type': str
    }
    __slots__ = list(_slots_dict.keys())

    def __init__(self, parent_sc, _type='nodes' or 'transactions'):
        """
        Initiate the list instance based on list
        :param parent_sc: the parent supply chain object
        :param _type: defines
        :return:
        """
        self.parent_sc = parent_sc
        super().__init__(self)
        self._type = _type

    def __copy__(self):
        """
        Copy function
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for attribute in self.__slots__:
            setattr(result, attribute, getattr(self, attribute))
        return result

    def __deepcopy__(self, memo):
        """
        Deep copy function (creates a completely copied object)
        :param memo: data to be copied
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for attribute in self.__slots__:
            setattr(result, attribute, copy.deepcopy(getattr(self, attribute), memo))
        return result

    def __setstate__(self, pickled_dict):
        """
        Sets the attributes from a pickled state
        :param pickled_dict: the pickled __dict__
        :return:
        """
        for var in pickled_dict.keys():
            try:
                setattr(self, var, pickled_dict[var])
            except AttributeError:
                pass

    def __getstate__(self):
        """
        Generates a dict of internal variables
        :return: a dict object of internal variables
        """
        dict_to_pickle = {}
        for var in self.__slots__:
            try:
                dict_to_pickle[var] = getattr(self, var)
            except AttributeError:  # accommodates earlier versions of the instance which don't use __slots__
                pass
        return dict_to_pickle

    def append(self, item, target_node=None):
        """
        Appends an item to the TemporaryList Object.
        :param item: a node or transaction item to append
        :param target_node: the target node that the node item is linked to, use only if type is node
        :return:
        """
        if self._type is 'nodes' and isinstance(item, Node):
            if isinstance(target_node, Node):
                self._cleanup(target_node)
                if self._check_nodes_consistency(item):
                    super().append(item)
                else:
                    raise IndexError('Trying to append a node with the wrong stage sequence')

            elif self == []:  # if we're adding the root node
                if self._check_nodes_consistency(item):
                    super().append(item)
                else:
                    raise IndexError('Trying to append a node with the wrong stage sequence')
            else:
                pass  # raising the type error is causing trouble with copy.deepcopy
                # raise TypeError('The target node passed to the append method is not a Node')

        elif self._type is 'transactions' and isinstance(item, Transaction):
            self._cleanup(item)
            if self._check_transaction_consistency(item):
                # print('adding the following transaction ==> ', item)
                super().append(item)
            else:
                raise IndexError('Trying to append a transaction with the wrong target node')
        else:
            pass  # raising the type error is causing trouble with copy.deepcopy
            # raise TypeError('The item passed to the append method of the'
            #                 ' TemporaryList class is not a Node or a Transaction')

    def already_exists(self, item):
        """
        Checks if the item is already in the TemporaryList object
        :param item: A Node or Transaction object
        :return: True or False
        """
        if self._type is 'nodes' and isinstance(item, Node):
            id_list = [node.get_node_id() for node in self]
            if item.get_node_id() in id_list:
                return True
            else:
                return False

    def _cleanup(self, target):
        """
        Cleans up the list by removing superfluous nodes/transactions
        :param target: A Node or Transaction object to be cleaned up from the TemporaryList object
        :return:
        """
        # print('============== CLEANING UP == ', self._type)
        # print('state BEFORE cleanup --> ', self)
        index = self._get_cleanup_index(target)
        self._clear_from_index(index)
        # print('\n','state AFTER cleanup --> ', self, '\n\n')

    def _clear_from_index(self, index):
        """
        Truncates the TemporaryList object from the index onwards, keeps only the first part
        :param index: the index that we use to truncate the list
        :return:
        """
        if self._type is 'transactions' and index == 'clear transactions list':
            del self[0:]  # empties the lists
        else:
            del self[index + 1:]

    def _get_cleanup_index(self, target: Node or Transaction):
        """
        Determines the index in the TemporaryList object until which we need to keep the elements
        :param target: A Node or Transaction object that is used to determine the index from which the list is cleared
        :return: the index in question
        """
        index = 0
        if isinstance(target, Node):
            target_node_id = target.get_node_id()
            for running_index, node in enumerate(self):
                if node.get_node_id() == target_node_id:
                    index = running_index
            return index

        elif isinstance(target, Transaction):
            target_node_id = target.target_node.get_node_id()
            if target_node_id == self.parent_sc.root_node.get_node_id():
                index = 'clear transactions list'
            else:
                for running_index, transaction in enumerate(self):
                    running_origin_node_id = transaction.origin_node.get_node_id()
                    if running_origin_node_id == target_node_id:
                        index = running_index
            return index

        else:
            raise TypeError('The target item passed is neither a Node nor a Transaction object')

    def remove_last_item(self):
        """
        Removes the last node/transaction in the TemporaryList object
        :return:
        """
        self.pop()

    def is_empty(self):
        """
        Checks if the TemporaryList object is empty
        :return: True or false
        """
        if self == []:
            return True
        else:
            return False

    def _check_nodes_consistency(self, node):
        """
        Makes sure that all nodes in the TemporaryList object are properly ordered by stage
        :return: True or false
        """
        if self == []:
            # print('temp nodes is empty -> Can add node')
            return True
        elif len(self) == 1:
            # print('temp nodes has one node -> can add node')
            return True
        elif self[-1].stage == node.stage - 1:
            # print('Stages of node to add is sequentially consistent...adding node')
            return True
        else:  # if the stage of the node to append is not sequentially correct
            # print('NODES INCONSISTENT')
            # print('Stage of last node in temp -- > ', self[-1].stage)
            # print('Stage of node to add  -- > ', node.stage)
            # print('state of temp_nodes --> ', self)
            return False

    def _check_transaction_consistency(self, transaction):
        """
        Makes sure that all Transaction objects in the TemporaryList object are properly ordered in terms of origin
        and target Node objects.
        :param transaction: a Transaction object
        :return: True or False
        """
        if self == []:
            return True
        elif len(self) == 1:
            return True
        elif self[-1].origin_node.get_node_id() == transaction.target_node.get_node_id():
            # print('Stages of node to add is sequentially consistent...adding node')
            return True
        else:  # if the stage of the node to append is not sequentially correct
            # print('NODES INCONSISTENT')
            # print('Stage of last node in temp -- > ', self[-1].stage)
            # print('Stage of node to add  -- > ', node.stage)
            # print('state of temp_nodes --> ', self)
            return False


class SupplyChain:
    """
    A class representing the entire supply chain associated with an input-output sector or with a process.
    """

    _attribute_dict = {
        'target_ID': int,
        'sector_definition_dict': dict,
        'a_matrix': np.array,
        'dr_vectors_dict': dict,
        'tr_vectors_dict': dict,
        'thresholds_dict': dict,
        'max_stage': int,
        'nature': str,
        'pathways_list': PathwayList,
        'root_node': Node,
        'flows_dict': dict
    }

    def __init__(self, target_ID: int, sector_definition_dict: dict, a_matrix: np.array, dr_vectors_dict: dict,
                 tr_vectors_dict: dict, thresholds_dict: dict, flows_dict: dict, max_stage: int = 7,
                 nature: str = 'io' or 'process', pathways_list=None, root_node=None):
        """
        Instantiates the supply chain class.
        :param target_ID: the target sector number for which the supply chain is created
        :param sector_definition_dict: a dictionary listing the sector numbers and their characteristics
        :param a_matrix: the technological matrix of the input-output tables considered
        :param dr_vectors_dict: a dictionary of vectors of direct flow intensity for each flow assessed and each
        sector of the A matrix
        :param tr_vectors_dict: a dictionary of vectors of total flow intensity for each flow assessed and each
        sector of the A matrix
        :param thresholds_dict: a dictionary of thresholds for each flow analysed - thresholds can be modified from
        flow to flow
        :param flows_dict: a dictionary containing flow names as keys (e.g. 'energy') and a sub dictionary containing
        the 'name' and 'unit' of that flow.
        :param max_stage: the maximum number of stages upstream to consider during the SPA - the ultimate maximum stage
        is currently five.
        :param nature: specifies whether the supply chain data is pure input-output or process.
        :param pathways_list: the list of pathways constituting the supply chain, only used for loading from a pickle
        :param root_node: the root node object of the supply chain, only used for loading from a pickle
        :return:
        """
        self.target_ID = target_ID  # numerical index of the sector or process being analysed
        self.sector_definition_dict = sector_definition_dict
        self.a_matrix = a_matrix
        self.dr_vectors_dict = dr_vectors_dict
        self.tr_vectors_dict = tr_vectors_dict
        self.flows_dict = flows_dict
        self.max_stage = max_stage
        self.thresholds_dict = thresholds_dict
        self.nature = nature

        self.header_line = \
            '% of total intensity\tdirect intensity of last node in pathway\ttotal intensity of pathway\t'

        self.chosen_flows_list = list(self.flows_dict.keys())
        self.temp_nodes = TemporaryList(parent_sc=self, _type='nodes')
        self.temp_transactions = TemporaryList(parent_sc=self, _type='transactions')

        if pathways_list is None:
            self.pathways_list = PathwayList()
        else:
            self.pathways_list = pathways_list

        self._validate_data()  # initiate tests on the variables to check that their format is correct

        if root_node is None:
            self.root_node = self._generate_root_node()
        else:
            self.root_node = root_node

        # generate all necessary functions to extract pathways up to the max stage. Avoids recursive code.
        for stage in range(1, self.max_stage + 1):
            _add_extraction_stage(stage)

    def __repr__(self):
        """
        Representation of a SupplyChain object when printed.
        :return: a string
        """
        return 'SupplyChain => Target: %s, Max Stage: %s, Nature: %s' % \
               (self.sector_definition_dict[self.target_ID]['Name'], self.max_stage, self.nature)

    def __eq__(self, other):
        """
        Compares an instance of a SupplyChain object to another by checking the equality of its attributes
        :param other: the other object used in the comparison
        :return:
        """
        if isinstance(other, self.__class__):
            return self.target_ID == other.target_ID and self.thresholds_dict == other.thresholds_dict and \
                   self.max_stage == other.max_stage and self.a_matrix.shape == other.a_matrix.shape and \
                   self.sector_definition_dict == other.sector_definition_dict
        else:
            return False

    def __ne__(self, other):
        """
        Compares an instance of a supply chain to another by checking the equality of its attributes
        :param other: the other object used in the comparison
        :return:
        """
        return not self.__eq__(other)

    def __copy__(self):
        """
        Copy function
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for attribute in self.__slots__:
            setattr(result, attribute, getattr(self, attribute))
        return result

    def __deepcopy__(self, memo):
        """
        Deep copy function (creates a completely copied object)
        :param memo: data to be copied
        :return: copied object
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for attribute in self.__slots__:
            setattr(result, attribute, copy.deepcopy(getattr(self, attribute), memo))
        return result

    def __setstate__(self, pickled_dict):
        """
        Sets the attributes from a pickled state
        :param pickled_dict: the pickled __dict__
        :return:
        """
        for var in pickled_dict.keys():
            try:
                setattr(self, var, pickled_dict[var])
            except AttributeError:
                pass

    def __getstate__(self):
        """
        Generates a dict of internal variables
        :return: a dict object of internal variables
        """
        dict_to_pickle = {}
        for var in self.__slots__:
            try:
                dict_to_pickle[var] = getattr(self, var)
            except AttributeError:  # accommodates earlier versions of the instance which don't use __slots__
                pass
        return dict_to_pickle

    def _validate_data(self):
        """
        A series of tests used to validate the initiated data. It verifies whether the dictionary of thresholds
        provided contains any information, whether it is as long as the list of flows used for the analysis, and
        whether the thresholds are within acceptable ranges. It then verifies that the dictionary of vectors provided
        for TR and DR are in line with the flows assessed, and are provided as numpy arrays. Finally it checks if the
        number of stages set to be considered is within the scope of the code (ie whether or not it is greater than
        the ultimate maximum stage).
        """
        # Test on thresholds_dict STARTS
        if len(self.thresholds_dict) == 0:
            raise ValueError('The thresholds dictionary is empty')
        elif len(self.thresholds_dict) > len(self.chosen_flows_list):
            raise ValueError('The thresholds dictionary includes %s too many flows' %
                             str(len(self.thresholds_dict) - len(self.chosen_flows_list)))
        for flow, threshold in self.thresholds_dict.items():
            if threshold > 1 or threshold < 0.0000000001:
                raise ValueError('The threshold defined for %s is outside the scope.' % flow)

        # Test on dr_vector_dict and tr_vector_dict STARTS
        for flow, arrays in self.dr_vectors_dict.items() and self.tr_vectors_dict.items():
            if flow not in self.chosen_flows_list:
                raise ValueError('%s is not included in the current list of flows' % flow)
            if type(arrays) != np.ndarray:
                raise ValueError('Value provided for %s is not a numpy array' % flow)

    def get_pathways_as_dict(self, list_of_pathway_indices: list = None):
        """
        Generates a dictionary containing the pathway index as a key and the pathway object as a value
        :param list_of_pathway_indices: a custom list of pathway indices
        :return: a dictionary of pathways with pathway index as keys and a PathwayList object as a value
        """
        pathways_dict = {}
        if list_of_pathway_indices is None:
            for pathway_num, pathway in enumerate(self.pathways_list):
                pathways_dict[pathway_num] = pathway
        else:
            for index in list_of_pathway_indices:
                pathways_dict[index] = self.pathways_list[index]
        return pathways_dict

    def get_pathways(self, list_of_pathway_indices: list = None, list_of_pathway_short_ids: list = None):
        """
        Returns all pathways as a list, or a list of specified pathways, by indices
        :param list_of_pathway_indices: a custom list of pathway indices
        :param list_of_pathway_short_ids: a custom list of pathway short ids
        :return: a PathwayList object
        """
        pathways_list = []
        if list_of_pathway_indices is None and list_of_pathway_short_ids is None:
            pathways_list = self.pathways_list
        elif list_of_pathway_indices is not None:
            for index in list_of_pathway_indices:
                pathways_list.append(self.pathways_list[index])
        elif list_of_pathway_short_ids is not None:
            pathways_list = \
                [pathway for pathway in self.pathways_list if pathway.get_short_id() in list_of_pathway_short_ids]
        return pathways_list

    def find_pathways(self, node_indices_list: list):
        """
        Finds all Pathway objects in which the specified node indices occur
        :param node_indices_list: a list of node indices
        :return: a dictionary of pathways, with node indices as keys and a list of matching pathway indices as a value
        """
        matching_pathways = {}
        pathway_number = 0
        for pathway in self.pathways_list:
            for node in pathway.nodes[1:]:  # skip the root node
                if node.index_reference in node_indices_list:
                    try:
                        if isinstance(matching_pathways[node.index_reference], list):
                            matching_pathways[node.index_reference].append(pathway_number)
                    except KeyError:
                        matching_pathways[node.index_reference] = [pathway_number]
            pathway_number += 1
        return matching_pathways

    def get_coverage_of(self, flow: str, rounding=None):
        """
        Computes the coverages of the specified flow in the current SupplyChain object
        :param flow: the specified flow, as a string, e.g. 'energy', 'water', 'carbon'
        :param rounding: round the results to the specified number of digits
        :return: a ratio of the total intensity covered by the SupplyChain object, as a float
        """
        covered_flows = self.get_flows()
        if flow not in covered_flows:
            raise (AttributeError('The specified flow was not covered in this structural path analysis'))
        else:
            total_direct_intensity_covered = 0.0  # self.root_node.direct_intensities[flow]
            for pathway in self.pathways_list:
                if pathway.nodes[-1].direct_intensities[flow] is not None:
                    total_direct_intensity_covered += pathway.nodes[-1].direct_intensities[flow]
            try:
                percentage_covered = total_direct_intensity_covered / self.root_node.total_intensities[flow]
            except ZeroDivisionError:
                percentage_covered = 0.0  # if the total is zero, then the share is 0

            if rounding is not None:
                percentage_covered = round(percentage_covered, rounding)

            return percentage_covered

    def get_flows(self) -> list:
        """
        Returns a list of flows covered
        :return: a list object
        """
        return list(self.thresholds_dict.keys())

    def get_regions(self):
        """
        Reads the regions in ref_dataframe and determines which regions are actually included in the  file
        :return: List of regions
        """
        regions_list = [info_dict['Region'] for info_dict in self.sector_definition_dict.values()]

        return list(set(regions_list))  # eliminate duplicates

    def get_root_node(self):
        """
        Returns the root node of the supply chain
        :return: a Node object
        """
        return self.root_node

    def _is_multiregional(self):
        """
        Checks if the supply chain is multiregional
        :return: True or False
        """
        if len(self.get_regions()) > 1:
            return True
        elif len(self.get_regions()) == 1:
            return False
        else:
            raise ValueError('The supply chain contains no regions, check your input files')

    def _generate_root_node(self):
        """
        Generates the root Node object of the SupplyChain object.
        :return: a root Node object
        """
        node_dr_dict = {}
        node_tr_dict = {}
        for flow in self.chosen_flows_list:
            node_dr_dict[flow] = self.dr_vectors_dict[flow][self.target_ID]
            node_tr_dict[flow] = self.tr_vectors_dict[flow][self.target_ID]
        if self.nature == 'io' or self.nature == 'process':
            root_node = Node(index_reference=self.target_ID,
                             stage=0,
                             direct_intensities=node_dr_dict, total_intensities=node_tr_dict,
                             nature=self.nature,
                             unit=self.sector_definition_dict[self.target_ID]['Unit'])
        else:
            raise TypeError(
                "The nature of the SupplyChain object was not recognised. Make sure it is either 'process' "
                "or 'io'")
        return root_node

    def get_number_of(self, count_what='nodes' or 'transactions' or 'pathways'):
        """
        Retrieves the number of nodes in the SupplyChain object.
        :param count_what: specify what object is being counted, i.e. 'nodes', 'transactions', 'pathways'
        :return: Number of object (int) in the supply chain
        """
        if count_what == 'pathways':
            return len(self.pathways_list)
        elif count_what == 'nodes' or count_what == 'transactions':
            num_nodes = 0
            num_transactions = 0
            for pathway in self.pathways_list:
                num_nodes += pathway.get_num_nodes()
                num_transactions += pathway.get_num_transactions()
            if count_what == 'nodes':
                return num_nodes
            elif count_what == 'transactions':
                return num_transactions
        else:
            raise (TypeError("Count what must be 'nodes', 'transactions', or 'pathways'"))

    def add_pathway(self, pathway):
        """
        Adds the specified Pathway object to the SupplyChain object.
        :param pathway: a Pathway object
        :return: an updated PathwayList object
        """
        self.pathways_list.append(pathway)
        return self.pathways_list

    def remove_pathway(self, pathway):
        """
        Removes the specified Pathway object from the SupplyChain object.
        :param pathway: a Pathway object
        :return: an updated PathwayList object
        """
        self.pathways_list.remove(pathway)
        return self.pathways_list

    def get_filename(self, full=False) -> str:
        """
        Generates a filename for the supply chain object
        :param full: boolean referring to whether the filename is requested is long or short.
        :return: filename string
        """
        if self.nature == 'io':
            suffix = 'sectors'
        elif self.nature == 'process':
            suffix = 'processes'
        else:
            raise (TypeError('The nature of an spa can only be "io" or "process"'))

        clean_target_name = _get_clean_filename(self.sector_definition_dict[self.target_ID]['Name'])

        if full:
            num_sectors = len(self.sector_definition_dict.keys())
            flows_str = '__'.join(['{}_threshold_{}'.format(k, v) for k, v in self.thresholds_dict.items()])

            filename = '_'.join([
                'SPA',
                clean_target_name,
                '( %s )' % str(self.target_ID),
                'Max_stage',
                str(self.max_stage),
                flows_str,
                str(num_sectors),
                suffix
            ])
        else:
            filename = '_'.join([
                'SPA',
                clean_target_name,
                '( %s )' % str(self.target_ID),
                'Max_stage',
                str(self.max_stage)
            ])
        return filename

    def save(self, path):
        """
        Saves the SupplyChain object to a file.
        :param path: directory where the SupplyChain object is saved
        """
        temp_dict = {attribute: getattr(self, attribute) for attribute in self._attribute_dict.keys()}
        _save_pickle(temp_dict, path=path)
        del temp_dict

    def load(self, path):
        """
        Loads the SupplyChain object from a file.
        :param path: directory from which the SupplyChain object is loaded
        """
        with gzip.open(path, 'rb') as file:
            loaded_vars = pickle.load(file)
        self.target_ID = loaded_vars['sector']
        self.pathways_list = loaded_vars['pathways_list']
        print("File %s loaded successfully" % path)

    def _generate_file_title_block(self, to_csv=True or False):
        """
        Creates a title block for the file exported reporting a summary of information on the SPA that has been
        conducted.
        :param to_csv: boolean flag referencing whether the data is to be saved to csv file or not
        :return: analysis metadata in the form of a list of strings to be used for the file header
        """
        file_title_block = [
            ['Target sector: ', str(self.sector_definition_dict[self.target_ID]['Name'])],
            ['Sector ID: ', str(self.target_ID + 1)],  # add one to match the index in the csv file
            ['Number of Regions in input data: ', str(len(self.get_regions()))],
            ['Number of sectors in A matrix: ', str(self.a_matrix.shape[0])],
            ['Total number of pathways extracted: ', str(len(self.pathways_list))],
            ['Stages analysed: ', str(self.max_stage)],
            ['Date of extraction: ', time.strftime("%x")],
            ['Time of extraction: ', time.strftime("%X")]
        ]
        if to_csv:
            new_title_block = []
            for row in file_title_block:
                new_row = '\t'.join(row)
                new_title_block.append(new_row)
            return new_title_block
        else:
            return file_title_block

    def _generate_spa_title_block(self, flow, to_csv=True or False):
        """
        Creates a title block for the SPA results produced for each flows.
        :param flow: flow being analysed
        :param to_csv: boolean flag referencing whether the data is to be saved to csv file or not
        :return: title block of the SPA results for a particular flow
        """
        try:
            flow_unit = self.flows_dict[flow].unit
        except AttributeError:
            raise AttributeError('The flow %s is not defined' % flow)

        spa_title_block_list = [
            ['flow analysed:', flow],
            ['unit:',
             "%s/%s" % (flow_unit, self.root_node.get_node_attribute(self.sector_definition_dict, 'Unit'))],
            ['thresholds:', str(self.thresholds_dict[flow])],
            ['direct:', str(self.root_node.direct_intensities[flow]), "%s/%s" %
             (flow_unit, self.root_node.get_node_attribute(self.sector_definition_dict, 'Unit'))],
            ['total:', str(self.root_node.total_intensities[flow]), "%s/%s" %
             (flow_unit, self.root_node.get_node_attribute(self.sector_definition_dict, 'Unit'))],
            ['% of total covered by SPA:', "{:.2%}".format(self.get_coverage_of(flow)),
             'Note: Value may differ from sum of percentages in the table due to rounding']
        ]
        if to_csv:
            new_spa_title_block_list = []
            for row in spa_title_block_list:
                new_row = '\t'.join(row)
                new_spa_title_block_list.append(new_row)
            return new_spa_title_block_list
        else:
            return spa_title_block_list

    def _generate_spa_table_header(self, streamlined=True or False):
        """
        Creates a header for each table of SPA results.
        :param streamlined: whether the SPA table results is streamlined or not
        :return:
        """
        header_line = INITIAL_HEADER_LIST
        for stage in range(self.max_stage):
            if streamlined:
                pass
            else:
                header_line.append('Stage %s direct intensity' % str(stage + 1))
            header_line.append('Stage %s' % str(stage + 1))

        header_line = '\t'.join(header_line)
        return header_line

    def prepare_excel_export(self, streamlined=True or False):
        """
        Prepare SPA for an excel export - compile the data as a series of pandas.DataFrame in a dictionary
        :param streamlined: whether or not the output is streamlined (i.e. a shorter number of header is defined if
        streamlined)
        :return: dictionary of pd.DataFrames
        """
        df_data_dict = OrderedDict()

        block_title_lists = [self._generate_file_title_block(to_csv=False)]
        for flow in self.chosen_flows_list:
            spa_title_block = self._generate_spa_title_block(flow, to_csv=False)
            block_title_lists.append(spa_title_block)

        updated_block_title_list = []
        for list_ in block_title_lists:  # there is one general summary list and as many other list as flows
            for sublist in list_:
                modified_row = tuple(sublist)
                updated_block_title_list.append(modified_row)
            updated_block_title_list.append('\n')

        block_title_df = pd.DataFrame.from_records(updated_block_title_list)
        df_data_dict['Summary'] = [block_title_df, False, False]

        for flow in self.chosen_flows_list:
            flow_df = self.pathways_list.get_pathways_as_dataframe(flow, self.max_stage,
                                                                   self.sector_definition_dict,
                                                                   streamlined)
            df_data_dict[flow.title()] = [flow_df, False, True]

        return df_data_dict

    def export_to_excel(self, path=None, streamlined=True or False):
        """
        Exports the SupplyChain object results to a series of excel spreadsheet (according to the number of flows
        being covered by the Structural Path Analysis)
        :param path: directory of the file to be created (including directory and name).
        :param streamlined: whether or not the output is streamlined - meaning that the number of columns of the spa
        results table will vary to include the DR of all nodes in the pathways, or not.
        """
        writer = pd.ExcelWriter(path=path, engine='xlsxwriter')
        df_data_dict = self.prepare_excel_export(streamlined)
        for df_name, df_data_list in df_data_dict.items():
            df_data_list[0].to_excel(writer, sheet_name=df_name, index=df_data_list[1], header=df_data_list[2])
        writer.save()

    def export_to_csv(self, path=None, streamlined=True or False, multiregional=None):
        """
        Exports the SupplyChain object to a csv file.
        :param path: directory of the file to be created (including directory and name).
        :param streamlined: whether or not the output is streamlined (the number of columns of the spa results table
        :param multiregional: whether or not the data is multiregional (displays or not the region for each stage)
        will include the DR of all nodes in the pathways, or not.
        """
        if multiregional is None:
            multiregional = self._is_multiregional()
        elif multiregional is False or multiregional is True:
            pass
        else:
            raise ValueError('multiregional should be either None, True or False')

        with open(path, 'w', newline='') as csv_file:
            output_writer = csv.writer(csv_file, delimiter='\t')
            for line in self._generate_file_title_block(to_csv=True):
                output_writer.writerow([line])
            output_writer.writerow([''])
            for flow in self.chosen_flows_list:
                for line in self._generate_spa_title_block(flow, to_csv=True):
                    output_writer.writerow([line])
                output_writer.writerow([''])
                output_writer.writerow([self._generate_spa_table_header(streamlined)])
                for pathway in self.pathways_list.get_first_n_pathways(flow, len(self.pathways_list)):
                    # write up pathways extracted
                    if pathway.nodes[-1].direct_intensities[flow] is not None \
                            and pathway.nodes[-1].direct_intensities[flow] != 0.:
                        line = pathway.print_pathway(flow=flow, ref_dict=self.sector_definition_dict,
                                                     streamlined=streamlined, to_csv=True, multiregional=multiregional)
                        output_writer.writerow([line])
                    else:
                        continue
                output_writer.writerow('')

    def _get_property(self, sector_id: int, property_: str):
        """
        Retrieves the specified property_ of the sector with the associated ID.
        :param: sector_id: ID of the sector in the reference dict
        :param: property_: key name of the property_ to retrieve in the dictionary of reference
        """
        try:
            return self.sector_definition_dict[sector_id][property_]
        except KeyError:
            print('The specified property_ ', property_, ' was not found for sector ID ', sector_id)

    def _generate_pathway(self, nodes: 'list of nodes', transactions: 'list of transactions'):
        """
        Converts a list of nodes and a list of transactions into a pathway object.
        :param nodes: a list of nodes instances
        :param transactions: a list of transactions instances
        :return: a Pathway Object
        """
        # get the number of the pathway in the list
        # add 1 to save the first spot for DIRECT requirements
        # use for logs and to display
        pathway_num = len(self.pathways_list) + 1

        pathway = Pathway(nodes, transactions, self.nature, pathway_num)
        # pathway.print_pathway(ref_dict=self.sector_definition_dict)
        return pathway

    def _insert_direct_requirements(self):
        """
        Inserts an additional pathway to represent the direct requirements at stage 0 of the supply chain
        :return:
        """
        renamed_root_node = copy.deepcopy(self.root_node)
        renamed_root_node.root_node_boolean = True  # changes the boolean indicator to identify the root node

        stage_0_pathway = Pathway(nodes=[renamed_root_node], nature=self.nature, number=0)
        self.pathways_list.insert(0, stage_0_pathway)

    def extract_pathways(self, target_sector=None, thresholds_dict=None, max_stage=None):
        """
        Extracts all the pathways and stores them in the SupplyChain object.
        :param target_sector: the target sector ID
        :param thresholds_dict: a dictionary of thresholds for each flow assessed
        :param max_stage: the maximum stage upstream to be considered
        """
        if max_stage is not None:
            self.max_stage = max_stage
        if thresholds_dict is None:
            thresholds_dict = self.thresholds_dict
        if target_sector is None:
            target_sector = self.target_ID

        self._extract_stage_1_pathway(target_sector,
                                      thresholds_dict=thresholds_dict)  # method inserted dynamically into class
        self._insert_direct_requirements()

        return self.pathways_list

    def _remove_node_transaction(self):
        """
        Removes the last Node and Transaction object from the TemporaryList objects.
        """
        self.temp_nodes.remove_last_item()
        self.temp_transactions.remove_last_item()

    def _pathway_already_saved(self, transaction):
        """
        Checks if the passed Transaction object is already in the last pathway. Use to avoid double storing of pruned
        pathways.
        :param transaction: A Transaction object
        :return: True or False
        """
        try:
            test = transaction in self.pathways_list[-1].transactions
        except IndexError:
            test = False
        return test

    def _temp_nodes_and_temp_transactions_empty(self):
        """
        Checks if BOTH temps nodes AND temp_transactions are empty.
        :return: True or False
        """
        if self.temp_nodes.is_empty() and self.temp_transactions.is_empty():
            return True
        else:
            return False

    def _get_requirements_dict(self, sector_ID, tech_coeff, mode='Total' or 'Direct', flows_list=None):
        """
        Calculate the total or direct requirement of a node for every flow considered in the SupplyChain object.
        :param sector_ID: the target sector considered
        :param tech_coeff: technological coefficient in dollar value purchased from the target sector
        :param mode: whether it calculates direct or total requirements
        :param flows_list: list of flows considered in the SPA
        :return: a dictionary of total or direct requirements for each flow
        """
        if flows_list is None:
            flows_list = self.chosen_flows_list
        if mode == 'Total':
            requirements_arrays_dict = self.tr_vectors_dict
        elif mode == 'Direct':
            requirements_arrays_dict = self.dr_vectors_dict
        else:
            raise KeyError('Only Direct or Total requirements can be considered.')
        requirements_sector_dict = {flow: requirements_arrays_dict[flow][sector_ID] * tech_coeff for flow in
                                    flows_list}

        return requirements_sector_dict

    def _get_thresholds_test(self, total_requirements_dict):
        """
        Tests the values in the total requirement dictionary against the thresholds defined for each flow
        :param total_requirements_dict: dictionary of total coefficient calculated for each flow assessed
        """

        threshold_test_dict = {flow: total_requirements_dict[flow] > self.thresholds_dict[flow] for flow in
                               self.chosen_flows_list}
        return threshold_test_dict

    @staticmethod
    def _test_thresholds(threshold_test_dict):
        """
        Tests the dictionary of test results for each flow assessed to determine whether or not to continue the
        extraction. If all test results in the dictionary are FALSE, returns FALSE, if at least one of the results is
        TRUE, returns TRUE.
        :param threshold_test_dict: dictionary of test results for each flow assessed against the defined thresholds
        :return: True or False
        """
        if True in threshold_test_dict.values():
            return True
        else:
            return False

    def _store_pathway_in_temp(self, input_sector: int, target_sector: int, tech_coeff: float,
                               acc_tech_coeff: float,
                               total_coeff_dict: dict, stage: int, threshold_test_dict: dict):
        """
        Stores the pathway (nodes and transaction) to the temp_pathway
        :param input_sector: the input sector considered
        :param target_sector: the target sector considered
        :param tech_coeff: the technological coefficient of the sector
        :param acc_tech_coeff: the accumulated technological coefficient at this stage
        :param total_coeff_dict: a dictionary of total environmental coefficients for this node
        :param stage: the current stage
        :param threshold_test_dict: a dictionary of threshold testing for each flow
        :return:
        """
        direct_coeff_dict = {}
        for flow, test_result in threshold_test_dict.items():
            if test_result:
                direct_coeff_dict[flow] = acc_tech_coeff * self.dr_vectors_dict[flow][input_sector]
            else:
                direct_coeff_dict[flow] = None
        if self.nature == 'io' or self.nature == 'process':
            origin_node = Node(index_reference=input_sector,
                               stage=stage,
                               direct_intensities=direct_coeff_dict,
                               total_intensities=total_coeff_dict,
                               unit=self.sector_definition_dict[input_sector]['Unit'],
                               nature=self.nature)

        else:
            raise TypeError('The node to be added has no io or Process nature')

        # print('origin node: ', origin_node)
        if stage == 1:
            target_node = self.root_node
        else:
            if self.nature == 'io' or self.nature == 'process':
                target_node = Node(index_reference=target_sector,
                                   stage=stage - 1,
                                   direct_intensities=direct_coeff_dict,
                                   total_intensities=total_coeff_dict,
                                   unit=self.sector_definition_dict[input_sector]['Unit'],
                                   nature=self.nature)
            else:
                raise TypeError('The node to be added has no IO or Process nature')

        transaction = Transaction(origin_node, target_node, tech_coeff, acc_tech_coeff, self.nature)

        if self.temp_nodes.already_exists(target_node) is False:
            self.temp_nodes.append(target_node)
        self.temp_nodes.append(origin_node, target_node)
        self.temp_transactions.append(transaction)

    def complete_analysis_lci(self, path: str, filename: str, sector_selection: list = None,
                              as_absolute=True or False,
                              threshold_dict=None, max_stage=None):
        """
        Analyse the total contribution at each stage of the supply chain to the total requirement of every
        input-output sector or process contained in the database analysed, and copy the results to a csv file.
        :param path: directory of the file to be created (excluding file name).
        :param filename: name of the file to be created.
        :param sector_selection: selection of sector IDs to be assessed within the database - by default the entire
        database is analysed
        :param as_absolute: whether the results are provided as absolute values or as a percentage of the total.
        :param threshold_dict: dict of threshold for each indicator
        :param max_stage: maximum stage of analysis
        """

        if threshold_dict is None:
            threshold_dict = self.thresholds_dict
        else:
            threshold_dict = threshold_dict

        if max_stage is None:
            max_stage = self.max_stage
        else:
            max_stage = max_stage

        if sector_selection is None:
            sector_selection = list(self.sector_definition_dict.keys())
        else:
            sector_selection = sector_selection

        complete_analysis_dict = {
            flow: {
                self.sector_definition_dict[ids]['Name']: {}
                for ids in sector_selection}
            for flow in list(threshold_dict.keys())
        }

        complete_path = path + '\\' + 'ALL_SPAs_Threshold_' + str(threshold_dict[list(threshold_dict.keys())[0]])
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        else:
            pass

        list_files = os.listdir(complete_path)

        for ids in sector_selection:
            spa_filename = self.sector_definition_dict[ids]['Name'][:60] + '.sc'
            if spa_filename in list_files:
                pass
            else:
                self.target_ID = ids
                self.root_node = self._generate_root_node()
                self.pathways_list = PathwayList()
                self.extract_pathways(ids, threshold_dict, max_stage)
                self.save(complete_path + '\\' + spa_filename)
                for flow in self.chosen_flows_list:
                    total_dict = self.pathways_list.get_per_stage_results(self.root_node, max_stage,
                                                                          self.get_coverage_of(flow), flow,
                                                                          as_absolute)
                    total_dict['Pathways (Total)'] = self.get_number_of(count_what='pathways')
                    total_dict['Nodes (Total)'] = self.get_number_of(count_what='nodes')
                    # print(flow)
                    total_dict['Pathways (' + flow + ')'] = self.pathways_list.get_nb_pathways_per_flow([flow])[
                        flow]
                    complete_analysis_dict[flow][self.sector_definition_dict[ids]['Name']] = total_dict

        for flow in self.chosen_flows_list:
            threshold = str(threshold_dict[flow])
            df_output = pd.DataFrame.from_dict(complete_analysis_dict[flow])
            df_output.to_csv('%s\\%s_%s_%s.csv' % (path, flow, filename, threshold))


def _add_extraction_stage(stage):
    """
    Function used to create the extraction functions for every stage for which the SPA is conducted. The functions
    are dynamically created at the initialisation of the SupplyChain Class.
    :param stage: Stage of the supply chain at which the SPA is conducted/
    :return:
    """
    fn_name = '_extract_stage_%s_pathway' % str(stage)

    def fn_extraction_stage(self, target_sector, prev_tech_coeff=1.0, thresholds_dict=None):
        """
        Template for the pathway extraction function.
        :param self:
        :param target_sector: Sector considered. At Stage 1 it is the root sector, from the following stages it will be
        each of the input sector considered in Stage 1.
        :param prev_tech_coeff: The accumulated technological coefficient. It is equal to 1 at Stage 1, and will then
        evolve during the next stages up
        :param thresholds_dict: A dictionary of total requirement threshold.
        :return:
        """
        if thresholds_dict is None:
            thresholds_dict = self.thresholds_dict
        for input_sector in range(self.a_matrix.shape[0]):
            tech_coeff = self.a_matrix[input_sector][target_sector]
            # Tech_coeff must be declared independently as only used when storing the pathway
            acc_tech_coeff = prev_tech_coeff * tech_coeff
            # Acc_tech_coef is the technical coefficient downstream, relative to the stage considered
            # Prev_tech_coeff is the accumulated tech coefficient that is passed from the previous stage and ...
            # .. updated by the calculation above.
            total_requirements_dict = self._get_requirements_dict(input_sector, acc_tech_coeff, 'Total')
            threshold_test_dict = self._get_thresholds_test(total_requirements_dict)
            direct_requirement_dict = self._get_requirements_dict(input_sector, acc_tech_coeff, 'Direct')
            direct_requirement_test = \
                all(result == 0. for result in [direct_requirement_dict[flow] for flow in self.chosen_flows_list])
            if self._test_thresholds(threshold_test_dict):
                self._store_pathway_in_temp(input_sector, target_sector, tech_coeff, acc_tech_coeff,
                                            total_requirements_dict, stage, threshold_test_dict)
                if stage <= self.max_stage:
                    if not self._temp_nodes_and_temp_transactions_empty():
                        pathway = self._generate_pathway(self.temp_nodes, self.temp_transactions)
                        if not self._pathway_already_saved(
                                pathway.transactions[-1]) and not direct_requirement_test:
                            self.add_pathway(pathway)
                    if stage < self.max_stage:
                        eval('self._extract_stage_%s_pathway(input_sector, acc_tech_coeff, thresholds_dict)' %
                             str(stage + 1))
            else:
                continue

    setattr(SupplyChain, fn_name, fn_extraction_stage)


def get_spa(target_ID: int, max_stage: int, a_matrix_file_path: str, infosheet_file_path: str,
            thresholds_file_path: str, interactive=False) -> SupplyChain:
    """
    Main function in the module, runs a structural path analysis on the target sector/process ID in the specified files,
    according to the specified thresholds and up to the max_stage upstream in the supply chain
    :param target_ID: the numerial ID of the target sector/process on which to run the SPA
    :param max_stage: the maximum number of stages upstream
    :param a_matrix_file_path: the file path to the technological A matrix, csv or xls
    :param infosheet_file_path: the file path to the infoosheet, csv or xls
    :param thresholds_file_path: the file path to the thresholds, csv or xls
    :param interactive: Boolean that flags if the user wants the code to dynamically ask for input
    :return: A supply chain object
    """
    interface = Interface(a_matrix_file_path, infosheet_file_path, thresholds_file_path)
    sc = SupplyChain(target_ID=target_ID - 1,  # need to remove 1 to align with 0-indexing in python
                     sector_definition_dict=interface.infosheet_dataframe.to_dict(orient='index'),
                     a_matrix=interface.a_matrix,
                     dr_vectors_dict=interface.dr_vectors_dict,
                     tr_vectors_dict=interface.tr_vectors_dict,
                     thresholds_dict=interface.thresholds_dict,
                     flows_dict=interface.flows_dict,
                     max_stage=max_stage
                     )
    print('Supply Chain object created, extracting pathways, which will take some time...')
    start_time = datetime.datetime.now()
    print('Started at ' + start_time.strftime('%H:%M:%S'))
    sc.extract_pathways()
    end_time = datetime.datetime.now()
    print(
        'Ended at ' + end_time.strftime('%H:%M:%S') + '. It took ' + str((end_time - start_time).seconds) + ' seconds '
        + 'to extract ' + str(sc.get_number_of('pathways')) + ' pathways.')
    if interactive:
        export_to_csv = input('Export them to csv? (yes/no) >>> ')
        if export_to_csv.lower().strip() in ['yes', 'y', 'yeah']:
            csv_path = input('Please specify the full path of the csv file you want to export to >>> ')
            if csv_path.endswith('.csv'):
                try:
                    sc.export_to_csv(csv_path)
                except:
                    raise FileNotFoundError('There was an issue exporting the csv')
            else:
                raise FileNotFoundError('Please specify a csv file (explicitly write .csv at the end)')
    else:
        pass
    return sc
