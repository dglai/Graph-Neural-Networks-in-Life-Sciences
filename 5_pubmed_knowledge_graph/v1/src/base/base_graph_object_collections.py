"""
This module include the basic objects for lists of nodes and edges of specific types that make it easy to write it then into Neptune format.

Example on how to use them under the __main__ flag. 
"""

import warnings
import pandas as pd
from base.base_graph_objects import Node, Edge


def get_neptune_format(x):
    """
    TODO: Review best practices with gwprice@
    TODO: Futher optimize with more kinds of integers, doubles and dates.

    Given a value retrieves the string characterizing the Neptune format of this value from: Bool, Int, Double or String.

    String parameter will be returned when the value is not Bool, Int or Float.

    Args:
        x (object): Python object

    Return:
        str: Type of the object
    """
    #     out=None
    #     if isinstance(x, bool): out = 'Bool'
    #     elif isinstance(x, int): out = 'Int'
    #     elif isinstance(x, float): out = 'Double'
    #     else: out = 'String'

    #     if out is None: raise ValueError
    #     return out
    return "String"


class BaseList(object):
    def __init__(
        self,
        label: str,
        list_dicts: list = [],
        output_file: str = False,
        unique_attr="uid",
    ):
        """
        Construct the NodeList from a list of dictionaries.

        Args:

            label (str) : Name of the entity.

            nodes_data (list): List where each elements is a dictionary with:
                ```
                'uid': str
                'properties_dict': dictionary where keys are the names of the properties and values their actual value.
                ```
        Return:
            None:
        """
        self.label = label
        self.output_file = output_file
        self.unique_attr = unique_attr

        if self.unique_attr == "__all__":
            raise NotImplemented()

        else:
            self.elements = {
                element_data[self.unique_attr]: self._base_obj.construct_from_dict(
                    label, **element_data
                )
                for element_data in list_dicts
            }

    def __contains__(self, item):
        """
        OOP re-implementation of the `in` operation.
        """
        return item in self.elements

    def __getitem__(self, key):
        """
        OOP re-implementation of indexing.
        """
        return self.elements[key]

    def __iter__(
        self,
    ):
        """
        OOP re-implementation of iterating functionality.
        """
        self.n = 0
        self._nodes_dicts = list(self.values())
        return self

    def __next__(self):
        """
        OOP re-implementation of next object during the iteration functionality.
        """
        n = self.n
        self.n += 1
        return self._nodes_dicts[n]

    def keys(
        self,
    ):
        """
        OOP re-implementation of the keys of the objects.
        """
        return self.elements.keys()

    def values(
        self,
    ):
        """
        OOP re-implementation of the values of the objects.
        """
        return self.elements.values()

    def items(
        self,
    ):
        """
        OOP re-implementation of the items of the objects.
        """
        return self.elements.items()

    def __len__(self):
        """
        OOP re-implementation of the length of the objects.
        """
        return len(self.elements)

    def _get_ordered_prop(self, column_names):
        """
        Return properties (column names) from the Neptune column names


        """
        properties = [p.split(":")[0] if ":" in p else p for p in column_names]
        return properties

    @classmethod
    def construct_from_list_dicts(cls, label: str, list_dicts: list):
        """
        Construct the NodeList from a list of dictionaries.

        Args:

        label (str): str with the label type.
        nodes_data (list) : List where each elements is a dictionary with:
            ```

            'uid': str
            'properties_dict': dictionary where keys are the names of the properties and values their actual value.
            ```
        Return:
            None:
        """
        return cls(label, list_dicts)

    @classmethod
    def construct_from_df(cls, label: str, df: pd.DataFrame):
        """
        Construct the NodeList from a dictionary.

        Args:

            label (str): str with the label type.
            df (pd.DataFrame): Pandas dataframe with columns as node and edges properties. Those must include `uid` in case of nodes and `uid`, `from_node`, `to_node` in case of edges.

        Return:
            object: Instantiation of the object.
        """
        list_dicts = list(df.to_dict(orient="records").values())
        return cls(label, list_dicts)

    def _get_properties_sample(
        self,
    ):
        """
        Loop through nodes to find not-none observations of all properties.

        Args:

        Return:
            list: Return list of samples.
        """

        nodes = list(self.elements.values())

        properties = [p for p in nodes[0].keys() if p not in self._base_attributes]
        samples = {}

        for element in nodes:
            i = 0
            while i < len(properties):
                p = properties[i]
                if len(element.get_prop_value(p)) > 0:
                    samples[p] = element[p]
                    properties = properties[:i] + properties[i + 1 :]
                else:
                    i += 1
            if len(properties) == 0:
                return samples

        warnings.warn(
            f"Not a single not-None sample was found for the following properties: {properties}. Assigning value found at node in index 0."
        )

        for p in properties:
            samples[p] = nodes[0][p]

        return samples

    def _get_neptune_column_names(
        self,
    ):
        """
        Get mapping from property to neptune column name.

        Return:
            dict: mapping from property to neptune column name.
        """
        samples = self._get_properties_sample()
        properties = list(samples.keys())
        property2col_names = {p: p for p in self._base_attributes}

        property2col_names.update(
            {p: f"{p}:{get_neptune_format(samples[p])}" for p in properties}
        )
        #         col_names = list(self.base_attributes) + [f"{p}:{get_neptune_format(samples[p])}" for p in properties]
        return property2col_names

    def _to_csv(self, fpath: str):
        """
        Write the collection object into the csv format expected by Neptune's bulk upload.

        Args:
            fpath (str): file path where to write the collection class.

        Return:
            None:
        """
        assert fpath or self.output_file, f"You need to provide an output file"
        fpath = fpath if fpath else self.output_file
        neptune2col_names = self._get_neptune_column_names()
        records = [e.get_dict() for e in self.elements.values()]

        df = pd.DataFrame(records)
        df.rename(neptune2col_names, axis=1, inplace=True)
        df.to_csv(fpath, index=False)

    def record_object(self, element_dict):
        """
        Record a single object in the collect using the defining gym.

        Args:
            element_dict (dict): Dictionary defining the attributes of the object.

        Return:
            None:
        """
        self.elements.setdefault(
            element_dict[self.unique_attr],
            self._base_obj.construct_from_dict(self.label, **element_dict),
        )

    def record_list_of_objects(self, elements_list):
        """
        Expects a list of dictionaries defining the nodes.

        Args:
            elements_list (list): List of dictionaries defining the attributes of the object.

        Return:
            None:
        """
        for element_dict in elements_list:
            self.record_object(element_dict)

    def get_map_attribute2id(self, attr):
        """
        Create a map from an attribute `attr` to the node id.

        Note: if the attribute is not unique in the collection the mapping will be corrupted.

        Args:
            attr (str): Attribute name to use in the mapping.

        Return:
            dict: Dictionary from attribute to node id.
        """
        return {v[attr]: v["~id"] for v in self.elements.values()}


class NodeList(BaseList):
    """
    Class to hold all the nodes of a given type (label).

    This class assumes all the edges in a list have the same properties.
    """

    _base_attributes = set(["~id", "~label"])
    _base_obj = Node

    def __init__(
        self,
        node_label: str,
        list_dicts: list = [],
        output_file: str = False,
        unique_attr: str = "uid",
    ):
        """
        Construct the NodeList from a list of dictionaries.

        Args:

        label (str): str with the label type.
        nodes_data (list): List where each elements is a dictionary with:
            ```
            'uid': str
            'properties_dict': dictionary where keys are the names of the properties and values their actual value.
            ```

        Return:
            None:
        """
        super(NodeList, self).__init__(node_label, list_dicts, output_file, unique_attr)

    def to_csv(self, fpath: str = False):
        """
        Write the collection object into the csv format expected by Neptune's bulk upload.

        Args:
            fpath (str): file path where to write the collection class.

        Return:
            None:
        """
        assert len(self.elements) > 0, f"No nodes in NodeList to write"
        self._to_csv(fpath)


class EdgeList(BaseList):
    """
    Class to hold all edges of a given type (label).
    """

    _base_attributes = set(["~from", "~to", "~id", "~label"])
    _base_obj = Edge

    def __init__(
        self,
        edge_label: str,
        edges_data: list = [],
        output_file: str = False,
        record_by_orig_dest: bool = False,
        unique_attr: str = "uid",
    ):
        """
        This function assumes all the edges in a list have the same properties.

        Args:

        label (str): str with the label type.
        edges_data (list): List where each elements is a dictionary with:
        ```
            'uid': str
            'from_node': str
            'to_node': str
            'properties_dict': dictionary where keys are the names of the properties and values their actual value.
        ```
        output_file (str): Path to the csv to write the output. Can be provided at creation or when calling `to_csv()`.
        record_by_orig_dest(bool): Keep unique edges by origin and destination rather than edge id.
                                   Default to False making the recording (uniqueness) by uid.



        """
        super(EdgeList, self).__init__(edge_label, edges_data, output_file, unique_attr)

        if record_by_orig_dest:
            self.record_object = self._record_by_origin_dest

    def to_csv(self, fpath: str = False):
        """
        Write the collection object into the csv format expected by Neptune's bulk upload.

        Args:
            fpath (str): file path where to write the collection class.

        Return:
            None:
        """
        assert len(self.elements) > 0, f"No edges in EdgeList to write"
        self._to_csv(fpath)

    def _record_by_origin_dest(self, element_dict):
        """
        Record a single object in the collect using origin and destination of the edge.

        Args:
            element_dict (dict): Dictionary defining the attributes of the edge.

        Return:
            None:
        """
        k = (element_dict["from_node"], element_dict["to_node"])
        self.elements.setdefault(k, element_dict)


# Usage example
if __name__ == "__main__":

    label = "test"
    node_prop_example = [
        {
            "uid": 134,
            "prop_bool": True,
            "prop_int": 1,
            "prop_str": "test",
            "prop_float": 0.35,
        },
        {
            "uid": 2345,
            "prop_bool": False,
            "prop_int": 4,
            "prop_str": "train",
            "prop_float": 0.45,
        },
    ]

    #     test_nodes = nodeList(label, node_prop_example)
    #     test_nodes.to_csv('./test.csv')

    edge_prop_example = [
        {
            "uid": 134,
            "from_node": 123,
            "to_node": 345,
            "prop_bool": True,
            "prop_int": 1,
            "prop_str": "test",
            "prop_float": 0.35,
        },
        {
            "uid": 2345,
            "from_node": 345,
            "to_node": 678,
            "prop_bool": False,
            "prop_int": 4,
            "prop_str": "train",
            "prop_float": 0.45,
        },
    ]

    test_edges = edgeList(label, edge_prop_example)
    test_edges.to_csv("./test.csv")
